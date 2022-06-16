# -*- encoding: utf-8 -*-

import random
import time
from typing import List, Union
import os
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.autonotebook import trange
from transformers import BertConfig, BertModel, BertTokenizer, PreTrainedTokenizer

# 基本参数
from text2vec.similarity import cos_sim
from text2vec.text_matching_dataset import load_test_data
from text2vec.utils.stats_util import compute_spearmanr

EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-5
MAXLEN = 64
POOLING = 'cls'  # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 预训练模型目录
# BERT = 'pretrained_model/bert_pytorch'
# BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
# model_path = BERT
#
# model_path = '../pretrained_model/bert-base-chinese'
#
# # 微调后参数存放位置
SAVE_PATH = 'saved_model/'
#
# # 数据位置
# # SNIL_TRAIN = './dataset/cnsd-snli/train.txt'
# STS_TRAIN_LARGE = '../dataset/STS-B/sts-train.txt'
# STS_TRAIN_BASE = '../dataset/STS-B/sts-train-base.txt'
# STS_DEV = '../dataset/STS-B/sts-val.txt'
# STS_TEST = '../dataset/STS-B/sts-test.txt'


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集
    """

    # TODO: 把lqcmc的数据生成正负样本, 拿来做测试
    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            # return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]
            return [(line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]) for line in f]

    assert name in ["snli", "lqcmc", "sts"]
    if name == 'snli':
        return load_snli_data(path)
    return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path)


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data: List):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def text_2_id(self,  text: str):
        return self.tokenizer([text[0], text[1], text[2]], max_length=MAXLEN,
                         truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data: List):
        self.data = data
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=MAXLEN, truncation=True,
                         padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])


class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer =  BertTokenizer.from_pretrained(pretrained_model)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    def save_model(self, output_dir, results=None):
        """
        Saves the model to output_dir.
        :param output_dir:
        :param model:
        :param results:
        :return:
        """
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.bert.module if hasattr(self.bert, "module") else self.bert
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    # def get_sentence_embeddings(self):

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, show_progress_bar: bool = False):
        """
                Returns the embeddings for a batch of sentences.
                """
        self.bert.eval()
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Tokenize sentences
            inputs = self.tokenizer(sentences_batch, max_length=64, truncation=True,
                                     padding='max_length', return_tensors='pt')
            input_ids = inputs.get('input_ids').squeeze(1).to(DEVICE)
            attention_mask = inputs.get('attention_mask').squeeze(1).to(DEVICE)
            token_type_ids = inputs.get('token_type_ids').squeeze(1).to(DEVICE)

            # Compute sentences embeddings
            with torch.no_grad():
                # embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
                embeddings = self.forward(input_ids, attention_mask, token_type_ids)
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]

    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss




def eval(model, dataloader) -> float:
    """模型评估函数
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
            # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation




def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        results = {}
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                early_stop_batch = 0
                best = corrcoef
                results['spearman'] = corrcoef
                # torch.save(model.state_dict(), SAVE_PATH)
                model.save_model(SAVE_PATH, results=results)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue
            early_stop_batch += 1
            if early_stop_batch == 10:
                logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {(batch_idx - 10) * BATCH_SIZE}")
                return
        model.save_model(SAVE_PATH, results=results)

#
# if __name__ == '__main__':
#
#     logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
#     tokenizer = BertTokenizer.from_pretrained(model_path)
#
#     # load data
#     train_data = load_data('sts', STS_TRAIN_BASE)
#     random.shuffle(train_data)
#     dev_data = load_data('sts', STS_DEV)
#     test_data = load_data('sts', STS_TEST)
#     train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
#     dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=BATCH_SIZE)
#     test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE)
#
#     # load model
#     assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
#     model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
#     model.to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#
#     # train
#     best = 0
#     for epoch in range(EPOCHS):
#         logger.info(f'epoch: {epoch}')
#         train(model, train_dataloader, dev_dataloader, optimizer)
#     logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
#
#     # eval
#     test_data = load_test_data(STS_TEST)
#
#     srcs = []
#     trgs = []
#     labels = []
#     for terms in test_data:
#         src, trg, label = terms[0], terms[1], terms[2]
#         srcs.append(src)
#         trgs.append(trg)
#         labels.append(label)
#     logger.debug(f'{test_data[0]}')
#     # sentence_embeddings = model.encode(srcs)
#     # logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")
#
#
#     model = SimcseModel(pretrained_model=SAVE_PATH, pooling=POOLING)
#
#     t1 = time.time()
#     e1 = model.encode(srcs)
#     e2 = model.encode(trgs)
#     spend_time = time.time() - t1
#
#     s = cos_sim(e1, e2)
#     sims = []
#     for i in range(len(srcs)):
#         sims.append(s[i][i])
#     sims = np.array(sims)
#     labels = np.array(labels)
#     spearman = compute_spearmanr(labels, sims)
#
#     logger.debug(f'labels: {labels[:10]}')
#     logger.debug(f'preds:  {sims[:10]}')
#     logger.debug(f'Spearman: {spearman}')
#     logger.debug(
#         f'spend time: {spend_time:.4f}, count:{len(srcs + trgs)}, qps: {len(srcs + trgs) / spend_time}')
#     logger.debug(f'spearman: {spearman}')

    # dev_corrcoef = eval(model, dev_dataloader)
    # test_corrcoef = eval(model, test_dataloader)
    # logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    # logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
    # results['spearman'] = test_corrcoef
    #
    # model.save_model(SAVE_PATH, results=results)


    # with open(os.path.join(SAVE_PATH, "eval_results.txt"), "w") as writer:
    #     for key in sorted(results.keys()):
    #         writer.write("{} = {}\n".format(key, str(results[key])))
