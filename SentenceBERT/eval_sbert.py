# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import numpy as np
from loguru import logger
from datasets import load_dataset

sys.path.append('..')

from SentenceBERT import SentenceBertModel
from text2vec.similarity import cos_sim, EncoderType
from text2vec.utils.stats_util import compute_spearmanr
from text2vec.text_matching_dataset import load_test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Text Matching task')
    args = parser.parse_args()
    args.model_name = '../pretrained_model/bert-base-chinese/'  #预训练模型
    args.encoder_type = EncoderType['CLS']                      #pooling选择
    args.max_seq_length = 64                                    #文本最大分割长度
    args.output_dir = 'saved_model/'                            #训练模型保存目录
    args.num_epochs = 1
    args.batch_size = 64
    args.learning_rate = 2e-5
    args.train_file = '../dataset/STS-B/sts-train.txt'
    args.valid_file = '../dataset/STS-B/sts-val.txt'
    args.test_file = '../dataset/STS-B/sts-test.txt'

    logger.info(args)

    model = SentenceBertModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                              max_seq_length=args.max_seq_length)

    model.train_model(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        train_file=args.train_file,
        eval_file=args.valid_file
    )

    logger.info(f"Model saved to {args.output_dir}")

    model = SentenceBertModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                              max_seq_length=args.max_seq_length)

    test_data = load_test_data(args.test_file)

    srcs = []
    trgs = []
    labels = []
    for terms in test_data:
        src, trg, label = terms[0], terms[1], terms[2]
        srcs.append(src)
        trgs.append(trg)
        labels.append(label)
    logger.debug(f'{test_data[0]}')
    sentence_embeddings = model.encode(srcs)
    logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")


    t1 = time.time()
    e1 = model.encode(srcs)
    e2 = model.encode(trgs)
    spend_time = time.time() - t1

    s = cos_sim(e1, e2)
    sims = []
    for i in range(len(srcs)):
        sims.append(s[i][i])
    sims = np.array(sims)
    labels = np.array(labels)
    spearman = compute_spearmanr(labels, sims)

    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'preds:  {sims[:10]}')
    logger.debug(f'Spearman: {spearman}')
    logger.debug(
        f'spend time: {spend_time:.4f}, count:{len(srcs + trgs)}, qps: {len(srcs + trgs) / spend_time}')
    logger.debug(f'spearman: {spearman}')
