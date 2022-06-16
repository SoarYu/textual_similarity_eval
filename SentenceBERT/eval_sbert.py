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
    # parser.add_argument('--model_arch', default='cosent', const='cosent', nargs='?',
    #                     choices=['cosent', 'sentencebert', 'bert'], help='model architecture')
    # parser.add_argument('--task_name', default='STS-B', const='STS-B', nargs='?',
    #                     choices=['ATEC', 'STS-B', 'BQ', 'LCQMC', 'PAWSX'], help='task name of dataset')
    # parser.add_argument('--model_name', default='hfl/chinese-macbert-base', type=str,
    #                     help='Transformers model model or path')
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    # parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    # parser.add_argument('--output_dir', default='./outputs', type=str, help='Model output directory')
    # parser.add_argument('--max_seq_length', default=64, type=int, help='Max sequence length')
    # parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    # parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    # parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    # parser.add_argument('--encoder_type', default='FIRST_LAST_AVG', type=lambda t: EncoderType[t],
    #                     choices=list(EncoderType), help='Encoder type, string name of EncoderType')
    args = parser.parse_args()

    args.model_name = '../pretrained_model/bert-base-chinese/'
    args.encoder_type = EncoderType['CLS']
    args.max_seq_length = 64
    args.output_dir = 'saved_model/'
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


    # file_path = f'outputs/txt/15/{args.model_arch}_{args.encoder_type}_{args.task_name}_{args.num_epochs}.txt'
    # with open(file_path, 'a', encoding='utf-8') as file:
    #     file.write(f'{args.model_name}\tspearman: {spearman:.4f}\t\n')
    # return spearman

#
# if __name__ == '__main__':
#     main()
