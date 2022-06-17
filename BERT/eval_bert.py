# -*- coding: utf-8 -*-


import sys
import time
import os
import argparse
import numpy as np
from loguru import logger
from datasets import load_dataset
sys.path.append('..')

from text2vec.bertmatching_model import BertMatchModel
from text2vec.similarity import cos_sim, EncoderType
from text2vec.utils.stats_util import compute_spearmanr
from text2vec.text_matching_dataset import load_test_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Text Matching task')

    args = parser.parse_args()

    args.model_name = '../pretrained_model/bert-base-chinese/'  #预训练模型
    args.encoder_type = EncoderType['CLS']                      #pooling
    args.max_seq_length = 64                                    #文本最大分割长度
    args.output_dir = 'saved_model/'                            #模型保存目录
    args.num_epochs = 1                                         #训练epoch
    args.batch_size = 64
    args.learning_rate = 2e-5
    args.train_file = '../dataset/STS-B/sts-train.txt'
    args.valid_file = '../dataset/STS-B/sts-val.txt'
    args.test_file = '../dataset/STS-B/sts-test.txt'

    logger.info(args)

    model = BertMatchModel(model_name_or_path=args.model_name, encoder_type=args.encoder_type,
                           max_seq_length=args.max_seq_length, num_classes=6)
    model.train_model(args.train_file,
                      args.output_dir,
                      eval_file=args.valid_file,
                      num_epochs=args.num_epochs,
                      batch_size=args.batch_size,
                      lr=args.learning_rate)
    logger.info(f"Model saved to {args.output_dir}")

    model = BertMatchModel(model_name_or_path=args.output_dir, encoder_type=args.encoder_type,
                           max_seq_length=args.max_seq_length, num_classes=6)
    test_dataset = load_test_data(args.test_file)

    t1 = time.time()
    spearman, pearsonr = model.predict(test_dataset=test_dataset, batch_size=args.batch_size)
    spend_time = time.time() - t1
    logger.debug(
        f'spend time: {spend_time:.4f}')
    logger.debug(f'spearman: {spearman}, pearsonr:{pearsonr}')

