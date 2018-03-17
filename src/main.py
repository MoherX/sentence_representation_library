#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 上午8:55
# @Author  : yizhen
# @Site    : 
# @File    : main.py
# @Software: PyCharm
# function sentence_representation_library


import argparse
from datautils import *
from module import *
import logging
import codecs
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from MyDataset import MyDataset
from utils import flatten
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.utils.data as Data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')


def evaluate(instance_x, instance_y, model, batch_size):
    '''

    :param instance_x:
    :param instance_y:
    :param model:
    :return:
    '''
    dataset = MyDataset(instance_x, instance_y)

    dataloader = Data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=collate_batch)
    # start_id, end_id = 0, 0
    model.eval()
    # lst = list(range(len(instance_x)))
    gold_all, predict_all = [], []

    for step, (batch_instance_x, batch_instance_y) in enumerate(dataloader):
        model.eval()
        predict = model.forward(batch_instance_x, batch_instance_y)  # 进到forward的时候，顺序是变了,降序排列了

        lst = range(len(batch_instance_x))
        lst = sorted(lst, key=lambda d: -len(batch_instance_x[d]))
        batch_instance_y = [batch_instance_y[index] for index in lst]  # sorted by descend

        predict_all.append(predict.data.tolist())

        gold_all.append(batch_instance_y)

    gold = flatten(gold_all)
    predict = flatten(predict_all)

    sum_all = len(gold)
    correct = map(cmp, gold, predict).count(0)

    return correct * 1.0 / sum_all


def main():
    cmd = argparse.ArgumentParser("sentence_representation_library")
    cmd.add_argument("--data_dir", help='data_path', type=str, default='../data/')
    cmd.add_argument("--batch_size", help='batch_size', type=int, default=16)
    cmd.add_argument("--max_epoch", help='max_epoch', type=int, default=100)
    cmd.add_argument("--input_size", help='input_size', type=int, default=200)
    cmd.add_argument("--hidden_size", help='hidden_size', type=int, default=200)
    cmd.add_argument("--embedding_size", help='embedding_size', type=int, default=200)
    cmd.add_argument("--lr", help='lr', type=float, default=0.001)
    cmd.add_argument("--seed", help='seed', type=int, default=1)
    cmd.add_argument("--dropout", help="dropout", type=float, default=0.5)
    cmd.add_argument("--kernel_size", help="kernel_size", type=str, default="3*4*5")
    cmd.add_argument("--kernel_num", help="kernel_num", type=str, default="100*100*100")
    cmd.add_argument("--l2", help="l2 norm", type=int, default=3)
    cmd.add_argument("--encoder", help="options:[lstm, bilstm, gru, cnn, tri-lstm]", type=str, default='bilstm')
    cmd.add_argument("--gpu", action="store_true", help="use gpu")

    args = cmd.parse_args()
    torch.manual_seed(args.seed)  # fixed the seed
    random.seed(args.seed)

    use_cuda = torch.cuda.is_available() and args.gpu  # gpu use

    batch_size = args.batch_size
    logging.info("args:{0}".format(args))

    train_x, train_y, valid_x, valid_y, test_x, test_y = preprocess(args.data_dir)

    lang = Lang()
    lang.generate_dict(train_x)  # 构建train语料词典
    word_size = lang.get_word_size()  # 获取词典大小

    logging.info("dict generated finished! dict size:{0}".format(word_size))

    # 根据句子获得句子对应id序列，以及获取label对应id
    train_x_idx, train_y_idx = lang.sentence_to_idx2(train_x, train_y)
    valid_x_idx, valid_y_idx = lang.sentence_to_idx2(valid_x, valid_y)
    test_x_idx, test_y_idx = lang.sentence_to_idx2(test_x, test_y)

    logging.info(
            'train size:{0}, valid size:{1}, test size:{2}'.format(len(train_x_idx), len(valid_x_idx), len(test_x_idx)))

    model = Model(args, args.input_size, args.hidden_size, 5, word_size, args.embedding_size,
                  args.dropout)  # control module

    if use_cuda:
        model = model.cuda()

    dataset = MyDataset(train_x_idx, train_y_idx)

    dataloader = Data.DataLoader(dataset=dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_batch)  # self-defined collate function

    # 待会可以写个程序画出loss的曲线
    best_valid_acc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.max_epoch):
        round_loss = 0
        logging.info("epoch:{0} begins!".format(epoch))
        for step, (batch_instance_x, batch_instance_y) in enumerate(
                dataloader):  # 这里只是按照之前的进行自动shuffle，没有其它的要求，出来进行padding了和转换为Variable了

            optimizer.zero_grad()  # 梯度清零
            loss = model.forward(batch_instance_x, batch_instance_y)  # 送进去forward的只是原始的idx表示，还没有padding
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            round_loss += loss

        logging.info("epoch:{0} loss:{1}".format(epoch, round_loss.data[0]))

        valid_acc = evaluate(valid_x_idx, valid_y_idx, model, batch_size)
        logging.info("valid_acc = {0}".format(valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            test_acc = evaluate(test_x_idx, test_y_idx, model, batch_size)  # 在测试集上进行测试
            # save model
            logging.info(
                "epoch:{0} New Record! valid_accuracy:{1}, test_accuracy:{2}".format(epoch, valid_acc, test_acc))

    # finally, we evaluate valid and test dataset accuracy
    valid_acc = evaluate(valid_x_idx, valid_y_idx, model, batch_size)
    test_acc = evaluate(test_x_idx, test_y_idx, model, batch_size)

    logging.info("Train finished! saved model valid acc:{0}, test acc: {1}".format(valid_acc, test_acc))


if __name__ == '__main__':
    main()
