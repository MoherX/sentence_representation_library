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
from utils import flatten
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.utils.data as Data

use_cuda = torch.cuda.is_available()
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
    # for start_id in range(0, len(instance_x), batch_size):
    #     end_id = start_id + batch_size if start_id + batch_size < len(instance_x) else len(instance_x)
    #     batch_lst = lst[start_id:end_id]
    #     batch_instance_x, batch_instance_y = get_batch(instance_x, instance_y, batch_size, batch_lst)
        # batch_instance_x, b_s个instance
        batch_instance_x, batch_instance_y = padding(batch_instance_x, batch_instance_y)
        model.eval()
        predict = model.forward(batch_instance_x, batch_instance_y)
        predict_all.append(predict.data.tolist())

        gold_all.append(batch_instance_y.data.tolist())

    gold = flatten(gold_all)
    predict = flatten(predict_all)

    sum_all = len(gold)
    correct = map(cmp, gold, predict).count(0)

    return correct*1.0 / sum_all



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

    args = cmd.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    batch_size = args.batch_size
    logging.info("args:{0}".format(args))
    train_x, train_y, valid_x, valid_y, test_x, test_y = preprocess(args.data_dir)

    lang = Lang()
    lang = generate_dict(lang, train_x)
    word_size = lang.get_word_size()

    logging.info("dict generated finished! dict size:{0}".format(word_size))

    train_x_idx, train_y_idx = sentence_to_idx(lang, test_x, test_y)
    valid_x_idx, valid_y_idx = sentence_to_idx(lang, test_x, test_y)
    test_x_idx, test_y_idx = sentence_to_idx(lang, test_x, test_y)

    logging.info('train size:{0}, valid size:{1}, test size:{2}'.format(len(train_x_idx), len(valid_x_idx), len(test_x_idx)))
    lst = list(range(len(train_x_idx)))

    model = lstm_model(args.input_size, args.hidden_size, 5, word_size, args.embedding_size, args.dropout)

    if use_cuda:
        model = model.cuda()


    # dataset = Data.TensorDataset(data_tensor=train_x_idx, target_tensor=train_y_idx)
    dataset = MyDataset(train_x_idx, train_y_idx)

    dataloader = Data.DataLoader(dataset = dataset,
                                 batch_size = args.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_batch)
    # dataloader = Data.DataLoader(dataset=dataset,
    #                              batch_size=args.batch_size,
    #                              shuffle=True)


    # 待会可以写个程序画出loss的曲线
    best_valid_acc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.max_epoch):
        Round_loss = 0
        # final_test_acc = 0
        # best_valid_acc = 0.0
        logging.info("epoch:{0} begins!".format(epoch))
        # start_id, end_id = 0, 0 # order from start_id to end_id
        # sort or not sort
        for step, (batch_instance_x, batch_instance_y) in enumerate(dataloader): # 这里只是按照之前的进行自动shuffle，没有其它的要求，出来进行padding了和转换为Variable了
        #for start_id in range(0, len(train_x_idx), batch_size):
            batch_instance_x, batch_instance_y = padding(batch_instance_x, batch_instance_y)  # we padding those
            optimizer.zero_grad()

            # end_id = start_id + batch_size if start_id + batch_size < len(train_x_idx) else len(train_x_idx)
            # batch_lst = lst[start_id:end_id]
            # batch_instance_x, batch_instance_y = get_batch(train_x_idx, train_y_idx, batch_size, batch_lst)
            # batch_instance_x, batch_instance_y = dataloader(dataset)
            # batch_instance_x, b_s个instance
            model.train()

            loss = model.forward(batch_instance_x, batch_instance_y)

            loss.backward()
            optimizer.step()
            Round_loss += loss
        logging.info("epoch:{0} loss:{1}".format(epoch, Round_loss.data[0]))

        valid_acc = evaluate(valid_x_idx, valid_y_idx, model, batch_size)
        logging.info("valid_acc = {0}".format(valid_acc))

        if(valid_acc > best_valid_acc):
            best_valid_acc = valid_acc
            test_acc = evaluate(test_x_idx, test_y_idx, model, batch_size)  # 在测试集上进行测试

            logging.info("epoch:{0} New Record! valid_accuracy:{1}, test_accuracy:{2}".format(epoch, valid_acc, test_acc))

    # finally, we evaluate valid and test dataset accuracy
    valid_acc = evaluate(valid_x_idx, valid_y_idx, model, batch_size)
    test_acc = evaluate(test_x_idx, test_y_idx, model, batch_size)

    logging.info("Train finished! saved model valid acc:{0}, test acc: {1}".format(valid_acc, test_acc))


if __name__ == '__main__':
    main()



