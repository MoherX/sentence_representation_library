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
from ModelFactory import ModelFactory
from utils import flatten
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from data import Data
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')


def evaluate(ids, model, batch_size):
    '''

    :param instance_x:
    :param instance_y:
    :param model:
    :return:
    '''
    dataset = MyDataset(ids)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    # start_id, end_id = 0, 0
    model.eval()
    # lst = list(range(len(instance_x)))
    gold_all, predict_all = [], []

    for step, (batch_words, batch_chars, batch_label) in enumerate(dataloader):
        model.eval()
        predict = model.forward(batch_words, batch_chars, batch_label)  # 进到forward的时候，顺序是变了,降序排列了

        lst = range(len(batch_words))
        lst = sorted(lst, key=lambda d: -len(batch_words[d]))
        batch_instance_y = [batch_label[index] for index in lst]  # sorted by descend

        predict_all.append(predict.data.tolist())

        gold_all.append(batch_instance_y)

    gold = flatten(gold_all)
    predict = flatten(predict_all)

    sum_all = len(gold)
    correct = map(cmp, gold, predict).count(0)

    return correct * 1.0 / sum_all


def main():
    cmd = argparse.ArgumentParser("sentence_representation_library")
    # dataset
    cmd.add_argument("--train", help='train data_path', type=str, default='../data2/train.txt')
    cmd.add_argument("--dev", help='dev data_path', type=str, default='../data2/valid.txt')
    cmd.add_argument("--test", help='test data_path', type=str, default='../data2/test.txt')
    cmd.add_argument("--batch_size", help='batch_size', type=int, default=16)
    cmd.add_argument("--max_epoch", help='max_epoch', type=int, default=100)
    cmd.add_argument("--hidden_size", help='hidden_size', type=int, default=200)
    cmd.add_argument("--embedding_size", help='embedding_size', type=int, default=200)
    cmd.add_argument("--embedding_path", default="", help="pre-trained embedding path")
    cmd.add_argument("--lr", help='lr', type=float, default=0.001)
    cmd.add_argument("--seed", help='seed', type=int, default=1)
    cmd.add_argument("--dropout", help="dropout", type=float, default=0.5)
    cmd.add_argument("--kernel_size", help="kernel_size", type=str, default="3*4*5")
    cmd.add_argument("--kernel_num", help="kernel_num", type=str, default="100*100*100")
    cmd.add_argument("--l2", help="l2 norm", type=int, default=3)
    cmd.add_argument("--encoder", help="options:[lstm, bilstm, gru, cnn, tri-lstm, sum]", type=str, default='bilstm')
    cmd.add_argument("--gpu", action="store_true", help="use gpu")
    # character
    cmd.add_argument("--char_encoder", help="options:[bilstm, cnn]", type=str, default='bilstm')
    # char lstm
    cmd.add_argument("--char_hidden_dim", help="char_hidden_dim", type=int, default=50)
    cmd.add_argument("--char_embedding_path", help='char_embedding_path', default="")
    cmd.add_argument("--char_embedding_size", help='char_embedding_size', type=int, default=50)

    args = cmd.parse_args()
    torch.manual_seed(args.seed)  # fixed the seed
    random.seed(args.seed)

    use_cuda = torch.cuda.is_available() and args.gpu  # gpu use

    if args.char_encoder:
        use_char = True
    else:
        use_char = False

    data = Data(args, use_cuda, use_char)
    data.number_normalized = True  # 替换数字为0
    # 构建词语、字符、标签词典
    data.build_alphabet(args.train)
    # data.build_alphabet(args.dev)
    # data.build_alphabet(args.test)
    data.fix_alphabet()
    # 准备数据
    data.generate_instance(args.train, 'train')
    data.generate_instance(args.dev, 'dev')
    data.generate_instance(args.test, 'test')

    # 加载预训练向量
    if args.embedding_path:
        data.build_word_pretrain_emb(args.embedding_path)
    if args.char_embedding_path:
        data.build_char_pretrain_emb(args.char_embedding_path)

    # 创建工厂 根据参数实例化对应model
    factory = ModelFactory()

    # 根据是否使用character表示以及类型来改变输入dim
    # if use_char and args.char_encoder is "bilstm":
    #     input_size = data.word_emb_dim + 2 * data.HP_char_hidden_dim
    # else:
    input_size = data.word_emb_dim

    print data.word_alphabet.size()
    print data.word_alphabet.instance2index
    print data.label_alphabet.instance2index

    model = factory.get_model(args.encoder, args, input_size, data.HP_hidden_dim, data.label_alphabet_size,
                              data.word_alphabet.size(), data.word_emb_dim, data.HP_dropout,
                              data.pretrain_word_embedding)

    if use_cuda:
        model = model.cuda()

    dataset = MyDataset(data.train_Ids)

    dataloader = DataLoader(dataset=dataset, batch_size=data.HP_batch_size, shuffle=True,
                            collate_fn=collate_batch)

    # 待会可以写个程序画出loss的曲线
    best_valid_acc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=data.HP_lr)

    model.train()
    data.show_data_summary()
    for epoch in range(data.HP_iteration):
        round_loss = 0
        logging.info("epoch:{0} begins!".format(epoch))
        for step, (batch_words, batch_chars, batch_label) in enumerate(
                dataloader):  # 这里只是按照之前的进行自动shuffle，没有其它的要求，出来进行padding了和转换为Variable了

            model.train()
            optimizer.zero_grad()  # 梯度清零
            loss = model.forward(batch_words, batch_chars, batch_label)  # 送进去forward的只是原始的idx表示，还没有padding
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            round_loss += loss

        logging.info("epoch:{0} loss:{1}".format(epoch, round_loss.data[0]))

        valid_acc = evaluate(data.dev_Ids, model, data.HP_batch_size)
        logging.info("valid_acc = {0}".format(valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            test_acc = evaluate(data.test_Ids, model, data.HP_batch_size)  # 在测试集上进行测试
            # save model
            logging.info(
                "epoch:{0} New Record! valid_accuracy:{1}, test_accuracy:{2}".format(epoch, valid_acc, test_acc))

    # finally, we evaluate valid and test dataset accuracy
    valid_acc = evaluate(data.dev_Ids, model, data.HP_batch_size)
    test_acc = evaluate(data.test_Ids, model, data.HP_batch_size)

    logging.info("Train finished! saved model valid acc:{0}, test acc: {1}".format(valid_acc, test_acc))


if __name__ == '__main__':
    main()
