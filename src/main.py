#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 上午8:55
# @Author  : yizhen
# @Site    : 
# @File    : main.py
# @Software: PyCharm
# function sentence_representation_library


import argparse

from visdom import Visdom

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
    cmd.add_argument("--train", help='train data_path', type=str, default='../data/train.txt')
    cmd.add_argument("--dev", help='dev data_path', type=str, default='../data/valid.txt')
    cmd.add_argument("--test", help='test data_path', type=str, default='../data/test.txt')
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
    cmd.add_argument("--model_name", default="sr", help="model name")
    cmd.add_argument("--optim", default="Adam", help="options:[Adam,SGD]")
    cmd.add_argument("--load_model", default="", help="model path")
    # character
    cmd.add_argument("--char_encoder", help="options:[bilstm, cnn]", type=str, default='bilstm')
    # char lstm
    cmd.add_argument("--char_hidden_dim", help="char_hidden_dim", type=int, default=50)
    cmd.add_argument("--char_embedding_path", help='char_embedding_path', default="")
    cmd.add_argument("--char_embedding_size", help='char_embedding_size', type=int, default=50)

    args = cmd.parse_args()

    # fixed the seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # gpu use
    use_cuda = torch.cuda.is_available() and args.gpu

    if args.char_encoder:
        use_char = True
    else:
        use_char = False

    data = Data(args, use_cuda, use_char)

    # set some hyper parameters
    data.number_normalized = True  # replace all the number with zero
    data.HP_encoder_type = args.encoder  # encode type
    data.HP_model_name = args.model_name  # model name
    data.HP_optim = args.optim  # SGD or Adam

    # build word character label alphabet
    data.build_alphabet(args.train)
    data.fix_alphabet()

    # prepare data
    data.generate_instance(args.train, 'train')
    data.generate_instance(args.dev, 'dev')
    data.generate_instance(args.test, 'test')

    # load pre-trained embedding(if not,we random init the embedding using nn.Embedding())
    if args.embedding_path:
        data.build_word_pretrain_emb(args.embedding_path)
    if args.char_embedding_path:
        data.build_char_pretrain_emb(args.char_embedding_path)

    # create visdom enviroment
    vis = Visdom(env=data.HP_model_name)

    # if use_char and args.char_encoder is "bilstm":
    #     input_size = data.word_emb_dim + 2 * data.HP_char_hidden_dim
    # else:
    data.input_size = data.HP_word_emb_dim

    # create factory and type create the model according to the encoder
    factory = ModelFactory()
    model = factory.get_model(data, args)

    # load model
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

    if use_cuda:
        model = model.cuda()

    # Dataset、DataLoader for batch
    dataset = MyDataset(data.train_Ids)
    dataloader = DataLoader(dataset=dataset, batch_size=data.HP_batch_size, shuffle=True, collate_fn=collate_batch)

    best_valid_acc = 0.0

    # optimizer
    if data.HP_optim.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr)
    elif data.HP_optim.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr)

    model.train()
    data.show_data_summary()  # show information about the hyper parameters and some datas
    for epoch in range(data.HP_iteration):
        round_loss = 0
        logging.info("epoch:{0} begins!".format(epoch))
        for step, (batch_words, batch_chars, batch_label) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()  # zero grad
            loss = model.forward(batch_words, batch_chars, batch_label)
            loss.backward()  # back propagation
            optimizer.step()  # update parameters
            round_loss += loss  # the sum of the each epoch`s loss

        logging.info("epoch:{0} loss:{1}".format(epoch, round_loss.data[0]))

        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor(round_loss.data), win='loss',
                 update='append' if epoch > 0 else None)

        # use current model to test on the dev set
        valid_acc = evaluate(data.dev_Ids, model, data.HP_batch_size)
        logging.info("valid_acc = {0}".format(valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            # test on the test set
            test_acc = evaluate(data.test_Ids, model, data.HP_batch_size)

            vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([test_acc]), win='test_acc',
                     update='append' if epoch > 0 else None)
            # save model
            torch.save(model.state_dict(), "../model/" + data.HP_model_name + ".model")
            logging.info(
                "epoch:{0} New Record! valid_accuracy:{1}, test_accuracy:{2}".format(epoch, valid_acc, test_acc))

    # finally, we evaluate valid and test dataset accuracy
    valid_acc = evaluate(data.dev_Ids, model, data.HP_batch_size)
    test_acc = evaluate(data.test_Ids, model, data.HP_batch_size)

    logging.info("Train finished! saved model valid acc:{0}, test acc: {1}".format(valid_acc, test_acc))


if __name__ == '__main__':
    main()
