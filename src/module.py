#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 下午12:28
# @Author  : yizhen
# @Site    : 
# @File    : module.py
# @Software: PyCharm

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from datautils import padding
import torch.nn.functional as F
from torch import optim
import numpy as np
import random

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
use_cuda = torch.cuda.is_available()

class Model(nn.Module):
    def __init__(self, args, input_size_, hidden_size_, output_size, vocal_size, embedding_size, dropout):
        super(Model, self).__init__()
        self.input_size = input_size_
        self.hidden_size = hidden_size_
        self.output_size = output_size
        self.vocal_size = vocal_size
        self.embedding_size =embedding_size
        self.args = args

        if args.encoder == 'lstm':
            self.encoder = lstm_model(self.args, self.input_size, self.hidden_size, self.output_size, self.vocal_size, self.embedding_size, dropout)

    def forward(self, input_x, input_y):
        """
         intput_x: b_s instances， 没有进行padding和Variable
        :param input_x:
        :param input_y:
        :return:
        """
        return self.encoder.forward(input_x, input_y)  # interface， implementated by every people


class lstm_model(nn.Module):
    def __init__(self, args, input_size_, hidden_size_, output_size, vocal_size, embedding_size, dropout):
        super(lstm_model, self).__init__()

        self.input_size = input_size_
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size_
        self.output_size = output_size
        self.vocal_size = vocal_size
        self.seed = args.seed
        torch.manual_seed(self.seed)  # fixed the seed
        random.seed(self.seed)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.embedding = nn.Embedding(self.vocal_size, self.embedding_size)
        self.NLLoss = nn.NLLLoss()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax()
        self.lstm  = nn.LSTM(input_size = self.input_size,
                             hidden_size = self.hidden_size,
                             batch_first=True,
                             dropout = dropout)

        self.w_i_in, self.w_i_on = self.lstm.all_weights[0][0].size()
        self.w_h_in, self.w_h_on = self.lstm.all_weights[0][1].size()
        self.lstm.all_weights[0][0] = Parameter(torch.randn(self.w_i_in, self.w_i_on)) * np.sqrt(2./ self.w_i_on)
        self.lstm.all_weights[0][1] = Parameter(torch.randn(self.w_h_in, self.w_h_on)) * np.sqrt(2. / self.w_h_on)

    def forward(self, input_x, input_y):
        """
        intput_x: b_s instances， 没有进行padding和Variable
        :param input:
        :return:
        """
        # input = input_x.squeeze(1)

        #
        input_x, input_y, sentence_lens = padding(input_x, input_y)
        max_len = len(input_x[0])

        if use_cuda:
            input_x = Variable(torch.LongTensor(input_x)).cuda()
            input_y = Variable(torch.LongTensor(input_y)).cuda()
        else:
            input_x = Variable(torch.LongTensor(input_x))
            input_y = Variable(torch.LongTensor(input_y))

        input = input_x.squeeze(1)

        embed_input_x = self.embedding(input) # embed_intput_x: (b_s, m_l, em_s)
        embed_input_x = self.dropout(embed_input_x)


        embed_input_x_packed = pack_padded_sequence(embed_input_x, sentence_lens, batch_first=True)
        encoder_outputs_packed, (h_last, c_last) = self.lstm(embed_input_x_packed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)


        predict = self.linear(h_last)  # predict: [1, b_s, o_s]
        predict = self.softmax(predict.squeeze(0)) # predict.squeeze(0) [b_s, o_s]

        loss = self.NLLoss(predict, input_y)

        if(self.training):  # if it is in training module
            return loss
        else:
            value, index = torch.max(predict, 1)
            return index  # outsize, cal the acc
