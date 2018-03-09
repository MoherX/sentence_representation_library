#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 下午12:28
# @Author  : yizhen
# @Site    : 
# @File    : module.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class lstm_model(nn.Module):
    def __init__(self, input_size_, hidden_size_, output_size, vocal_size, embedding_size):
        super(lstm_model, self).__init__()

        self.input_size = input_size_
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size_
        self.output_size = output_size
        self.vocal_size = vocal_size
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.embedding = nn.Embedding(self.vocal_size, self.embedding_size)
        self.NLLoss = nn.NLLLoss()
        self.softmax = nn.LogSoftmax()
        self.lstm  = nn.LSTM(input_size = self.input_size,
                             hidden_size = self.hidden_size,
                             batch_first=True)

    def forward(self, input_x, input_y):
        """
        intput_x: b_s instances
        :param input:
        :return:
        """
        input = input_x.squeeze(1)

        embed_input_x = self.embedding(input) # embed_intput_x: (b_s, m_l, em_s)


        encoders, (h_last, c_last) = self.lstm(embed_input_x) # encoders: [b_s, m_l, h_s], h_last: [b_s, h_s]


        predict = self.linear(h_last)  # output: [b_s, o_s]
        predict = self.softmax(predict.squeeze(0))

        loss = self.NLLoss(predict, input_y)

        if(self.training):  # if it is in training module
            return loss
        else:
            value, index = torch.max(predict, 1)
            return index  # outsize, cal the acc
