#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 上午9:03
# @Author  : yizhen
# @Site    :
# @File    : datautils.py
# @Software: PyCharm

import codecs
import os

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

use_cuda = torch.cuda.is_available()


def collate_batch(batch):
    outputs_words = []
    outputs_chars = []
    outputs_lables = []
    for key in batch:
        outputs_words.append(key[0])
        outputs_chars.append(key[1])
        for label in key[2]:
            outputs_lables.append(label)
    return outputs_words, outputs_chars, outputs_lables


def padding(instance_x, batch_chars, instance_y):
    '''
    return padded data
    :param instance_x:  []
    :param instance_y:  []
    :return:
    '''
    # lst = sorted(lst, lambda )
    lst = range(len(instance_x))
    # 按照长度排序
    lst = sorted(lst, key=lambda d: -len(instance_x[d]))
    # 重新排序过后的
    instance_x_sorted = [instance_x[index] for index in lst]  # be sorted in decreasing order for packed
    instance_y_sorted = [instance_y[index] for index in lst]
    # 记录padding之前的长度
    sentence_lens = [len(sentence) for sentence in instance_x_sorted]  # for pack-padded deal
    max_len = max(sentence_lens)
    # 根据词典，使用1来进行padding
    instance_x_sorted = [sentence + (max_len - len(sentence)) * [0] for sentence in instance_x_sorted]

    # print instance_x_sorted
    # print instance_y_sorted

    batch_chars_sorted = [batch_chars[index] for index in lst]  # 首先根据句子长度进行交换
    # print batch_chars

    words_lens = []
    character_padding_res = []
    # 对character级别进行padding
    for index, sentence in enumerate(batch_chars_sorted):
        # print sentence
        c_lst = range(len(sentence))
        # print c_lst
        c_lst = sorted(c_lst, key=lambda d: -len(sentence[d]))
        # print c_lst
        sentence_sorted = [sentence[index] for index in c_lst]
        # print sentence_sorted
        words_len = [len(word) for word in sentence_sorted]
        # print words_len
        words_lens.append(words_len)
        # print words_lens
        max_word_len = max(words_len)
        # print max_word_len
        sentence_sorted = [word + (max_word_len - len(word)) * [0] for word in sentence_sorted]
        # print sentence_sorted
        character_padding_res.append(sentence_sorted)

    return instance_x_sorted, character_padding_res, instance_y_sorted, sentence_lens, words_lens


def normalize_word(word):
    """
    讲英语单词中的数字全部变为0
    :param word:
    :return:
    """
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized):
    in_lines = open(input_file, 'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        line = line.strip()
        if line:
            pairs = line.strip().split()
            label = pairs[0]
            labels.append(label)
            label_Ids.append(label_alphabet.get_index(label))
            for word in pairs[2:]:
                if number_normalized:
                    word = normalize_word(word)
                words.append(word)
                word_Ids.append(word_alphabet.get_index(word))
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            instence_texts.append([words, chars, labels])
            instence_Ids.append([word_Ids, char_Ids, label_Ids])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            # 除去最开始的单词
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0].decode('utf-8')] = embedd
    return embedd_dict, embedd_dim


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    """
    构建预训练向量
    :param embedding_path:
    :param word_alphabet:
    :param embedd_dim:
    :param norm:
    :return:
    """
    embedd_dict = dict()

    # 加载文件中有的预训练向量
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0

    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embedd_dim
