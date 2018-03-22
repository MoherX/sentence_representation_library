# coding=utf-8
import sys
import numpy as np
from alphabet import Alphabet
import cPickle as pickle

from datautils import normalize_word, read_instance, build_pretrain_embedding

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"


class Data:
    def __init__(self, args, use_char, use_gpu):
        self.number_normalized = True
        self.word_alphabet = Alphabet('word')  # 词语表
        self.char_alphabet = Alphabet('character')  # 字母表
        self.label_alphabet = Alphabet('label', True)  # 标记表

        self.char_features = args.char_encoder

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []

        self.input_size = 0

        self.word_emb_dim = args.embedding_size
        self.char_emb_dim = args.char_embedding_size
        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        # hyper parameters 超参数
        self.HP_iteration = args.max_epoch
        self.HP_batch_size = args.batch_size
        # self.HP_average_batch_loss = False
        self.HP_char_hidden_dim = args.char_hidden_dim
        self.HP_hidden_dim = args.hidden_size
        self.HP_dropout = args.dropout
        # self.HP_lstm_layer = 1
        # self.HP_bilstm = True
        self.HP_use_char = use_char
        self.HP_gpu = use_gpu
        self.HP_lr = args.lr
        # self.HP_lr_decay = 0.05
        # self.HP_clip = None
        # self.HP_momentum = 0
        self.HP_model_name = ""
        self.HP_encoder_type = ""
        self.HP_optim = ""

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Word  alphabet size: %s" % self.word_alphabet_size)
        print("     Char  alphabet size: %s" % self.char_alphabet_size)
        print("     Label alphabet size: %s" % self.label_alphabet_size)
        print("     Word embedding size: %s" % self.word_emb_dim)
        print("     Char embedding size: %s" % self.char_emb_dim)
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Hyper       iteration: %s" % self.HP_iteration)
        print("     Hyper      batch size: %s" % self.HP_batch_size)
        print("     Hyper              lr: %s" % self.HP_lr)
        print("     Hyper      hidden_dim: %s" % self.HP_hidden_dim)
        print("     Hyper         dropout: %s" % self.HP_dropout)
        print("     Hyper             GPU: %s" % self.HP_gpu)
        print("     Hyper        use_char: %s" % self.HP_use_char)
        if self.HP_use_char:
            print("             Char_features: %s" % self.char_features)
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            line = line.strip()
            if line:
                pairs = line.strip().split()
                label = pairs[0].strip()
                self.label_alphabet.add(label)
                for word in pairs[2:]:
                    if self.number_normalized:
                        word = normalize_word(word)
                    self.word_alphabet.add(word)
                    for char in word:
                        self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

    def extend_word_char_alphabet(self, input_file_list):
        """
        更新词语表和字母表
        :param
        :return:
        """
        old_word_size = self.word_alphabet_size
        old_char_size = self.char_alphabet_size
        for input_file in input_file_list:
            in_lines = open(input_file, 'r').readlines()
            for line in in_lines:
                line = line.strip()
                if line:
                    pairs = line.strip().split()
                    for word in pairs[2:]:
                        if self.number_normalized:
                            word = normalize_word(word)  # 如果单词中有数字，变为0
                        self.word_alphabet.add(word)
                        for char in word:
                            self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        print("Extend word/char alphabet finished!")
        print("     old word:%s -> new word:%s" % (old_word_size, self.word_alphabet_size))
        print("     old char:%s -> new char:%s" % (old_char_size, self.char_alphabet_size))
        for input_file in input_file_list:
            print("     from file:%s" % input_file)

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet,
                                                             self.label_alphabet, self.number_normalized)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet,
                                                         self.label_alphabet, self.number_normalized)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet,
                                                           self.label_alphabet, self.number_normalized)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % name)

    def build_word_pretrain_emb(self, emb_path):
        """
        预训练词向量
        :param emb_path:
        :return:
        """
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim)

    def build_char_pretrain_emb(self, emb_path):
        """

        :param emb_path:
        :return:
        """

        self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(emb_path, self.char_alphabet,
                                                                                   self.char_emb_dim)
