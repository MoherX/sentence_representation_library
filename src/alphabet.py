# -*- coding: utf-8 -*-

import json
import os


class Alphabet:
    def __init__(self, name, label=False, keep_growing=True):
        self.__name = name
        self.UNKNOWN = "</unk>"  # 为登录标签</unk>
        self.PADDING = "</pad>"
        self.label = label
        self.instance2index = {}  # 字典
        self.instances = []
        self.keep_growing = keep_growing

        self.next_index = 0
        if not self.label:
            self.add(self.PADDING)
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        """
        重置 reset
        :param keep_growing:
        :return:
        """
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        self.next_index = 0

    def add(self, instance):
        """
        增加一个instance，并指定id
        :param instance:
        :return:
        """
        # 如果不在目前的词典之中
        if instance not in self.instance2index:
            self.instances.append(instance)  # 在列表后面添加该instance
            self.instance2index[instance] = self.next_index  # 在词典中添加。并指定id
            self.next_index += 1

    def get_index(self, instance):
        """
        找instance对应的id
        :param instance:
        :return:
        """
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:  # 如果keep growing，直接新增
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        """
        根据id找instance
        :param index:
        :return:
        """
        try:
            return self.instances[index]
        except IndexError:
            return self.instances[1]

    def size(self):
        return len(self.instances)

    def iteritems(self):
        return self.instance2index.iteritems()

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
