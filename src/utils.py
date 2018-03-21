#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 下午2:56
# @Author  : yizhen
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

import itertools


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))



