
import numpy as np
import os
import pickle
import torch
import random

from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.get_index(word)
        self.counter[token_id] += 1
        self.total += 1
        return self.get_index(word)

    def __len__(self):
        return len(self.idx2word)

    def get_index(self, word):
        return self.word2idx.get(word, 0)


class Corpus(object):
    def __init__(self, path=None, data_type=None):
        self.dictionary = Dictionary()
        self.train = None
        self.valid = None
        self.test = None