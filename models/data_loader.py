# -*- coding: utf-8 -*-
# @Author: Yue Peng
# @Email: yuepaang@gmail.com
import os
import sys
import codecs
import numpy as np
import pandas as pd
import torch
import jieba
from gensim.models import KeyedVectors
from torch.utils.data import TensorDataset, DataLoader, Dataset

# DOCKER data file
data_dir = "/data"

jieba.load_userdict(data_dir+"/cut_dict_uniq.txt")
STOPWORDS = [line.strip() for line in codecs.open(data_dir+"/stopwords_1009.txt", "r", "utf-8").readlines()]
# =================================
#  Char Data Loader (Embedding)
# =================================
def load_embedding():
    char_vectors = KeyedVectors.load_word2vec_format(data_dir+"/embedding_char_300.bin", binary=True)
    char2index = {}
    zeros = np.zeros(char_vectors.vectors.shape[1], dtype=np.float32)
    embedding = np.insert(char_vectors.vectors, 0, zeros, axis=0)
    print("Char Embedding: ", embedding.shape)
    padding_value = 0

    for i, w in enumerate(char_vectors.index2word):
        char2index[w] = i + 1
    return embedding, char2index, padding_value


class OnlineQA(Dataset):
    def __init__(self, max_len, data_fn, char2index):
        self.char2index = char2index
        self.max_len = max_len
        self.load(data_fn)
        self.y = torch.LongTensor(self.df["label"].tolist())

    def load(self, data_fn):
        self.df = pd.read_csv(data_dir+"/{}".format(data_fn)).reset_index(drop=True)
        self.label = pd.unique(self.df["label"])
        self.data = []
        for _, row in self.df.iterrows():
            text = row["query"]
            label = row["label"]
            # text = [c for c in text if '\u4e00' <= c <= '\u9fa5']
            # (str, int)
            self.data.append((text, label))

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        x_oh = self.one_hot(index) # (1, V)
        x_seq = self.seq(index) # (1, L)
        y = self.y[index]
        return x_oh, x_seq, y

    def __len__(self):
        return len(self.data)

    def one_hot(self, index):
        x_oh = torch.zeros(len(self.char2index))
        sequence = self.data[index][0]
        if len(sequence) > self.max_len:
            sequence = "".join([w for w in jieba.cut(sequence) if w not in STOPWORDS])
        for index_char, char in enumerate(sequence):
            if index_char == self.max_len:
                break
            try:
                x_oh[self.char2index[char]] = 1.0
            except KeyError:
                x_oh[0] = 1.0
        return x_oh

    def seq(self, index):
        """To Index
        
        Indices Sequence 
        
        Args:
            index: [description]
        
        Returns:
            [description]
            [type]
        """
        x_seq = []
        sequence = self.data[index][0]
        if len(sequence) > self.max_len:
            sequence = "".join([w for w in jieba.cut(sequence) if w not in STOPWORDS])
        for index_char, char in enumerate(sequence):
            if index_char == self.max_len:
                break
            try:
                x_seq.append(self.char2index[char])
            except KeyError:
                x_seq.append(0)
        if len(x_seq) < self.max_len:
            x_seq += [0] * (self.max_len - len(x_seq))
        x_seq = torch.tensor(x_seq[:self.max_len]).long()
        return x_seq

    def get_class_weight(self):
        num_samples = self.__len__()
        labels = self.df["label"].tolist()
        num_class = [labels.count(l) for l in self.label]
        class_weights = [num_samples/float(labels.count(l)) for l in self.label]
        return class_weights, num_class

