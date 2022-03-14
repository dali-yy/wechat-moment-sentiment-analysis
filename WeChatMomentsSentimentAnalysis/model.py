# -*- coding: utf-8 -*-
# @Time : 2021/5/24 19:14
# @Author : XXX
# @Site : 
# @File : model.py
# @Software: PyCharm
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from tokenizer import Tokenizer

config = Config()  # 配置类


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        """Implements Figure 2"""
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x):
        embeding = self.lut(x) * math.sqrt(self.d_model)
        return embeding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = Embeddings(config.vocab_size, config.d_model)
        self.attn = MultiHeadedAttention(config.h, config.d_model)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)
        out = self.attn(x, x, x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def pred(seq, model):
    """
    垃圾邮件预测
    :param seq:
    :param model:
    :return:
    """
    tokenizer = Tokenizer(config.vocab_path, config.stopwords_path)  # 分词处理器
    ids = tokenizer.seq_to_ids(seq, config.padding_size)  # 分词处理
    # 转化成模型输入的格式
    ids = np.array(ids).reshape(1, -1)
    x = torch.LongTensor(ids)
    # 预测
    y_ = model(x)
    y = y_.argmax().item()
    return y


if __name__ == "__main__":
    model = Model()  # 定义模型
    model.load_state_dict(torch.load(config.save_path))
    seq = "大半夜被@大?豆子 叫醒看外面下雪鸟[雪]这是北京这个冬季最大的一场雪[雪人][太开心]"
    print(pred(seq, model))
