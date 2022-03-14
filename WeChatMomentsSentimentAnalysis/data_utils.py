# -*- coding: utf-8 -*-
# @Time : 2021/5/25 8:20
# @Author : XXX
# @Site : 
# @File : data_process.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import tqdm
from config import Config
from tokenizer import Tokenizer

config = Config()  # 配置类


def train_test_split(file_path, train_path, test_path, train_radio):
    """
    划分训练集和测试集
    :param test_path: 划分后测试集文件存放的地址
    :param train_path: 划分后训练集文件存放的地址
    :param train_radio: 训练集所占比率
    :param file_path: 数据集文件地址
    :return:
    """
    df = pd.read_csv(file_path)  # 读取文件
    indices = np.arange(0, len(df))  # 索引
    np.random.shuffle(indices)  # 随机打乱索引

    df_train = df.iloc[indices[: int(len(df) * 0.8)]]  # 训练集
    df_test = df.iloc[indices[int(len(df) * 0.8):]]  # 测试集

    df_train.to_csv(train_path, index=False)  # 训练集写进文件
    df_test.to_csv(test_path, index=False)  # 测试集写进文件


def data_preprocess(file_path):
    """
    数据预处理，将文本转换成向量表示
    :param file_path:
    :return:
    """
    tokenizer = Tokenizer(config.vocab_path, config.stopwords_path)  # 分词器
    input_ids = []  # 输入模型的文本id表示
    labels = []  # 标签

    df = pd.read_csv(file_path)  # 读取文件
    #  将文本转换成向量表示
    for idx in tqdm(df.index):
        text_id = tokenizer.seq_to_ids(df.loc[idx, "review"], config.padding_size)
        input_ids.append(text_id)
        labels.append([int(df.loc[idx, "label"])])
    # 转换成np.array格式的
    input_ids = np.array(input_ids, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    return input_ids, labels


def get_data_loader(input_ids, labels, batch_size):
    """
    数据迭代器
    :param input_ids:
    :param labels:
    :param batch_size: 批次大小
    :return:
    """
    # 包装成数据集
    dataset = TensorDataset(torch.LongTensor(input_ids),
                            torch.LongTensor(labels))
    data_sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return data_loader


if __name__ == "__main__":
    # train_test_split("./data/weibo.csv", config.train_path, config.test_path, 0.8)  # 划分训练集和测试集
    input_ids, labels = data_preprocess("./data/test.csv")
    print(input_ids)
