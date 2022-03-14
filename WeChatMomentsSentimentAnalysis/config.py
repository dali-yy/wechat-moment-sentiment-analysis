# -*- coding: utf-8 -*-
# @Time : 2021/5/25 8:46
# @Author : XXX
# @Site : 
# @File : config.py
# @Software: PyCharm
import torch


class Config:
    """
    配置类
    """

    def __init__(self):
        self.train_path = "./data/train.csv"  # 训练集地址
        self.test_path = "./data/test.csv"  # 测试集地址

        self.stopwords_path = "./data/stopwords.txt"  # 停用词文件地址
        self.vocab_path = "./data/vocab.json"  # 语料库文件地址
        self.vocab_size = 200471  # 语料库大小
        self.padding_size = 25  # 文本向量维度

        self.d_model = 200  # encoder输出向量维度
        self.h = 8  # 注意力的头数
        self.N = 1  # 注意力的层数
        self.d_ff = 800  # 全连接层维度
        self.dropout = 0.3  # 随机丢失率

        self.num_filters = 256  # 卷积核数量(channels数)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.embed = 200
        self.num_classes = 2  # 类别数

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU

        self.batch_size = 32  # 批次大小
        self.num_epochs = 10  # 训练次数

        self.save_path = "./checkpoint/model.ckpt"
