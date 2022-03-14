# -*- coding: utf-8 -*-
# @Time : 2021/5/25 8:20
# @Author : XXX
# @Site : 
# @File : tokenizer.py
# @Software: PyCharm
import jieba
import pandas as pd
import json
from tqdm import tqdm
from config import Config

config = Config()  # 配置类


def load_stopwords(stopwords_path):
    """
    加载停用词
    :return: list: 停用词列表
    """
    stop = open(stopwords_path, 'r+', encoding='utf-8')  # 打开停用词文件
    stopwords = stop.read().split("\n")  # 按‘\n’划分停用词
    return stopwords


def load_vocab(vocab_path):
    """
    加载语料库
    :param vocab_path: 语料库地址
    :return:
    """
    with open(vocab_path, 'r', encoding='utf-8') as fr:
        vocab = json.load(fr)
    return vocab


def build_vocab(file_path):
    """
    根据csv文件建立语料库
    :param file_path: 数据集文件地址
    :return:
    """
    vocab = {}  # 语料库字典
    stopwords = load_stopwords(config.stopwords_path)  # 停用词列表
    df = pd.read_csv(file_path)  # 打开csv文件
    for idx in tqdm(df.index):
        content = df.loc[idx, "review"].strip()  # 获取内容
        # 统计词频
        for word in jieba.cut(content):
            if word not in stopwords:
                vocab[word] = vocab.get(word, 0) + 1  # 若word在语料库中，则计数加一，若不在，初始化为1
    # 根据词频对词进行排序
    vocab = sorted(list(vocab.items()), key=lambda x: x[1], reverse=True)
    # 建立语料库（{word:id}）
    vocab = {word_count[0]: idx + 2 for idx, word_count in enumerate(vocab)}
    # UNK: 未知分词（即语料库中不存在的分词）， PAD: 填充标志
    vocab = {**{"[UNK]": 0, "[PAD]": 1}, **vocab}
    return vocab


class Tokenizer:

    def __init__(self, vocab_path, stopwords_path):
        self.stopwords = load_stopwords(stopwords_path)  # 停用词列表
        self.vocab = load_vocab(vocab_path)  # 语料库

    def tokenize(self, seq):
        """
        分词处理
        :param seq: 文本序列
        :return:list 分词列表
        """
        jieba.setLogLevel(20)
        tokens = jieba.cut(seq)  # 使用jieba库进行分词处理
        tokens = list(tokens)
        # 去除停用词
        for token in tokens:
            if token in self.stopwords:
                tokens.remove(token)
        return tokens

    def token_to_id(self, token):
        """
        将分词转化成id
        :param token: 分词
        :return: int id
        """
        if token in self.vocab.keys():  # 如果分词在语料库中
            return self.vocab[token]
        else:  # 分词不在语料库中
            return self.vocab['[UNK]']

    def seq_to_ids(self, seq, max_size):
        """
        将文本序列转换成id表示
        :param seq: 文本序列
        :param max_size: ids最大长度
        :return:
        """
        ids = []  # id列表
        tokens = self.tokenize(seq)  # 对文本序列进行分词处理
        #  将每个词转换成对应id
        for token in tokens:
            ids.append(self.token_to_id(token))
        # 长度与maxsize比较，多截少补
        if len(ids) < max_size:
            ids += [self.vocab['[PAD]']] * (max_size - len(ids))
        elif len(ids) > max_size:
            ids = ids[0: max_size]
        return ids


if __name__ == '__main__':
    # vocab = build_vocab("./data/weibo.csv")  # 根据数据集建立语料库
    # # 若建立的语料库不为空，将语料库保存至json文件
    # if vocab:
    #     with open("data/vocab.json", "w", encoding='utf-8') as f:
    #         json.dump(vocab, f)
    tokenizer = Tokenizer(config.vocab_path, config.stopwords_path)
    seq = "姑娘都羡慕你呢…还有招财猫高兴……//@爱在蔓延-JC:[哈哈]小学徒一枚，等着明天见您呢//@李欣芸SharonLee:大佬范儿[书呆子]"
    print(tokenizer.seq_to_ids(seq, 30))