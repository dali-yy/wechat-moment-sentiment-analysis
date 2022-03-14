import loguru
import pandas as pd
import torch

from model import Model
from model import pred
from moments import Moments
from tokenizer import Tokenizer
from config import Config

config = Config()  # 配置
model = Model()  # 定义模型
model.load_state_dict(torch.load(config.save_path))  # 加载模型


def senti_analyse(moments):
    """
    情感分析
    :return:
    """
    senti = {0: "negative", 1: "positive"}
    result = []  # 分析结果
    for moment in moments:
        y = pred(moment, model)
        result.append(senti[y])
    return result


def analyse_all(num):
    """
    分析所有
    :param num:
    :return:
    """
    m = Moments()  # 爬取朋友圈的类
    moments_dict = m.crawl_all(num)
    result_dict = {}
    loguru.logger.info("开始情感分析：")
    for nickname, moments in moments_dict.items():
        result_dict[nickname] = senti_analyse(moments)
    loguru.logger.info("情感分析结束！")
    loguru.logger.info("情感分析结果写入文件：")
    df = pd.DataFrame()
    nicknames = []
    moments = []
    sentiments = []
    for key in moments_dict.keys():
        nicknames.extend([key] * len(result_dict[key]))
        moments.extend(moments_dict[key])
        sentiments.extend(result_dict[key])
    df["Nickname"] = nicknames
    df["Moment"] = moments
    df["Sentiment"] = sentiments
    df.to_excel("./data/result.xlsx", index=False)


def analyse_one(nickname, num):
    """
    分析某个人
    :param nickname:
    :param num:
    :return:
    """
    m = Moments()  # 爬取朋友圈的类
    moments = m.crawl_one(nickname, num)
    loguru.logger.info("开始情感分析：")
    sentiments = senti_analyse(moments)
    loguru.logger.info("情感分析结束！")
    loguru.logger.info("情感分析结果写入文件：")
    df = pd.DataFrame()
    nicknames = [nickname] * len(moments)
    df["Nickname"] = nicknames
    df["Moment"] = moments
    df["Sentiment"] = sentiments
    df.to_excel("./data/" + nickname + "result.xlsx", index=False)


if __name__ == '__main__':
    analyse_all(10)
