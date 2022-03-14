# -*- coding: utf-8 -*-
# @Time : 2021/5/25 9:59
# @Author : XXX
# @Site : 
# @File : train.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time : 2021/5/24 10:00
# @Author : XXX
# @Site :
# @File : train.py
# @Software: PyCharm
import loguru
import torch
import time
import torch.nn.functional as F
import numpy as np

import data_utils
from config import Config
from model import Model
from tokenizer import Tokenizer

config = Config()  # 配置类


def train(model, device, train_loader, optimizer, epoch):
    """训练模型"""
    model.train()
    best_acc = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        start_time = time.time()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)  # 得到预测结果
        model.zero_grad()  # 梯度清零
        loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        if (batch_idx + 1) % 100 == 0:  # 打印loss
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))  # 记得为loss.item()


def test(model, device, test_loader):
    """测试模型, 得到测试集评估结果"""
    model.eval()
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x)
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)


if __name__ == "__main__":
    tokenizer = Tokenizer(config.vocab_path, config.stopwords_path)  # 分词处理器
    # 数据预处理
    loguru.logger.info("训练集预处理中...")
    input_ids_train, y_train = data_utils.data_preprocess(config.train_path)
    loguru.logger.info("测试集预处理中...")
    input_ids_test, y_test = data_utils.data_preprocess(config.test_path)
    # 数据迭代器
    train_loader = data_utils.get_data_loader(input_ids_train, y_train, config.batch_size)  # 训练集数据迭代器
    test_loader = data_utils.get_data_loader(input_ids_test, y_test, config.batch_size)  # 测试集数据迭代器

    # config.n_vocab = len(tokenizer.vocab)  # 获取语料库大小

    model = Model()  # 定义模型
    model = model.to(config.device)  # 在GPU上运行

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 优化器

    best_acc = 0.0
    for epoch in range(1, config.num_epochs + 1):  # 3个epoch
        train(model, config.device, train_loader, optimizer, epoch)
        acc = test(model, config.device, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), config.save_path)  # 保存最优模型
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
