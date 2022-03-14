# -*- coding: utf-8 -*-
# @Time : 2021/5/25 14:01
# @Author : XXX
# @Site :
# @File : moments.py
# @Software: PyCharm
import json
import time

import loguru
from appium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from collections import defaultdict


class Moments(object):
    def __init__(self):
        # 驱动配置
        server = "http://localhost:4723/wd/hub"
        desired_caps = {
            "platformName": "Android",
            "deviceName": "PYB0220918005498",
            "appPackage": "com.tencent.mm",
            'appActivity': 'com.tencent.mm.ui.LauncherUI',  # apk的launcherActivity
            'noReset': True,  # 每次运行脚本不用重复输入密码启动微信
            'unicodeKeyboard': True,  # 使用unicodeKeyboard的编码方式来发送字符串
        }
        self.driver = webdriver.Remote(server, desired_capabilities=desired_caps)
        self.wait = WebDriverWait(self.driver, 30)

    def enter_friends_circle(self):
        """
        进入朋友圈
        :return:
        """
        loguru.logger.info("点击发现：")
        finds = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/dtf")))
        finds[2].click()
        loguru.logger.info("进入朋友圈：")
        m_btn = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/h6o")))
        m_btn[0].click()

    def get_all_friends(self):
        """
        获取好友列表
        :return:
        """
        loguru.logger.info("点击好友列表：")
        finds = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/dtf")))
        finds[1].click()

        flick_start_x = 300
        flick_start_y = 300
        flick_distance = 1000

        friends = []

        while True:
            before_swipe = self.driver.page_source
            self.driver.swipe(flick_start_x, flick_start_y + flick_distance, flick_start_x, flick_start_y)
            try:
                nicknames = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/ft6")))
                for nickname in nicknames:
                    if nickname.text not in friends:
                        friends.append(nickname.text)

            # 当获取不到新的信息时退出循环
            except NoSuchElementException:
                break
            after_swipe = self.driver.page_source
            if before_swipe == after_swipe:
                break
        return friends

    def crawl_all(self, num=None):
        """
        爬取朋友圈所有好友数据
        :param time_span: 爬取时间跨度，以秒为单位
        :return:
        """
        self.enter_friends_circle()

        flick_start_x = 300
        flick_start_y = 300
        flick_distance = 800

        moments_dict = defaultdict(list)  # 默认字典元素为list

        loguru.logger.info("开始爬取朋友圈数据:")
        while True:
            before_swipe = self.driver.page_source  # 滑动前页面
            self.driver.swipe(flick_start_x, flick_start_y + flick_distance, flick_start_x, flick_start_y)
            try:
                contents = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/bmy")))  # 内容
                names = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/fzg")))  # 昵称
                for (name, content) in zip(names, contents):
                    if content.text not in moments_dict[name.text]:
                        moments_dict[name.text].append(content.text)
            # 当获取不到新的信息时退出循环
            except NoSuchElementException:
                break
            after_swipe = self.driver.page_source  # 滑动后页面
            if before_swipe == after_swipe:  # 如果滑动前后页面相同，则说明到达了最底部
                break
            # 当设置了爬取的条数限制时
            if num is not None and sum([len(value) for value in moments_dict.values()]) >= num:
                break

        loguru.logger.info("爬取结束！")
        return moments_dict

    def crawl_one(self, nickname, num):
        """
        爬取某个好友num条朋友圈数据
        :param nickname:
        :param num:
        :param time_span:
        :return:
        """
        #  判断是否有该好友
        friends = self.get_all_friends()
        if nickname not in friends:
            loguru.logger.warning("微信中无昵称为 " + nickname + " 的好友")
            return None

        moments = []  # 默认字典元素为list

        self.enter_friends_circle()  # 进入朋友圈

        #  滑动设置
        flick_start_x = 300
        flick_start_y = 300
        flick_distance =800

        loguru.logger.info("开始爬取" + nickname + "的朋友圈数据:")
        before_swipe = self.driver.page_source  # 滑动前页面
        while len(moments) < num:
            self.driver.swipe(flick_start_x, flick_start_y + flick_distance, flick_start_x, flick_start_y)
            try:
                contents = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/bmy")))  # 内容
                names = self.wait.until(EC.presence_of_all_elements_located((By.ID, "com.tencent.mm:id/fzg")))  # 昵称
                for (name, content) in zip(names, contents):
                    if name.text == nickname and content.text not in moments:
                        moments.append(content.text)
            # 当获取不到新的信息时退出循环
            except NoSuchElementException:
                break
            after_swipe = self.driver.page_source  # 滑动后页面
            if before_swipe == after_swipe:  # 如果滑动前后页面相同，则说明到达了最底部
                break
        loguru.logger.info("爬取结束！")
        return moments


if __name__ == "__main__":
    m = Moments()
    moments = m.crawl_all(10)
    print(len(moments))
