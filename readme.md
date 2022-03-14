# 微信朋友圈情感分析

#### 一、相关配置和使用流程

python环境安装：
https://www.runoob.com/python/python-install.html

java环境安装：
https://www.runoob.com/java/java-environment-setup.html

Appium+Android SDK安装：
https://www.cnblogs.com/soundcode/p/12682366.html

相关python软件包：

```
jieba

pytorch

Appium-Python-Client

pandas

numpy
```

```
安装命令：pip install 包名
```


程序运行流程：首先用USB连接手机，打开手机的开发者模式，并打开USB调试，之后再打开Appium，最后运行main.py文件即可



#### 2、项目实现步骤

##### 1、情感分析模型

由于微信的限制，无法爬取他人朋友圈内容，故本项目中模型训练使用的数据集来自公开数据集

模型基于 self-attention + cnn 实现



#### 2、微信朋友圈内容的获取

主要基于python 中的 Appium-Python-Client库，自动化操作手机中的app来获取自己朋友圈的内容