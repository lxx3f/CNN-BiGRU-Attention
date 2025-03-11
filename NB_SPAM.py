# -*- coding: utf-8 -*-
import os
import random
import time
import math
import numpy as np


class NB_SPAM():
    """
    //>>>NB = NB_SPAM()
    //>>>NB.load_data()
    />>>NB.train()
    """

    def __init__(self, ratio_train_all=0.6, isprinted=True):
        '''
        初始化一些变量
        '''

        self.PH = None  # 一个向量,每个值代表训练集中正常邮件中出现某个词的频率
        self.PS = None  # 一个向量,每个值代表训练集中垃圾邮件中出现某个词的频率
        self.P_ham = None  # 训练集中正常邮件的概率
        self.P_spam = None  # 训练集中垃圾邮件的概率
        self.round_digits = 4  # 输出保留小数位数
        self.accuracy = None  # 准确率(预测成功的次数/总预测次数)
        self.BETA = 0.1  # f_beta的计算参数
        self.f_beta = None  # 结合查准率和召回率的评价指标
        self.recall = None  # TP/(TP+FN)
        self.precision = None  # TP/(TP+FP)
        self.train_vecs = None  # 保存训练集样本
        self.train_labels = None  # 保存训练集标签
        self.test_vecs = None  # 保存测试集样本
        self.test_labels = None  # 保存测试集标签
        self.is_printed = isprinted  # 是否打印过程信息
        self.ratio_train_all = ratio_train_all  # 训练集占数据集的比例
        self.time_start = time.time()

    def load_data(self, data_dir='./data'):
        '''
        读取数据集
        '''

        labels = os.listdir(data_dir)
        data = list()
        for label in labels:
            label_dir = data_dir + os.sep + label
            filenames = os.listdir(label_dir)
            for filename in filenames[:len(filenames)]:
                data_path = label_dir + os.sep + filename
                with open(data_path, encoding='utf-8') as f:
                    text = f.read()
                data.append([1 if label == 'SPAM' else 0, text.split(',')])

        if self.is_printed:
            print('读取数据完毕， 用时：{}'.format(time.time() - self.time_start))

        random.shuffle(data)  # 随机打乱数据集顺序
        # 拆分为训练集和测试集
        train_data = data[:int(len(data) * self.ratio_train_all)]
        test_data = data[int(len(data) * self.ratio_train_all):]

        wordset = set()  # 词集
        for label, text in train_data:
            wordset |= set(text)
        wordset = list(wordset)
        wordict = {word: key for key, word in enumerate(wordset)}  # 给每个词一个编号
        if self.is_printed:
            print('词集长度：{}'.format(len(wordset)))

        # 数据集向量化
        self.train_vecs = np.zeros([len(train_data),
                                    len(wordset)])  # 二维数组,稀疏矩阵
        self.train_labels = np.zeros(len(train_data))
        for k, [label, text] in enumerate(train_data):
            for word in text:
                if word in wordict:
                    self.train_vecs[k, wordict[word]] = 1  # 不考虑重复出现的词
            self.train_labels[k] = label
        self.test_vecs = np.zeros([len(test_data), len(wordset)])
        self.test_labels = np.zeros(len(test_data))
        for k, [label, text] in enumerate(test_data):
            for word in text:
                if word in wordict:
                    self.test_vecs[k, wordict[word]] = 1
            self.test_labels[k] = label
        if self.is_printed:
            print('文档向量化完毕, 用时：{}'.format(time.time() - self.time_start))
            print(self.train_vecs[0][:10])
            print(self.train_labels[0])

    def train(self):
        '''
        训练模型
        '''
        if self.is_printed:
            print('training\n-----')

        self.P_spam = sum(self.train_labels) / len(self.train_labels)
        self.P_ham = 1 - self.P_spam
        SN = np.ones(self.train_vecs.shape[1])  # 垃圾邮件的概率分布向量
        HN = np.ones(self.train_vecs.shape[1])  # 正常邮件的概率分布向量
        for k, d in enumerate(self.train_vecs):
            if self.train_labels[k]:
                SN += d  # 向量相加
            else:
                HN += d
        self.PS = SN / sum(SN)
        self.PH = HN / sum(HN)

        if self.is_printed:
            print('训练完毕， 用时：{}'.format(time.time() - self.time_start))

    def predict(self):
        '''
        预测结果
        '''
        if self.is_printed:
            print('testing\n------')

        # 自然对数转换
        self.PS = np.log(self.PS)
        self.PH = np.log(self.PH)
        self.P_spam = math.log(self.P_spam)
        self.P_ham = math.log(self.P_ham)
        # 计算预测结果
        predict_vec = [
            1 if (self.P_spam + sum(d * self.PS)) >= (self.P_ham +
                                                      sum(d * self.PH)) else 0
            for d in self.test_vecs
        ]
        # 计算评价指标
        self.accuracy = np.mean([
            1 if la == predict_vec[idx] else 0
            for idx, la in enumerate(self.test_labels)
        ])
        self.precision = np.mean([
            1 if la == predict_vec[idx] else 0
            for idx, la in enumerate(self.test_labels) if predict_vec[idx] == 1
        ])
        self.recall = np.mean([
            1 if la == predict_vec[idx] else 0
            for idx, la in enumerate(self.test_labels) if la == 1
        ])
        self.f_beta = (
            1 + self.BETA * self.BETA) * self.precision * self.recall / (
                (self.BETA * self.BETA * self.precision) + self.recall)
        self.accuracy = round(self.accuracy, self.round_digits)
        self.precision = round(self.precision, self.round_digits)
        self.recall = round(self.recall, self.round_digits)
        self.f_beta = round(self.f_beta, self.round_digits)
        if self.is_printed:
            print('accuracy:{}'.format(self.accuracy))
            print('precision:{}'.format(self.precision))
            print('recall:{}'.format(self.recall))
            print('f_beta:{}'.format(self.f_beta))
            print('总用时：{}'.format(time.time() - self.time_start))

    def save_to_log(self, filepath="log/log_NB.txt"):
        '''
        保存结果记录
        '''
        with open(filepath, encoding='utf-8', mode='a') as f:
            f.write("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(
                len(self.train_labels), len(self.test_labels), self.accuracy,
                self.precision, self.recall, self.f_beta))


if __name__ == "__main__":
    NB = NB_SPAM(ratio_train_all=0.1)
    NB.load_data()
    NB.train()
    NB.predict()
    NB.save_to_log()
