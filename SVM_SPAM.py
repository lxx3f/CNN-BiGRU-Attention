import os
import random
import time

import pandas as pd
import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, fbeta_score
import tensorflow as tf
import matplotlib.pyplot as plt


class SVM_SPAM():
    """
    SVM
    example:
    # >>> svm = SVM_SPAM()
    # >>> svm.load_data()
    # >>> svm.train()
    # >>> svm.save_to_log()
    """

    def __init__(self, max_len=300, max_features=2000, ratio_train_test=0.6, isprinted=True):
        self.accuracy = None
        self.rount_ndight = 4
        self.fbeta = None
        self.recall = None
        self.precision = None
        self.data_y = None
        self.data_x = None
        self.isprinted = isprinted
        self.max_len = max_len  # 最长邮件长度
        self.max_features = max_features  # 字典容量
        self.ratio_train_test = ratio_train_test  # 训练集占数据集的比例

    def load_data(self, data_dir='./data'):
        labels = os.listdir(data_dir)
        data_x = list()
        data_y = list()
        for label in labels:
            label_dir = data_dir + os.sep + label
            filenames = os.listdir(label_dir)
            for filename in filenames:
                data_path = label_dir + os.sep + filename
                with open(data_path, encoding='utf-8') as f:
                    text = f.read()
                data_x.append(text.split(','))
                data_y.append(1 if label == 'SPAM' else -1)
        combined = list(zip(data_x, data_y))
        random.shuffle(combined)
        data_x, data_y = zip(*combined)
        self.data_x = list(data_x)
        self.data_y = list(data_y)
        if self.isprinted:
            print('读取数据完毕')

    def set_max_len(self, max_len):
        self.max_len = max_len

    def set_max_features(self, max_features):
        self.max_features = max_features

    def train(self):
        tok = Tokenizer(num_words=self.max_features)
        tok.fit_on_texts(self.data_x)
        t_data_x = tok.texts_to_sequences(self.data_x)
        x_train = t_data_x[:int(self.ratio_train_test * len(t_data_x))]
        y_train = self.data_y[:int(self.ratio_train_test * len(self.data_y))]
        x_test = t_data_x[int(self.ratio_train_test * len(t_data_x)):]
        y_test = self.data_y[int(self.ratio_train_test * len(self.data_y)):]
        if self.isprinted:
            for ii, iterm in enumerate(tok.word_index.items()):
                if ii < 10:
                    print(iterm)
                else:
                    break
            print("===================")
            for ii, iterm in enumerate(tok.word_counts.items()):
                if ii < 10:
                    print(iterm)
                else:
                    break
        if self.isprinted:
            print(len(x_train), 'train sequences')
            print(len(x_test), 'test sequences')
            print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.convert_to_tensor(y_test)
        if self.isprinted:
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)
        svm = SVC(kernel='sigmoid', C=100,
                  random_state=0,
                  max_iter=10000000, )
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        self.accuracy = round(sum([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))])/len(y_pred), self.rount_ndight)
        self.precision = round(precision_score(y_test, y_pred), self.rount_ndight)
        self.recall = round(recall_score(y_test, y_pred), self.rount_ndight)
        self.fbeta = round(fbeta_score(y_test, y_pred, beta=0.1), self.rount_ndight)
        if self.isprinted:
            print("accuracy: ", self.accuracy)
            print("precision: ", self.precision)
            print("recall: ", self.recall)
            print("fbeta: ", self.fbeta)

    def save_to_log(self, filepath="log/log_SVM.txt"):
        with open(filepath, encoding='utf-8', mode='a') as f:
            f.write(
                "{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(self.max_len, self.max_features, self.precision, self.recall, self.fbeta))


if __name__ == "__main__":
    # 定义一些超参数
    max_len = 470  # 最长邮件长度
    max_features = 1000
    ratio_train_test = 0.6  # 训练集占数据集的比例
    time_start = time.time()
    svm = SVM_SPAM(max_len, max_features, ratio_train_test, True)
    svm.load_data()
    print("数据加载完成，用时", time.time() - time_start)
    svm.train()
    # print("训练完成，开始测试")
    print("测试完成，总用时", time.time() - time_start)
    svm.save_to_log()
