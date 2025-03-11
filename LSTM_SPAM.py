import os
import random
import time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import matplotlib.pyplot as plt

'''
max_features = 20000
# 在此数量的单词之后剪切文本（取最常见的 max_features 个单词）
max_len = 400
batch_size = 32
epochs = 3
ratio_train_test = 0.5  # 训练集占数据集的比例
F_beta = 0.1  # 查全率对查准率的相对重要性


time_start = time.time()
'''


class LSTM_SPAM():
    def __init__(self,
                 max_len=200,
                 max_features=200,
                 batch_size=128,
                 embedding_dims=50,
                 epochs=30,
                 ratio_train_test=0.2,
                 is_printed=True
                 ):
        # 评价指标
        self.round_digits = 4  # 小数点后保留位数
        self.accuracy = None
        self.BETA = 0.1
        self.f_beta = None
        self.recall = None
        self.precision = None

        # 模型参数
        self.max_len = max_len  # 最长邮件长度
        self.max_features = max_features  # 字典容量
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size

        # 训练集占数据集的比例
        self.ratio_train_test = ratio_train_test
        # 数据集
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # 是否打印过程信息,默认为true
        self.is_printed = is_printed
        self.time_start = time.time()

    def load_data(self, data_dir='./data'):
        labels = os.listdir(data_dir)
        data_x = list()
        data_y = list()
        if self.is_printed:
            print('读取数据...')
        for label in labels:
            label_dir = data_dir + os.sep + label
            filenames = os.listdir(label_dir)
            for filename in filenames:
                data_path = label_dir + os.sep + filename
                with open(data_path, encoding='utf-8') as f:
                    text = f.read()
                data_x.append(text.split(','))
                data_y.append(1 if label == 'SPAM' else 0)
        combined = list(zip(data_x, data_y))
        random.shuffle(combined)
        data_x, data_y = zip(*combined)
        data_x, data_y = list(data_x), list(data_y)
        if self.is_printed:
            print('读取数据完毕， 用时：{}'.format(time.time() - self.time_start))
        tok = Tokenizer(num_words=self.max_features)
        tok.fit_on_texts(data_x)
        if self.is_printed:
            print("data_x[0]: ", data_x[0])
            # 使用word_index属性查看每个词对应的编码
            for ii, iterm in enumerate(tok.word_index.items()):
                if ii < 10:
                    print(iterm)
                else:
                    break
            print("===================")
            # 使用word_counts属性查看每个词对应的频数
            for ii, iterm in enumerate(tok.word_counts.items()):
                if ii < 10:
                    print(iterm)
                else:
                    break
        data_x = tok.texts_to_sequences(data_x)
        self.x_train = data_x[:int(self.ratio_train_test * len(data_x))]
        self.y_train = data_y[:int(self.ratio_train_test * len(data_y))]
        self.x_test = data_x[int(self.ratio_train_test * len(data_x)):]
        self.y_test = data_y[int(self.ratio_train_test * len(data_y)):]
        if self.is_printed:
            print(self.x_train[:5])
            print(self.y_train[:5])
            print(len(self.x_train), 'train sequences')
            print(len(self.x_test), 'test sequences')
            print('Pad sequences (samples x time)')
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.max_len)
        self.x_train = tf.convert_to_tensor(self.x_train)
        self.y_train = tf.convert_to_tensor(self.y_train)
        self.x_test = tf.convert_to_tensor(self.x_test)
        self.y_test = tf.convert_to_tensor(self.y_test)
        if self.is_printed:
            print('x_train shape:', self.x_train.shape)
            print('x_test shape:', self.x_test.shape)

    def train(self):
        if self.is_printed:
            print('Build model...')
        model = Sequential()
        model.add(Embedding(self.max_features, self.embedding_dims))
        model.add(LSTM(self.embedding_dims, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', 'precision', 'recall'])
        if self.is_printed:
            model.summary()
            print('Train...')
        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=[self.x_test, self.y_test],
                            )
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('LSTM Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('LSTM Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        val_loss, self.accuracy, self.precision, self.recall = model.evaluate(
            self.x_test,
            self.y_test,
            batch_size=self.batch_size)
        self.f_beta = ((1 + self.BETA * self.BETA) * self.precision * self.recall
                       / ((self.BETA * self.BETA * self.precision) + self.recall))
        self.f_beta = round(self.f_beta, self.round_digits)
        if self.is_printed:
            print('accuracy:{}'.format(self.accuracy))
            print('precision:{}'.format(self.precision))
            print('recall:{}'.format(self.recall))
            print('f_beta:{}'.format(self.f_beta))
            print('总用时：{}'.format(time.time() - self.time_start))

    def save_to_log(self, filepath="log/log_LSTM.txt"):
        with open(filepath, encoding='utf-8', mode='a') as f:
            f.write(
                "{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(len(self.y_train), len(self.y_test),
                                                            self.accuracy, self.precision, self.recall,
                                                            self.f_beta))


if __name__ == "__main__":
    lstm = LSTM_SPAM()
    lstm.load_data()
    lstm.train()
    lstm.save_to_log()

'''
def load_data(rate=1):
    data_dir = './data'
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
            data_y.append(1 if label == 'SPAM' else 0)
    combined = list(zip(data_x, data_y))
    random.shuffle(combined)
    data_x, data_y = zip(*combined)
    print('读取数据完毕， 用时：{}'.format(time.time() - time_start))
    return list(data_x), list(data_y)

print('Loading data...')

data_x, data_y = load_data()
tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(data_x)
print(data_x[0])
data_x = tok.texts_to_sequences(data_x)
# 使用word_index属性查看每个词对应的编码
# 使用word_counts属性查看每个词对应的频数
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

x_train = data_x[:int(ratio_train_test * len(data_x))]
y_train = data_y[:int(ratio_train_test * len(data_y))]
x_test = data_x[int(ratio_train_test * len(data_x)):]
y_test = data_y[int(ratio_train_test * len(data_y)):]

print(x_train[:5])
print(y_train[:5])
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 尝试使用不同的优化器和优化器配置
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['precision', 'recall'])
# model.summary()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
val_loss, val_precision, val_recall = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('val_precision:', val_precision)
print('val_recall:', val_recall)
val_F_beta = (1+F_beta*F_beta)*val_precision*val_recall/((F_beta*F_beta*val_precision)+val_recall)
print('val_F1:', val_F_beta)

# model.summary()
'''
