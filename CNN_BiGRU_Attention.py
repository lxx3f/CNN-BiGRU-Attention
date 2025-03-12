import os
import random
import time
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Bidirectional, Lambda, Layer
from keras.layers import Conv1D, GlobalMaxPooling1D
from Attention_layer import Attention_layer
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import matplotlib.pyplot as plt


class CNN_BiGRU_SPAM():

    def __init__(self,
                 max_len=200,
                 max_features=500,
                 batch_size=32,
                 embedding_dims=25,
                 filters=50,
                 kernel_size=3,
                 epochs=3,
                 ratio_train_all=0.2,
                 is_printed=True):
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
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size

        # 训练集占数据集的比例
        self.ratio_train_all = ratio_train_all
        # 数据集
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_display = None
        self.y_display = None

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
        self.x_display = data_x[:10]
        self.y_display = data_y[:10]
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
        self.x_train = data_x[:int(self.ratio_train_all * len(data_x))]
        self.y_train = data_y[:int(self.ratio_train_all * len(data_y))]
        self.x_test = data_x[int(self.ratio_train_all * len(data_x)):]
        self.y_test = data_y[int(self.ratio_train_all * len(data_y)):]
        if self.is_printed:
            print(self.x_train[:5])
            print(self.y_train[:5])
            print(len(self.x_train), 'train sequences')
            print(len(self.x_test), 'test sequences')
            print('Pad sequences (samples x time)')
        self.x_train = sequence.pad_sequences(self.x_train,
                                              maxlen=self.max_len)
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
        input_layer = Input([
            self.max_len,
        ])
        embedding_layer = Embedding(self.max_features,
                                    self.embedding_dims)(input_layer)
        conv_layer = Conv1D(self.filters,
                            self.kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1)(embedding_layer)
        BiGRU_layer = Bidirectional(GRU(64, return_sequences=True,
                                        dropout=0.2))(conv_layer)
        # BiGRU_layer = Bidirectional(GRU(48, return_sequences=True, dropout=0.2))(BiGRU_layer)
        attention_layer = Attention_layer()(BiGRU_layer)
        # BiGRU_layer = Bidirectional(GRU(64, return_sequences=False, dropout=0.2))(attention_layer)
        # output_layer = Dense(1, activation='sigmoid')(BiGRU_layer)
        output_layer = Dense(1, activation='sigmoid')(attention_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', 'precision', 'recall'])
        if self.is_printed:
            model.summary()
            print('Train...')
        history = model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=[self.x_test, self.y_test],
        )
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('CNN-BiGRU Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('CNN-BiGRU Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        for i, x in enumerate(self.x_display):
            if self.y_display[i] == 0:
                print("正常邮件：")
            else:
                print("垃圾邮件：")
            if len(x) < 30:
                print(x)
            else:
                print(x[:30] + ["..."])
            if self.y_display[i] == 0:
                print("分类结果：正常邮件")
            else:
                print("分类结果：垃圾邮件")

        val_loss, self.accuracy, self.precision, self.recall = model.evaluate(
            self.x_test, self.y_test, batch_size=self.batch_size)
        self.f_beta = (
            (1 + self.BETA * self.BETA) * self.precision * self.recall /
            ((self.BETA * self.BETA * self.precision) + self.recall))
        self.f_beta = round(self.f_beta, self.round_digits)
        if self.is_printed:
            print('accuracy:{}'.format(self.accuracy))
            print('precision:{}'.format(self.precision))
            print('recall:{}'.format(self.recall))
            print('f_beta:{}'.format(self.f_beta))
            print('总用时：{}'.format(time.time() - self.time_start))
            # plot_model(model, to_file='model.png')
            model.summary()

    def save_to_log(self, filepath="log/log_CNN-BiGRU.txt"):
        with open(filepath, encoding='utf-8', mode='a') as f:
            f.write("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(
                len(self.y_train), len(self.y_test), self.accuracy,
                self.precision, self.recall, self.f_beta))


if __name__ == "__main__":
    cnn_bigru = CNN_BiGRU_SPAM(is_printed=True)
    cnn_bigru.load_data()
    cnn_bigru.train()
    cnn_bigru.save_to_log()
