# -*- coding: utf-8 -*-

import os
import queue
import random
import re
import shutil
import threading
import time
import jieba
import tqdm

# 数据集大小
ratio_size = 1
# 分词模式
process_mode = 0
# 是否去除停用词
process_rm_stop_word = True
# 是否仅提取正文
process_get_context = True
# 存放处理后数据的路径
new_dir1 = './data/SPAM'
new_dir2 = './data/HAM'
# words_filepath = './words.txt'
# 数据集索引文件路径
index_path = './trec06c/full/index'
# 线程锁
lock = threading.Lock()
filelock = threading.Lock()
# 线程数
threads_num = 50

# 进度条
pbar_cur = 0
pbar = None

if process_mode == 0:
    stop_word = open('./stop_words.txt', encoding='utf-8').read().split('\n')


def process(label, path):  # label大写
    data_dir = './data'
    with open(path, 'rb') as f:  # 二进制只读
        content = f.read()
        content = content.decode('gbk', 'ignore')
    new_dir = '{}/{}'.format(data_dir, label)  # ./data/HAM
    new_path = '{}/{}_{}.txt'.format(new_dir,
                                     path.split('/')[-2], path.split('/')[-1])
    if process_get_context:
        content = re.search(r'\n\n(.*)', content, re.S).group(1)
    text = content.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    if process_mode == 0:  # 精确模式分词
        text = jieba.cut(text, cut_all=False)
    elif process_mode == 1:  # 全模式分词
        text = jieba.cut(text, cut_all=True)
    else:  # 搜索引擎模式分词
        text = jieba.cut_for_search(text)
    if process_rm_stop_word:
        text = ','.join([word for word in text if word not in stop_word])
    else:
        text = ','.join(text)
    filelock.acquire()
    # with open(words_filepath, 'a', encoding='utf-8') as f:
    #     f.write(text)
    global pbar
    global pbar_cur
    pbar_cur += 1
    pbar.update(pbar_cur)
    filelock.release()
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(text)


# 继承线程类
class Worker(threading.Thread):
    def __init__(self, q: queue.Queue):
        threading.Thread.__init__(self)
        self.q = q

    def run(self):
        while 1:
            if not self.q.empty():
                lock.acquire()
                label, path = self.q.get()
                lock.release()
                process(label, path)
            else:
                break


def pre_data():
    if not os.path.exists(new_dir1):
        os.makedirs(new_dir1)
        os.makedirs(new_dir2)
    else:
        print('删除上一次预处理数据...')
        shutil.rmtree(new_dir1)
        shutil.rmtree(new_dir2)
        # if os.path.exists(words_filepath):
        #     os.remove(words_filepath)
        os.makedirs(new_dir1)
        os.makedirs(new_dir2)
    with open(index_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:int(ratio_size * len(lines))]
    print("训练数据集大小:", len(lines))
    global pbar
    pbar = tqdm.tqdm(total=len(lines), desc='处理进度', ncols=100, unit='B', unit_scale=True)
    data = list()
    for line in lines:
        label = line.split(' ')[0].upper()
        path = './trec06c' + line.split(' ')[1].replace('\n', '')[2:]
        data.append([label, path])
    q = queue.Queue()
    global count_spam
    global count_ham
    for label, path in data:
        if label == "SPAM":
            count_spam += 1
        else:
            count_ham += 1
        q.put([label, path])
    ws = list()
    for j in range(threads_num):
        w = Worker(q)
        w.start()
        ws.append(w)
    for w in ws:
        w.join()
    print('empty queue!')
    print('work over!!!')


if __name__ == "__main__":
    print('当前预处理设置:')
    print('    rm_stop_word:', process_rm_stop_word)
    print('    process_mode:', process_mode)
    print('    process_get_context:', process_get_context)
    print('    threads_num:', threads_num)
    time_start = time.time()
    count_spam = 0
    count_ham = 0
    pre_data()
    time_end = time.time()
    print('timecost:', time_end - time_start)
    print("count_spam:", count_spam)
    print("count_ham:", count_ham)
