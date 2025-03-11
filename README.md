# 项目结构

```
CNN-BiGRU-Attention
├─ Attention_layer.py # 注意力机制层
├─ CNN_BiGRU_Attention.py
├─ CNN_SPAM.py
├─ GRU_SPAM.py
├─ log # 存放日志文件
├─ LSTM_SPAM.py
├─ main.py
├─ model # 保存训练完成的模型
├─ NB_SPAM.py
├─ PreData_SPAM.py
├─ README.md
├─ reference # 部分参考代码
├─ stop_words.txt
└─ SVM_SPAM.py
```

# 数据集
[trec06c](https://plg.uwaterloo.ca/~gvcormac/treccorpus06/)
原始数据下载解压后放在./trec06c,预处理完成后数据存放在./data

# 预处理
PreData(SPAM).py 
分词，去除停用词。

结果示例：`这番话,说明,很,有心,赞,一下,今天,晚上,我家,出来,回家,开门,外面,还,很,闷热,雷声,......`

# 朴素贝叶斯
NB(SPAM).py 朴素贝叶斯算法

## 原理
$$
\begin{aligned}
& 贝叶斯公式: P(A|B) = \frac{P(B|A)P(A)}{P(B)} \\
& 其中P(A)称为先验概率,在垃圾邮件分类应用中P(A)为一封邮件是垃圾邮件的概率, \\
& P(B)则是邮件内容的概率分布;
\end{aligned}
$$

分类标准：当 P（垃圾邮件|邮件内容）> P（正常邮件|邮件内容）时，我们认为该邮件为垃圾邮件，但是单凭单个词而做出判断误差肯定相当大，因此需要将所有的词一起进行联合判断。

这里假设：所有词语彼此之间是不相关的（严格说这个假设不成立；实际上各词语之间不可能完全没有相关性，但这里可以忽略）。

为什么取对数：有些概率值很小，直接算容易丢失精度。

## 评价指标
```
TP(true positives): 真正例, 实际为正且被预测为正
TN(true negatives): 真反例, 实际为反且被预测为反
FP(false positives): 假正例, 实际为反且被预测为正
FN(false negatives): 假反例, 实际为正且被预测为反
```

`accuracy`: 预测准确率,=(TP+TN)/(TP+TN+FP+FN)

`precision`: 查准率,=TP/(TP+FP)

`recall`: 召回率,=TP/(TP+FN)

`f_beta`: beta为1时是precision和recall的调和平均数（倒数平均数）, = $(1+\beta^2)\cdot \frac{Precision \cdot Recall}{\beta^2\cdot Precision+Recall}$,其中beta越大说明recall的权重越大,在本例中beta设为0.1

# 支持向量机
SVM(SPAM).py 支持向量机方法

# 深度学习
CNN(SPAM).py 卷积神经网络方法

LSTM_SPAM.py 