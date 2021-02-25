# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np

######sklearn lr试试

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import OneHotEncoder

path=r"/Users/cody/Desktop/大数据/八斗19期/badou_project_data/all_path_for_recall/data/ua.base"
data = pd.read_csv(path, sep='\t', header=None)
data.columns = ["uid", "iid", "score", "ts"]

data["label"] = data["score"].map(lambda x: 1 if x >= 3 else 0)

X = data[["uid", "iid"]]

y = data["label"]

feature_map = {}

# 全局index索引：
index = 0

for name in X.columns:
    temp = pd.unique(X[name])
    feature_map[name] = {k: v for k, v in zip(temp, range(index, index + len(temp)))}
    index += len(temp)

# index 就是我们的特征长度


for name in X.columns:
    X[name + "_index"] = X[name].map(feature_map[name])

train_x = np.array(X[["uid_index", "iid_index"]]).astype(np.int32).reshape((-1, 2))

train_y = np.array(y).astype(np.float32).reshape((-1, 1))

#####################################建模阶段
# 占位符

x_input = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="feature")
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label")

# 初始化呢参数 weight


weight = tf.Variable(tf.truncated_normal(shape=(index, 1)))

bais = tf.Variable(tf.zeros(shape=[1]))

# 我们要进行加法合并：

# wx+b
linear = tf.reduce_sum(tf.nn.embedding_lookup(weight, x_input), axis=1) + bais

# sigmod(wx+b)
logit = tf.nn.sigmoid(linear)

# |w|^2
reg = tf.reduce_sum(tf.multiply(weight, weight))

# loss -1/batch_size*sum (yi*log(sigmod(wx+b))+(1-yi)*log(1-sigmod(wx+b))
loss = 1 / 32 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)) + 0.1 * reg

######step4 反向传播 不需要你去计算梯度
opt = tf.train.AdamOptimizer(0.00005).minimize(loss)


def batch(train_x, train_y, batch_size=32):
    m, n = train_x.shape
    for i in range(0, m, batch_size):
        lower = i
        supers = min(i + batch_size, m - 1)
        yield train_x[lower:supers], train_y[lower:supers]


los_list = []
with tf.Session() as sess:
    # 才相当于初始化所有变量
    sess.run(tf.initialize_all_variables())
    for epoch in range(30):
        for bx, by in batch(np.array(train_x), np.array(train_y)):
            _, los = sess.run([opt, loss], feed_dict={x_input: bx, y: by})
        los_list.append(los)
        print(los)

# 绘制loss图
import matplotlib.pyplot as plt

x = range(len(los_list))
plt.plot(x, los_list)
plt.show()

x = tf.constant([[1, 1, 1],
                 [1, 1, 1]])
t1 = tf.reduce_sum(x)
t2 = tf.reduce_sum(x, axis=0)
t3 = tf.reduce_sum(x, axis=1)

sess = tf.Session()

print(sess.run(t1))

print(sess.run(t2))

print(sess.run(t3))














