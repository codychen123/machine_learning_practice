# -*- coding: utf-8 -*-


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from sklearn import metrics

# x=np.random.random(size=(4,10))
#
# x.reshape((2,20)).shape
#
# x.reshape((-1,20)).shape

###step1设置输入占位符

X = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="feature")
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label")

####step2:初始化参数权重矩阵。 X  (m,5)   weight(5,1)

weight = tf.Variable(tf.truncated_normal(shape=(5, 1)))

bais = tf.Variable(tf.zeros(shape=[1]))

#####step3 构造图


logit = tf.nn.sigmoid(tf.matmul(X, weight) + bais)

reg = tf.reduce_sum(tf.multiply(weight, weight))

loss = 1 / 32 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)) + 0.1 * reg

######step4 反向传播 不需要你去计算梯度
opt = tf.train.AdamOptimizer(0.00005).minimize(loss)


###################喂数据

def batch(train_x, train_y, batch_size=32):
    m, n = train_x.shape
    for i in range(0, m, batch_size):
        lower = i
        supers = min(i + batch_size, m - 1)
        yield train_x[lower:supers], train_y[lower:supers]


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd

path = r"/Users/cody/Desktop/大数据/八斗19期/badou_project_data/all_path_for_recall/LR/data/data.csv"
data = pd.read_csv(path)
#######step1分开特征 和  label（分类任务所以只有1,0）
X1 = data[["x1", "x2", "x3", "x4", "x5"]]
y1 = data["y"].astype(np.float32)
########step2 标准化 (连续特征)
X_scale = scale(X1)
########step3 训练集 验证集划分
X_train, X_test, y_train, y_test = train_test_split(X_scale, y1, test_size=.2)

###
###这块就是为什么tf 要明显比 sklearn 地方，完全可以做一个文件队列，
### 去磁盘，一批一批拿数，不用把数据全部导入内存，将io密集型，和计算密集型分开处理。
# for bx,by in batch(X_train, y_train, batch_size=32):
#    print(bx,by)


with tf.Session() as sess:
    # 才相当于初始化所有变量
    sess.run(tf.initialize_all_variables())
    for epoch in range(30):
        for bx, by in batch(np.array(X_train), np.array(y_train).reshape(-1, 1)):
            _, los = sess.run([opt, loss], feed_dict={X: bx, y: by})
        print(los)
        print(sess.run(weight))

    pre = sess.run(logit, feed_dict={X: X_test.reshape(-1, 5)})

df1 = pd.DataFrame(np.concatenate([np.array(y_test).reshape(-1, 1), pre.reshape(-1, 1)], axis=1))

metrics.roc_auc_score(y_test, pre)










