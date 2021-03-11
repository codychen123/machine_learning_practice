# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import pandas as pd

data_path = r'./ua.base'
dt=pd.read_csv(data_path,sep="\t",header=None)

dt.columns=["uid","iid","target","ts"]

#######构造数据，target 小于d等于3的未0，大于3的为1
dt["label"] = dt["target"].map(lambda x: 1 if x>3 else 0)

#####tensorflow建立lr第一步 ，需要知道每个特征里有多少不一样的内容

x=["uid","iid"]

cat_map={}
###这是你做一次，后面就不用做了，后面自己维护了
#spark 来做
for feature_name in x:
    #k:实际数据值，v：自编码
    temp={}
    #去过重的
    x_feature = list(pd.unique(dt[feature_name]))
    temp={k:v for v,k in enumerate(x_feature)}
    cat_map[feature_name]=temp

#############第二步 我需要把数据变成 编码格式


for feature_name in x:
    dt[feature_name + "_enc"] = dt[feature_name].map(cat_map[feature_name])

import numpy as np

x = dt[["uid_enc", "iid_enc"]]
x = np.array(x).reshape((-1, 2))
label = dt["label"]
label = np.array(label).reshape((-1, 1))

###############做模型建构

uid_count = len(cat_map["uid"])
iid_count = len(cat_map["iid"])

embed_size = 1

input_x = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="feature")

input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="label")

# weight1  = tf.Variable(tf.truncated_normal(shape=(uid_count,embed_size)))
# weight2  = tf.Variable(tf.truncated_normal(shape=(iid_count,embed_size)))

weight = tf.Variable(tf.truncated_normal(shape=(uid_count + iid_count, embed_size)))

# layer1

##核心体验
out_put = tf.nn.embedding_lookup(weight, input_x)

logit = tf.nn.sigmoid(tf.reduce_sum(out_put, axis=1))

# layer2 out
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y, logits=logit)

print(loss)