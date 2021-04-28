from gensim.models import word2vec
import pandas as pd

data_path = r'ua.base'

dt = pd.read_csv(data_path,sep="\t",header=None)
dt.columns=["uid","iid","score","ts"]
# iid转string类型
dt["iid"]=dt['iid'].astype("str")

item_groupby_uid = dt.sort_values("ts").groupby("uid")
#1-> ['172', '168', '165', '156', '196', '166', '187', '14', '127', '250', '181',..]
item_list = dt.sort_values("ts").groupby("uid")['iid'].apply(list)

out_path= r"cut_ua.base"
with open(out_path,"w",encoding="utf-8") as f:
    for line in item_list:
        if len(line)>2:
            f.write(" ".join(line)+"\n")

def gensim_word2vector(cut_path):
    # 切分好词的语料库存放的路径
    path = cut_path
    # 词出调用gensim包提供给我们的 LineSentence引入文章。
    sentence = word2vec.LineSentence(path)
    # sentence已经以word2vec的格式将文本读入
    w2v = word2vec.Word2Vec(sentence, hs=1, min_count=1, window=20, size=16)

    return w2v
w2v = gensim_word2vector(out_path)
res = w2v.wv.similar_by_word("6")

####拿出所有的item向量 embedding
all_item= set(dt["iid"])
# shape(none,16)
embedding={}
for k, _ in w2v.wv.vocab.items():
        # w2v.wv[k],可以得到每个词的词向量
        embedding[k] = w2v.wv[k]

#########tf-idf 计算
from sklearn.feature_extraction.text import TfidfVectorizer

with open(out_path,"r+",encoding="utf-8") as f:
    # cut_item 用户iid集合
    cut_item =f.readlines()

# 用户iid集合
item_list_temp = []

for line in cut_item:
    item_list_temp.append(line.strip())

vector = TfidfVectorizer()
tf_idf = vector.fit_transform(item_list_temp)
word_list = vector.get_feature_names()
weight_list = tf_idf.toarray()

res1 = weight_list[1][1]

# 用户点击序列
item_hist_tf_idf_item = {}
uid = item_list.index
# weight_list {943,1671}
for i in range(len(weight_list)):
    # 1671
    for j in range(len(word_list)):
        if item_hist_tf_idf_item.get(uid[i], -1) == -1:
            item_hist_tf_idf_item[uid[i]] = {}

        if weight_list[i][j] > 0:
            item_hist_tf_idf_item[uid[i]].update({word_list[j]: weight_list[i][j]})

############挑选出用户最近点击的5个物品

# 构造用户列表
uid_near_list = {}
# 拿出用户索引
uid_set = item_list.index
for i in range(len(item_list)):
    # 获取用户最近点击的5个item
    uid_near_list[uid_set[i]] = item_list.iloc[i][-5:]

print(uid_near_list)

##########对用户1进行推荐
#1.拿出用户1 历史list
import numpy as np
uid_1=uid_near_list[1]
# uid_1
#2.拿出tf_idf

weight =[]

for item in uid_1:
    if item_hist_tf_idf_item[1].get(item,-1)==-1:
        # 没有权重，默认给权重
        weight.append(0.0001)
    else:
        weight.append(item_hist_tf_idf_item[1][item])
s=sum(weight)
# [0.1369301031933113, 0.0003111730971526818, 0.3002842907641859, 0.3369934539010405, 0.22548097904430955]
weight = [value/s for value in weight]
# weight
#3. 拿出对应的embedding 做 weight_pooling

res=[0.0]*16
index=0
for item in uid_1:
    res+=np.array(embedding[item])*weight[index]
    index+=1

# 基于res的embedding快速检索。
# [-0.17627933 -0.20100983 -0.75399621  0.59014821  0.78523762 -0.15364059,  0.52198296 -0.46529229 -0.02436952  0.56633694 -0.61992319 -0.14938237,  0.16962915  0.25165982  0.93778084  0.07999585]

#%%

import hnswlib
import time


#%%

index = hnswlib.Index(space='cosine', dim=16)
index.init_index(max_elements=len(set(dt["iid"])), ef_construction=200, M=16)
for iid in set(dt["iid"]):
    index.add_items(embedding[iid],iid)
start=time.time()
iid,sim = index.knn_query(res, k=20)
end= time.time()
print(end-start)
print(iid,sim)
#
# print(res)
