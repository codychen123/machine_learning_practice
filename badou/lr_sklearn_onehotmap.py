# -*- coding: utf-8 -*-

import tensorflow as tf
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
data = pd.read_csv(path,sep='\t',header=None)

data.columns=["uid","iid","score","ts"]

###score 下于3的为0 ，大于等于3的为1

data["label"] = data["score"].map(lambda x: 1 if x>=3 else 0)


######step1 处理数据 onehot，数据变成机器系数，libsvm

enc = OneHotEncoder()

X1= np.array(data[["uid","iid"]])

y=data["label"]


X = enc.fit_transform(X1).todense()
print(X1.take(10 ))

lr =LogisticRegression(max_iter=1000)
#训练
lr.fit(X,y)

print(lr.predict_proba(X[0]))
