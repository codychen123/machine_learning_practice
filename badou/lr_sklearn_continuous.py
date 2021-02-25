# -*- coding: utf-8 -*-

#################demo



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np

##########################
path = r"/Users/cody/Desktop/大数据/八斗19期/badou_project_data/all_path_for_recall/LR/data/data.csv"
data = pd.read_csv(path)

#######step1分开特征 和  label（分类任务所以只有1,0）
X = data[["x1","x2","x3","x4","x5"]]
y = data["y"]

########step2 标准化
X_scale =scale(X)
########step3 训练集 验证集划分
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=.2)

#先设置参数
lr =LogisticRegression()
#训练
lr.fit(X_train,y_train)

y_pre = lr.predict_proba(X_test)[::, 1]

df=pd.DataFrame(np.concatenate([np.array(y_test).reshape(-1,1),y_pre.reshape(-1,1)],axis=1))



metrics.roc_auc_score(y_test, y_pre)
