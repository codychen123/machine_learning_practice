
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np
if __name__ == '__main__':
    path = r"/Users/cody/Desktop/大数据/practiceProject/machine_learning_practice/data/data.csv"
    data = pd.read_csv(path)

    X = data[["x1" ,"x2" ,"x3" ,"x4" ,"x5"]]
    y = data["y"]

    # 标准化
    X_scale =scale(X)
    # 测试集占0.2， 总数10w
    X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=.2)

    lr = LogisticRegression()

    lr.fit(X_train, y_train)

    y_pre = lr.predict_proba(X_test)[::, 1]

    metrics.roc_auc_score(y_test, y_pre)

