import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.array([1,1,0,0,1,1,0])
y_scores = np.array([0.8,0.7,0.6,0.4,0.5,0.3,0.1])
print("y true is ", y_true);
print("y scores is ", y_scores);
print("auc is ",roc_auc_score(y_true,y_scores));

label_all = np.array([1,1,0,0,1,1,0]).reshape((-1,1))
pred_all = y_scores.reshape((-1,1))

print(label_all)
print(pred_all)
# res = list(filter(lambda s: s[0] == 1, label_all));
posNum = len(list(filter(lambda s: s[0] == 1, label_all)))
if (posNum > 0):
    negNum = len(label_all) - posNum
    #对preall进行排序
    sortedq = sorted(enumerate(pred_all), key=lambda x: x[0])
    print("vsss ",pred_all[0])
    print(sortedq)
    posRankSum = 0
    #遍历预测序列
    for j in range(len(pred_all)):
        if (label_all[j][0] == 1):
            posRankSum += list(map(lambda x: x[0], sortedq)).index(j) + 1
    auc = (posRankSum - posNum * (posNum + 1) / 2) / (posNum * negNum)
    print("auc:", auc)