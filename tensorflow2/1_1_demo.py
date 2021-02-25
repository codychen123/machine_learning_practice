import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers

dftrain_raw = pd.read_csv('../data/titanic/train.csv')
dftest_raw = pd.read_csv('../data/titanic/train.csv')

# Survived:0代表死亡，1代表存活【y标签】
# Pclass:乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
# Name:乘客姓名 【舍去】
# Sex:乘客性别 【转换成bool特征】
# Age:乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
# SibSp:乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
# Parch:乘客父母/孩子的个数(整数值)【数值特征】
# Ticket:票号(字符串)【舍去】
# Fare:乘客所持票的价格(浮点数，0-500不等) 【数值特征】
# Cabin:乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
# Embarked:乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】

# dftrain_raw.head(10)

# %matplotlib inline
# %config InlineBackend.figure_format = 'png'
# ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',
#      figsize = (4,8),fontsize=15,rot = 0)
# ax.set_ylabel('Counts',fontsize = 15)
# ax.set_xlabel('Survived',fontsize = 15)
# plt.show()

# ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 20,color= 'purple',
#                     figsize = (12,8),fontsize=15)
#
# ax.set_ylabel('Frequency',fontsize = 15)
# ax.set_xlabel('Age',fontsize = 15)
# plt.show()

# ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',
#                       figsize = (12,8),fontsize=15)
# dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
#                       figsize = (12,8),fontsize=15)
# ax.legend(['Survived==0','Survived==1'],fontsize = 12)
# ax.set_ylabel('Density',fontsize = 15)
# ax.set_xlabel('Age',fontsize = 15)
# plt.show()

def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )


tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

model.summary()

# 二分类问题选择二元交叉熵损失函数
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC'])

history = model.fit(x_train,y_train,
                    batch_size= 64,
                    epochs= 30,
                    validation_split=0.2 #分割一部分训练数据用于验证
                   )

import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history,"loss")

plot_metric(history,"AUC")