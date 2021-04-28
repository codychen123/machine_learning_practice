import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import   OneHotEncoder

def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)



cols = ['user','item','rating','timestamp']

train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)
#label数据
y_train=np.array(train["rating"].values).reshape(-1,1)
y_test=np.array(test["rating"].values).reshape(-1,1)
#x数据
x_train=np.array(train[['user',"item"]]).reshape(-1,2)
x_test=np.array(test[['user',"item"]]).reshape(-1,2)
# shape->(100000,2)
x_all=np.concatenate([x_train,x_test],axis=0)
####onehot
enc=OneHotEncoder()
enc.fit(x_all)
x_train=enc.fit_transform(x_train).toarray()
# shape(none,2073)
x_test=enc.fit_transform(x_test).toarray()
n,p = x_train.shape

k = 10
####################################tensorflow
#step1 定义模型输入，占位符也就是输入的数据,p->2623(onehot后的长度)
x = tf.placeholder('float',[None,p])
y = tf.placeholder('float',[None,1])

#steo2 初始化fm一阶参数   w0+w1*x1+w2*x2+...wn*xn
w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))
#初始化，V 为了二阶参数   wij=<vi，vj> ,k是指定的也叫作embedding_size
#p就是onehot之后特征维度，也就是特征内容个数综合，shape->(10,2623)
v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))

#y_hat = tf.Variable(tf.zeros([n,1]))
#这表示一阶相乘w0+w1*x1+w2*x2+...
# wn*xn    [x1,x2,x3,x4]  [w1,w2,w3,w4] tf.multiply ===》 [x1*w1,x2*w2,....]
#tf.reduce_sum,multiply->元素各自相乘
# w->shape(2623,none),x->(?,2623),w*x->(?,2623)
multiply_res = tf.multiply(w,x)
# shape(?,2623)->(?,1)
reduce_sum_res = tf.reduce_sum(multiply_res,1,keep_dims=True)
linear_terms = tf.add(w0,reduce_sum_res) # n * 1


#fm绘图中step4这一步。subtract->x-y
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        # x->(none,2623),v->(10,2623),(none,2623)*(2623,10)->(none,10)，对应[v(if)*x(i)]2
        tf.pow(tf.matmul(x,tf.transpose(v)),2),
        #
        tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
    ),axis = 1 , keep_dims=True)


y_hat = tf.add(linear_terms,pair_interactions)

lambda_w = tf.constant(0.001,name='lambda_w')
lambda_v = tf.constant(0.001,name='lambda_v')

l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w,tf.pow(w,2)),
        tf.multiply(lambda_v,tf.pow(v,2))
    )
)
# 均方差
error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error,l2_norm)


train_op = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)


epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 10轮
    for epoch in tqdm(range(epochs), unit='epoch'):
        # 打乱数据
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches 分批训练
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
            print(t)


    errors = []
    for bX, bY in batcher(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        print(errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print (RMSE)