import tensorflow as tf

# message = tf.constant("hello neural network")
# with tf.Session() as sess:
#     print(sess.run(message))

# v_1 = tf.constant([1,2,3,4])
# v_2 = tf.constant([5,2,2,2])
# v_add = tf.add(v_1, v_2)
# res = tf.zeros([3,3],dtype=tf.int32)
# with tf.Session() as sess:
    # print(sess.run(v_add))
    # print(sess.run(tf.zeros([3,3],dtype=tf.int32)))
    # print(sess.run(tf.ones([3,3],dtype=tf.int32)))
    # print(sess.run(tf.lin_space(2.0,5.0,5)))
    # tf_random = tf.random_normal([2,3],mean=1.0,stddev=2)
    # print(sess.run(tf_random))

    # tf_r = tf.random_uniform([50,50],0, 10, seed=10)
    # t_1 = tf.Variable(tf_r)
    # t_2 = tf.Variable(tf_r)
    # print(sess.run(tf_r))
    #
    # x = [[1, 2], [1, 2], [1, 2]]
    # y = [[0, 0], [0, 0], [0, 0]]
    # print(sess.run(tf.multiply(x, y)))
# t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
# t2 = tf.constant([[7, 8, 9], [10, 11, 12]])
# t3 = tf.concat([t1,t2],axis=1)
# with tf.Session() as sess:
#     print(t3.shape)
#     print(sess.run(t3))

# t1 = tf.zeros([1])
# t1 = tf.constant([1,1,3])
# with tf.Session() as sess:
#     print(sess.run(t1.shape))

# import numpy as np
#
# labels = [1, 3, 4, 8, 7, 5, 2, 9, 0, 8, 7]
# one_hot_index = np.arange(len(labels)) * 10 + labels
#
# print('one_hot_index:{}'.format(one_hot_index))
#
# one_hot = np.zeros((len(labels), 10))
# one_hot.flat[one_hot_index] = 1
#
# print('one_hot:{}'.format(one_hot))
print(tf.__version__)