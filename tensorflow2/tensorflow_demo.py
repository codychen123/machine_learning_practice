import tensorflow as tf

# message = tf.constant("hello neural network")
# with tf.Session() as sess:
#     print(sess.run(message))

v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([5,2,2,2])
v_add = tf.add(v_1, v_2)
res = tf.zeros([3,3],dtype=tf.int32)
with tf.Session() as sess:
    # print(sess.run(v_add))
    # print(sess.run(tf.zeros([3,3],dtype=tf.int32)))
    # print(sess.run(tf.ones([3,3],dtype=tf.int32)))
    # print(sess.run(tf.lin_space(2.0,5.0,5)))
    # tf_random = tf.random_normal([2,3],mean=1.0,stddev=2)
    # print(sess.run(tf_random))

    tf_r = tf.random_uniform([50,50],0, 10, seed=10)
    t_1 = tf.Variable(tf_r)
    t_2 = tf.Variable(tf_r)
    print(sess.run(tf_r))

    x = [[1, 2], [1, 2], [1, 2]]
    y = [[0, 0], [0, 0], [0, 0]]
    print(sess.run(tf.multiply(x, y)))

