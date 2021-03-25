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


def remove_transparency(img_pil, bg_colour=(255, 255, 255)):
    # Only process if image has transparency
    if img_pil.mode in ('RGBA', 'LA') or \
        (img_pil.mode == 'P' and 'transparency' in img_pil.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = img_pil.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", img_pil.size, bg_colour + (255,))
        bg.paste(img_pil, mask=alpha)
        return bg

    else:
        return img_pil

remove_transparency()