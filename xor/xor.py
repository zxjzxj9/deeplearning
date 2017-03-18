#! /usr/bin/env python

'''
    A simple xor function
'''

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    label = np.array([[0], [1], [1], [0]], dtype=np.float32)

    w1 = tf.Variable(tf.random_normal((2, 2), dtype=tf.float32))
    b1 = tf.Variable(tf.zeros((2,), dtype=tf.float32))

    w2 = tf.Variable(tf.random_normal((2, 1), dtype=tf.float32))
    b2 = tf.Variable(tf.zeros((1,), dtype=tf.float32))

    x = tf.placeholder(tf.float32, [None, 2])
    y_ = tf.placeholder(tf.float32, [None, 1])
    hidden1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
    y = tf.nn.relu(tf.add(tf.matmul(hidden1, w2), b2))
    loss = tf.reduce_mean((y-y_)**2)
    #loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_ , logits = y) #tf.reduce_mean((y - y_)**2)
    
    minimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)

    sess = tf.Session()
    g = tf.global_variables_initializer()
    sess.run(g)

    for i in range(100):
        #d = np.arange(4)
        #np.random.shuffle(d)
        sess.run(minimizer, feed_dict = {x: data, y_ : label})
        #print sess.run(y, feed_dict =  {x: data, y_ : label})
        #print sess.run(y_, feed_dict =  {x: data, y_ : label})
        print(sess.run(loss, feed_dict = {x: data, y_ : label}))
        #break
    
    #print(sess.run(w1))
    #print(sess.run(b1))
    #print(sess.run(w2))
    #print(sess.run(b2))
    print(sess.run(y, feed_dict = {x: data}))
