#! /usr/bin/python

# This is an example of AutoEncoder

import numpy as np
import tensorflow as tf

class denoiser(object):
    def __init__(self):
        pass

    def train(self, indata):
        noisyX = tf.placeholder(tf.float32, shape=(None,2))
        realX = tf.placeholder(tf.float32, shape=(None,2))

        W1 = tf.Variable(tf.truncated_normal(shape=(2,3), dtype=tf.float32))
        b1 = tf.Variable(tf.truncated_normal(shape=(3,), dtype=tf.float32))

        layer1 = tf.nn.relu(tf.matmul(noisyX, W1) + b1)

        W2 = tf.Variable(tf.truncated_normal(shape=(3,3), dtype=tf.float32))
        b2 = tf.Variable(tf.truncated_normal(shape=(3,), dtype=tf.float32))

        layer2 = tf.nn.relu(tf.matmul(layer2, W2) + b2)

        W3 = tf.Variable(tf.truncated_normal(shape=(3,3), dtype=tf.float32))
        b3 = tf.Variable(tf.truncated_normal(shape=(3,), dtype=tf.float32))

        hidden = tf.nn.relu(tf.matmul(layer2, W2) + b2)

        W4 = tf.Variable(tf.truncated_normal(shape=(3,3), dtype=tf.float32))
        b4 = tf.Variable(tf.truncated_normal(shape=(3,), dtype=tf.float32))

        layer3 = tf.nn.relu(tf.matmul(hidden, W4) + b4)

        W5 = tf.Variable(tf.truncated_normal(shape=(3,3), dtype=tf.float32))
        b5 = tf.Variable(tf.truncated_normal(shape=(3,), dtype=tf.float32))

        layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

        W6 = tf.Variable(tf.truncated_normal(shape=(3,2), dtype=tf.float32))
        b6 = tf.Variable(tf.truncated_normal(shape=(2,), dtype=tf.float32))
       
        output = tf.nn.relu(tf.matmul(layer5, W6) + b6)

        loss = tf.reduce_mean(tf.reduce_sum((noisyX - realX)**2, axis = -1), axis = -1)
        

if __name__ == "__main__":
    X = np.random.randn((1000,), 0, 1).astype(np.float32)
    Xnoisy = X + np.random.randn((1000,), 0, 1.1).astype(np.float32)
    Y = np.sin(X).astype(np.float32)
    Ynoisy = Y + np.random.randn((1000,), 0, 1.1).astype(np.float32)
    indata = np.vstack(X, Y).T
