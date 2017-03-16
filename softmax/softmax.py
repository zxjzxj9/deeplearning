#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys

class TFSoftMax(object):

    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label

    def train(self, batchsize = 100):

        wsize = self.data.shape[1]
        bsize = self.label.shape[1]

        W = tf.Variable(tf.zeros((wsize, bsize), dtype=tf.float32))
        b = tf.Variable(tf.zeros((bsize,), dtype=tf.float32))
        #W = tf.Variable(np.random.randn(wsize, bsize).astype(np.float32))
        #b = tf.Variable(np.random.randn(bsize).astype(np.float32))

        x = tf.placeholder(tf.float32, [None, wsize])
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder(tf.float32, [None, bsize])

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=[1]))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-10), axis=[1]))
        minimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        #tf.initialize_all_variables().run()

        for _ in range(1000):
            select = np.random.randint(0, 60000, batchsize)
            xs, ys = self.data[select, :], self.label[select, :]
            #print np.max(xs),np.min(xs),np.max(ys),np.min(ys)
            #print xs.shape, ys.shape
            sess.run(minimizer, feed_dict = {x: xs, y_: ys})                            
            #print(sess.run(tf.log(y), feed_dict = {x: xs, y_: ys}))
            #print(sess.run(W, feed_dict = {x: xs, y_: ys}))
            print(sess.run(cross_entropy, feed_dict = {x: xs, y_: ys}))
            #if i > 1: break
            #break

    def test(self, test_data, test_label):
        self.test_data = test_data
        self.test_label = test_label

if __name__ == "__main__":
   sys.path.append("../") 
   import DataReader
   tdata = DataReader.ImageReader("../dataset/train-images-idx3-ubyte.gz").to_tensor()
   ldata = DataReader.LabelReader("../dataset/train-labels-idx1-ubyte.gz").to_tensor()
   print tdata.shape
   print ldata.shape
   tf_softmax = TFSoftMax(tdata, ldata)
   tf_softmax.train()
