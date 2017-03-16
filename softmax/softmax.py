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

        W = tf.Variable(tf.zeros((wsize, bsize), dtype=tf.float32), name="W")
        b = tf.Variable(tf.zeros((bsize,), dtype=tf.float32), name="b")
        #W = tf.Variable(np.random.randn(wsize, bsize).astype(np.float32))
        #b = tf.Variable(np.random.randn(bsize).astype(np.float32))

        x = tf.placeholder(tf.float32, [None, wsize])
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder(tf.float32, [None, bsize])

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=[1]))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        minimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        #tf.initialize_all_variables()
        
        saver = tf.train.Saver()
        for i in range(10000):
            select = np.random.randint(0, 60000, batchsize)
            xs, ys = self.data[select, :], self.label[select, :]
            #print np.max(xs),np.min(xs),np.max(ys),np.min(ys)
            #print xs.shape, ys.shape
            ydata = sess.run(y, feed_dict = {x: xs, y_: ys})
            sess.run(minimizer, feed_dict = {x: xs, y_: ys})                            
            #print(sess.run(cross_entropy, feed_dict = {x: xs, y_: ys}))
            #print(sess.run(tf.log(y), feed_dict = {x: xs, y_: ys}))
            #print(sess.run(W, feed_dict = {x: xs, y_: ys}))
            #print np.min(ydata), np.max(ydata)
            #if i > 1: break
            #break
        saver.save(sess, "/tmp/model.ckpt")
        self.W = W
        self.b = b

    def test(self, test_data, test_label):

        
        wsize = self.data.shape[1]
        bsize = self.label.shape[1]

        #W = tf.Variable(tf.zeros((wsize, bsize), dtype=tf.float32), name="W")
        #b = tf.Variable(tf.zeros((bsize,), dtype=tf.float32), name="b")

        self.test_data = test_data
        self.test_label = test_label
        
        x = tf.placeholder(tf.float32, [None, wsize])
        y = tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        y_ = tf.placeholder(tf.float32, [None, bsize])

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/model.ckpt")
        #init = tf.global_variables_initializer()
        #sess.run(init)

        data_same = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
        mean_same = tf.reduce_mean(tf.cast(data_same, tf.float32))

        correct = sess.run(mean_same, feed_dict = {x: test_data, y_: test_label})
        print correct


if __name__ == "__main__":
    print tf.__version__
    sys.path.append("../") 
    import DataReader
    tdata = DataReader.ImageReader("../dataset/train-images-idx3-ubyte.gz").to_tensor()
    ldata = DataReader.LabelReader("../dataset/train-labels-idx1-ubyte.gz").to_tensor()
    print tdata.shape
    print ldata.shape
    tf_softmax = TFSoftMax(tdata, ldata)
    tf_softmax.train()


    ttest = DataReader.ImageReader("../dataset/t10k-images-idx3-ubyte.gz").to_tensor()
    ltest = DataReader.LabelReader("../dataset/t10k-labels-idx1-ubyte.gz").to_tensor()

    tf_softmax.test(ttest, ltest)

    tf_softmax.test
    
