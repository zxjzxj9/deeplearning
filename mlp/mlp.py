#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys

class TFMLP(object):

    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label

    def train(self, batchsize = 100, hsize = 400):

        # hsize is the hidden_layer size
        wsize = self.data.shape[1]
        bsize = self.label.shape[1]

        W1 = tf.Variable(tf.random_normal((wsize, hsize), dtype=tf.float32))
        b1 = tf.Variable(tf.random_normal((hsize, ), dtype=tf.float32))
        W2 = tf.Variable(tf.random_normal((hsize, hsize), dtype=tf.float32))
        b2 = tf.Variable(tf.random_normal((hsize, ), dtype=tf.float32))
        W3 = tf.Variable(tf.random_normal((hsize, bsize), dtype=tf.float32))
        b3 = tf.Variable(tf.random_normal((bsize, ), dtype=tf.float32))
        #W = tf.Variable(tf.zeros((wsize, bsize), dtype=tf.float32), name="W")
        #W = tf.Variable(tf.zeros((wsize, bsize), dtype=tf.float32), name="W")
        #b = tf.Variable(tf.zeros((bsize,), dtype=tf.float32), name="b")
        #W = tf.Variable(np.random.randn(wsize, bsize).astype(np.float32))
        #b = tf.Variable(np.random.randn(bsize).astype(np.float32))

        x = tf.placeholder(tf.float32, [None, wsize])
        y_ = tf.placeholder(tf.float32, [None, bsize])

        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2), b2))
        y =  tf.nn.softmax(tf.add(tf.matmul(layer2, W3), b3))

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=[1]))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        #cross_entropy = tf.reduce_mean(tf.pow(y_ - y, 2))
        
        minimizer = tf.train.AdamOptimizer(5e-3).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        #tf.initialize_all_variables()
       
        self.batch = batchsize
        saver = tf.train.Saver()
        for i in range(40):
            print("Epoch %d" %i)
            for xs, ys in self:
                #print np.max(xs),np.min(xs),np.max(ys),np.min(ys)
                #print xs.shape, ys.shape
                #print(sess.run(W1, feed_dict = {x: xs, y_: ys}))
                sess.run(minimizer, feed_dict = {x: xs, y_: ys})                            
                #ydata = sess.run(y, feed_dict = {x: xs, y_: ys})
                #print(sess.run(layer1, feed_dict = {x: xs, y_: ys}))
                #print(sess.run(layer2, feed_dict = {x: xs, y_: ys}))
                #print(sess.run(y, feed_dict = {x: xs, y_: ys}))
            print(sess.run(cross_entropy, feed_dict = {x: self.data, y_: self.label}))
                #print(sess.run(tf.log(y), feed_dict = {x: xs, y_: ys}))
                #print(sess.run(W, feed_dict = {x: xs, y_: ys}))
                #print np.min(ydata), np.max(ydata)
                #if i > 2: break
                #break
        saver.save(sess, "/tmp/model.ckpt")
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

    def test(self, test_data, test_label):

        
        wsize = self.data.shape[1]
        bsize = self.label.shape[1]

        self.test_data = test_data
        self.test_label = test_label
        
        x = tf.placeholder(tf.float32, [None, wsize])
        layer1 = tf.sigmoid(tf.matmul(x, self.W1) + self.b1)
        layer2 = tf.sigmoid(tf.matmul(layer1, self.W2) + self.b2)
        y =  tf.nn.softmax(tf.matmul(layer2, self.W3) + self.b3)
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

    def __iter__(self):
        self.maxlen = self.data.shape[0]
        self.index = np.arange(self.maxlen)
        np.random.shuffle(self.index)
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt == self.maxlen: raise StopIteration
        self.cnt += self.batch
        return self.data[self.index[self.cnt - self.batch: self.cnt], :], \
               self.label[self.index[self.cnt - self.batch: self.cnt], :]

    def next(self):
        return self.__next__()


if __name__ == "__main__":
    print tf.__version__
    sys.path.append("../") 
    import DataReader
    tdata = DataReader.ImageReader("../dataset/train-images-idx3-ubyte.gz").to_tensor()
    ldata = DataReader.LabelReader("../dataset/train-labels-idx1-ubyte.gz").to_tensor()
    print tdata.shape
    print ldata.shape
    tf_mlp = TFMLP(tdata, ldata)
    tf_mlp.train()

    ttest = DataReader.ImageReader("../dataset/t10k-images-idx3-ubyte.gz").to_tensor()
    ltest = DataReader.LabelReader("../dataset/t10k-labels-idx1-ubyte.gz").to_tensor()

    tf_mlp.test(ttest, ltest)
