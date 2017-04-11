#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys

class TFCNN(object):

    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def train(self, batchsize = 100, hsize = 400):
        self.batch = batchsize
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
       
        hsize = self.data.shape[1]
        wsize = self.data.shape[2]
        bsize = self.label.shape[1]

        x = tf.placeholder(tf.float32, [None, hsize, wsize])
        y_ = tf.placeholder(tf.float32, [None, bsize])
 
        # first conv and pooling
        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        # second conv and pooling
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
   
        # flatting, full connection 
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout layer
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            for xs, ys in self:
                sess.run(train_step, feed_dict = {x: xs, y_: ys, keep_prob: 0.5}) 
            if i%10 == 0:
                print sess.run(cross_entropy,  feed_dict = {x: xs, y_: ys, keep_prob: 1.0})

        self.accuracy = accuracy
        saver = tf.train.Saver()
        saver.save(sess, "/tmp/model.ckpt")
        self.x = x
        self.y_ = y_
        self.keep_prob = keep_prob
        #print sess.run(accuracy)

    def test(self, test_data, test_label):
        hsize = self.data.shape[1]
        wsize = self.data.shape[2]
        bsize = self.label.shape[1]
        print bsize, test_data.dtype, test_label.dtype

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/model.ckpt")
        print sess.run(self.accuracy, feed_dict = {self.x: test_data, self.y_: test_label, self.keep_prob: 1.0})
        
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
    tdata = DataReader.ImageReader("../dataset/train-images-idx3-ubyte.gz").to_tensor2d()
    ldata = DataReader.LabelReader("../dataset/train-labels-idx1-ubyte.gz").to_tensor()
    print tdata.shape
    print ldata.shape
    tf_mlp = TFCNN(tdata, ldata)
    tf_mlp.train()

    ttest = DataReader.ImageReader("../dataset/t10k-images-idx3-ubyte.gz").to_tensor2d()
    ltest = DataReader.LabelReader("../dataset/t10k-labels-idx1-ubyte.gz").to_tensor()
    print ttest.dtype
    print ltest.dtype
    tf_mlp.test(ttest, ltest)
