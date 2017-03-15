#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class TFSoftMax(object):

    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label

    def train(self, batchsize = 100):

        wsize = self.data.shape[1]
        bsize = self.label.shape[1]

        W = tf.Variables(np.random.randn(wsize, bszie))
        b = tf.Variables(np.random.randn(bsize))

        x = tf.placeholder(tf.float32, [None, wsize])
        y = tf.nn.softmax(matmul(x*w) + b)
        y_ = tf.placeholder(tf.float32, [None, bsize])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(

    def test(self, test_data, test_label):
        self.test_data = test_data
        self.test_label = test_label

if __name__ == "__main__":
    
