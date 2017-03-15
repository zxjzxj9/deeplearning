#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class TFSoftMax(object):

    def __init__(self, trainData, trainLabel):
        self.data = trainData
        self.label = trainLabel

    def train(self, batchsize = 100):
        #pass
        wsize = self.data.shape[1]
        bsize = self.label.shape[1]

        W = tf.Variables(np.random.randn(wsize, bszie))
        b = tf.Variables(np.random.randn(bsize))

        x = tf.placeholder(tf.float32, [None, wsize])
        y = tf.nn.softmax(matmul(x*w) + b)
        y_ = tf.placeholder(tf.float32, [None, bsize])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(


if __name__ == "__main__":
    
