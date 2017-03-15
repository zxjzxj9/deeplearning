#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class TFSoftMax(object):

    def __init__(self, train_data, train_label):
        self.data = train_data
        self.label = train_label

    def train(self):
        y = tf.placeholder(shape = (None, 10))
        x = tf.placeholder(shape = ())
    
    def test(self, test_data, test_label):
        self.test_data = test_data
        self.test_label = test_label

if __name__ == "__main__":
    
