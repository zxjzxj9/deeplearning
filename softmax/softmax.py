#! /usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class TFSoftMax(object):

    def __init__(self, trainData, trainLabel):
        self.data = trainData
        self.label = trainLabel

    def train(self):
        pass

if __name__ == "__main__":
    
