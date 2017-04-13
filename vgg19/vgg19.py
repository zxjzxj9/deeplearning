#! /usr/bin/python

import tensorflow as tf
import cPickle as pickle

# Graph structure from the following directory:
# https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md

class Vgg19(object):
    def __init__(self):
        pass

    def init_model(self):
        inputFig = tf.placeholder(tf.float32, shape=(224, 224, 3))
        W_conv11 = 
