#! /usr/bin/python

import tensorflow as tf
import cPickle as pickle

# Graph structure from the following directory:
# https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
# aimed at prediucting http://image-net.org/challenges/LSVRC/2014/browse-synsets

class Vgg19(object):
    def __init__(self):
        pass

    def init_model(self):
        inputFig = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        W_conv11 = tf.Variable(tf.truncated_normal((3, 3, 3, 64), 0, 0.1), name = "W_conv11")
        b_conv11 = tf.Variable(tf.truncated_normal((64,), 0, 0.1), name = "b_cov11")
        layer1 = tf.nn.relu(tf.nn.conv(inputFig, W_conv11, (1, 1, 1, 1), padding = "SAME") + b_conv11)

        # Add assertion of dimension        

        W_conv12 = tf.Variable(tf.truncated_normal((3, 3, 64, 64),  0, 0.1), name = "W_conv12")
        b_conv12 = tf.Variable(tf.truncated_normal((64,), 0, 0.1), name = "b_cov12")
        layer2 = tf.nn.relu(tf.nn.conv(layer1, W_conv12, (1, 1, 1, 1), padding = "SAME") + b_conv12)

        layer3 = tf.nn.max_pool(layer2, (1, 2, 2, 1), (1, 2, 2, 1), padding = "VALID")
        
        W_conv21 = tf.Variable(tf.truncated_normal((3, 3, 64, 128),  0, 0.1), name = "W_conv21")
        b_conv21 = tf.Variable(tf.truncated_normal((128,), 0, 0.1), name = "b_cov21")
        layer4 = tf.nn.relu(tf.nn.conv(layer3, W_conv21, (1, 1, 1, 1), padding = "SAME") + b_conv21)

        W_conv22 = tf.Variable(tf.truncated_normal((3, 3, 128, 128),  0, 0.1), name = "W_conv22")
        b_conv22 = tf.Variable(tf.truncated_normal((128,), 0, 0.1), name = "b_cov22")
        layer5 = tf.nn.relu(tf.nn.conv(layer4, W_conv22, (1, 1, 1, 1), padding = "SAME") + b_conv22)

        layer6 = tf.nn.max_pool(layer5, (1, 2, 2, 1), (1, 2, 2, 1), padding = "VALID")

        W_conv31 = tf.Variable(tf.truncated_normal((3, 3, 128, 256),  0, 0.1), name = "W_conv31")
        b_conv31 = tf.Variable(tf.truncated_normal((256,), 0, 0.1), name = "b_cov31")
        layer7 = tf.nn.relu(tf.nn.conv(layer6, W_conv31, (1, 1, 1, 1), padding = "SAME") + b_conv31)

        W_conv32 = tf.Variable(tf.truncated_normal((3, 3, 256, 256),  0, 0.1), name = "W_conv32")
        b_conv32 = tf.Variable(tf.truncated_normal((256,), 0, 0.1), name = "b_cov32")
        layer8 = tf.nn.relu(tf.nn.conv(layer7, W_conv32, (1, 1, 1, 1), padding = "SAME") + b_conv32)

        W_conv33 = tf.Variable(tf.truncated_normal((3, 3, 256, 256),  0, 0.1), name = "W_conv33")
        b_conv33 = tf.Variable(tf.truncated_normal((256,), 0, 0.1), name = "b_cov33")
        layer9 = tf.nn.relu(tf.nn.conv(layer8, W_conv33, (1, 1, 1, 1), padding = "SAME") + b_conv33)

        layer10 = tf.nn.max_pool(layer9, (1, 2, 2, 1), (1, 2, 2, 1), padding = "VALID")

        W_conv41 = tf.Variable(tf.truncated_normal((3, 3, 256, 512),  0, 0.1), name = "W_conv41")
        b_conv41 = tf.Variable(tf.truncated_normal((512,), 0, 0.1), name = "b_cov41")
        layer11 = tf.nn.relu(tf.nn.conv(layer10, W_conv41, (1, 1, 1, 1), padding = "SAME") + b_conv41)

        W_conv42 = tf.Variable(tf.truncated_normal((3, 3, 512, 512),  0, 0.1), name = "W_conv42")
        b_conv42 = tf.Variable(tf.truncated_normal((512,), 0, 0.1), name = "b_cov42")
        layer12 = tf.nn.relu(tf.nn.conv(layer11, W_conv42, (1, 1, 1, 1), padding = "SAME") + b_conv42)

        W_conv43 = tf.Variable(tf.truncated_normal((3, 3, 512, 512),  0, 0.1), name = "W_conv43"
        b_conv43 = tf.Variable(tf.truncated_normal((512,), 0, 0.1), name = "b_cov43")
        layer13 = tf.nn.relu(tf.nn.conv(layer12, W_conv43, (1, 1, 1, 1), padding = "SAME") + b_conv43)

        layer14 = tf.nn.max_pool(layer13, (1, 2, 2, 1), (1, 2, 2, 1), padding = "VALID")

        W_conv51 = tf.Variable(tf.truncated_normal((3, 3, 512, 512),  0, 0.1), name = "W_conv51")
        b_conv51 = tf.Variable(tf.truncated_normal((512,), 0, 0.1), name = "b_cov51")
        layer15 = tf.nn.relu(tf.nn.conv(layer14, W_conv51, (1, 1, 1, 1), padding = "SAME") + b_conv51)

        W_conv52 = tf.Variable(tf.truncated_normal((3, 3, 512, 512),  0, 0.1), name = "W_conv52")
        b_conv52 = tf.Variable(tf.truncated_normal((512,), 0, 0.1), name = "b_cov52")
        layer16 = tf.nn.relu(tf.nn.conv(layer15, W_conv52, (1, 1, 1, 1), padding = "SAME") + b_conv52)

        W_conv53 = tf.Variable(tf.truncated_normal((3, 3, 512, 512),  0, 0.1), name = "W_conv53")
        b_conv53 = tf.Variable(tf.truncated_normal((512,), 0, 0.1), name = "b_cov53")
        layer17 = tf.nn.relu(tf.nn.conv(layer16, W_conv53, (1, 1, 1, 1), padding = "SAME") + b_conv53)

        layer18 = tf.nn.max_pool(layer17, (1, 2, 2, 1), (1, 2, 2, 1), padding = "VALID")

        layer19 = tf.reshape(layer18, (-1, 7*7*512))

        W_fc1 = tf.Variable(tf.truncated_normal((7*7*512, 4096), 0 , 0.1), name = "W_fc1")
        b_fc1 = tf.Variable(tf.truncated_normal((4096,), 0 , 0.1), name = "b_fc1")

        layer20 = tf.nn.relu(tf.matmul(layer19, W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal((4096, 4096), 0 , 0.1), name = "W_fc2")
        b_fc2 = tf.Variable(tf.truncated_normal((4096,), 0 , 0.1), name = "b_fc2")

        layer21 = tf.nn.relu(tf.matmul(layer20, W_fc2) + b_fc2)

        W_fc2 = tf.Variable(tf.truncated_normal((4096, 4096), 0 , 0.1), name = "W_fc3")
        b_fc2 = tf.Variable(tf.truncated_normal((4096,), 0 , 0.1), name = "b_fc3")

        layer23 = tf.nn.relu(tf.matmul(layer20, W_fc3) + b_fc3)

        layer24 = tf.nn.softmax(layer23)

        # in addtion to the pooling layer, there are 24 layers in total
        



