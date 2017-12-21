#! /usr/bin/env python

"""
    Reconstruction of the neural style transfer code
    Author: Victor Zhang
"""

from tensorflow.contrib.slim import preprocessing
from tensorflow.contrib.slim import nets
import tensorflow.contrib.slim as slim
import tensorflow as tf
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class VGG(object):
    
    def __init__(self, content, style, content_names, style_names):
        """
            Suppose the content and style is a numpy array,
        """
        
        self.content_names = content_names
        self.style_names = style_names
        self.VGG_MEAN = [123.68, 116.78, 103.94]

        tf.reset_default_graph()
        content = tf.constant(content) - tf.reshape(tf.constant(self.VGG_MEAN), [1, 1, 3])
        _, self.content_layers = nets.vgg.vgg_19(tf.expand_dims(content, axis = 0), is_training = False, spatial_squeeze = False)

        layer_name, layer_value = zip(*filter(lambda x: x[0] in content_names,  self.content_layers.items()))
        init_fn = slim.assign_from_checkpoint_fn("./vgg_19.ckpt", slim.get_variables_to_restore())
        with tf.Session() as s, tf.device("/device:XLA_CPU:0"):
            init_fn(s)
            layer_value = s.run(layer_value)

        self.content_map = dict(zip(layer_name, layer_value))
        #print(content_map)
            
        tf.reset_default_graph()
        style = tf.constant(style) - tf.reshape(tf.constant(self.VGG_MEAN), [1, 1, 3])
        _, self.style_layers = nets.vgg.vgg_19(tf.expand_dims(style, axis = 0), is_training = False, spatial_squeeze =  False)
        layer_name, layer_value = zip(*filter(lambda x: x[0] in style_names,  self.style_layers.items()))
        init_fn = slim.assign_from_checkpoint_fn("./vgg_19.ckpt", slim.get_variables_to_restore())

        with tf.Session() as s, tf.device("/device:XLA_CPU:0"):
            init_fn(s)
            layer_value = s.run(layer_value)

        self.style_map = dict(zip(layer_name, layer_value))
        #print(content_map)

        tf.reset_default_graph()
        self.target = tf.Variable(np.random.randint(0, 256, content.shape), dtype = tf.float32, name = "generate_image")
        self._build_graph()

    def _build_graph(self):

        def sq_dist(mat1, mat2):
            return 0.5*tf.reduce_sum(tf.square(mat1 - mat2))

        def gram(mat1):
            tmp = tf.reshape(mat1, (-1, mat1.shape[-1]))
            size = tf.size(tmp, out_type = tf.int64)
            return tf.matmul(tmp, tmp, transpose_a = True)/tf.cast(size, tf.float32)

        def gram_dist(mat1, mat2):
            return 0.25*sq_dist(gram(mat1), gram(mat2))

        _, self.target_layers = nets.vgg.vgg_19(tf.expand_dims(self.target - tf.reshape(tf.constant(self.VGG_MEAN), [1, 1, 3]), \
                                                               axis = 0), is_training = False, spatial_squeeze =  False)

        self.loss1 = tf.reduce_sum([sq_dist(tf.constant(self.content_map[layer_name]), \
                                            self.target_layers[layer_name]) for layer_name in self.content_map])
        self.loss2 = tf.reduce_mean([gram_dist(tf.constant(self.style_map[layer_name]), \
                                            self.target_layers[layer_name]) for layer_name in self.style_map])
        
        self.loss = 1e-4*self.loss1 + self.loss2
        
    def train(self):

        s = tf.Session()

        init_fn = slim.assign_from_checkpoint_fn("./vgg_19.ckpt", slim.get_variables_to_restore(exclude = ['generate_image']))
        #optimizer = tf.train.AdamOptimizer(learning_rate = 1e-1, beta1 = 0.5, beta2 = 0.5).minimize(self.loss, var_list = [self.target])
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, options={'maxiter': 1000}, var_list = [self.target])
        
        s.run(tf.global_variables_initializer())
        init_fn(s)

        #for i in range(10000):
        #    _, loss_out = s.run([optimizer, self.loss])
        #    print("Current loss is: %.3f" %loss_out, end="\r")
        #print("")

        optimizer.minimize(s)
        loss_out = s.run(self.loss)
        print("Final loss: %.3f" %loss_out)

        plt.imshow(np.clip(s.run(self.target), 0, 255).astype(np.uint8))
        plt.show()

if __name__ == "__main__":
    content = PIL.Image.open("./Kanagawa.jpg").resize((960, 662))
    #print(content)
    #plt.imshow(content)
    #plt.show()
    content = np.array(content).astype(np.float32)
    print(content.shape)
    style = PIL.Image.open("./Starry_Night.jpg").resize((300, 239))
    #print(style)
    #plt.imshow(style)
    #plt.show()
    style = np.array(style).astype(np.float32)
    print(style.shape)

    vgg = VGG(content, style, ["vgg_19/conv4/conv4_2"], ["vgg_19/conv1/conv1_1", "vgg_19/conv2/conv2_1", "vgg_19/conv3/conv3_1", "vgg_19/conv4/conv4_1"])
    vgg.train()
    
