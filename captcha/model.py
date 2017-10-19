#! /usr/bin/env python

import tensorflow as tf
from PIL import Image
from collections import namedtuple
import glob
import random
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

class DataReader(object):
    mapping = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"\
              "abcdefghijklmnopqrstuvwxyz"\
              "0123456789"
    
    def __init__(self, datasrc, training_size = 1000):
        self.all_files = glob.glob("./data/*png")
        self.train = self.all_files[:-training_size]
        self.test = self.all_files[-training_size:]
        self.train_size = len(self.train)
        self.test_size = len(self.test)
        self.blank_idx = len(self.mapping) 

    def get_iter(self, padding_size, batch_size = 100):
        random.shuffle(self.train)
        maxbatch = int(math.ceil(self.train_size/float(batch_size)))

        def generator():
            for i in range(0, maxbatch):
                upper_bound = min((i+1)*batch_size, self.train_size)
                fns = self.train[i*batch_size: upper_bound]
                imgs = np.stack(map(lambda x: np.array(Image.open(x)                \
                             .resize((128, 32), Image.ANTIALIAS)), fns))            \
                             .astype(np.float32)/float(256)

                #plt.imshow(imgs[0, :, :, :])
                #plt.show()
                #print(imgs.shape)
                #sys.exit()

                # suitable for variational label length
                #print(os.path.basename(fns[0]).split(".")[0])

                label_idx = map(lambda x: map(lambda y: self.mapping.index(y)
                                ,os.path.basename(x).split(".")[0]), fns)

                label_idx_padded = map(lambda y: np.pad(y, (0, \
                       padding_size - len(y)), \
                       mode = "constant", \
                       constant_values = self.blank_idx), label_idx)

                labels = np.stack(label_idx_padded)

                yield imgs, self.labels_to_sparse(labels, batch_size), \
                      np.array(map(lambda x : len(x), label_idx))

        return generator()

    def get_test(self, batch_size = 100):
        maxbatch = int(math.ceil(self.test_size/float(batch_size)))

        def generator():
            for i in range(0, maxbatch):
                upper_bound = min((i+1)*batch_size, self.train_size)
                fns = self.test[i*batch_size: upper_bound]

                imgs = np.stack(map(lambda x: np.array(Image.open(x)                \
                             .resize((128, 32), Image.ANTIALIAS)), fns))            \
                             .astype(np.float32)/float(256)
                labels = map(lambda x: os.path.basename(x)\
                             .split(".")[0], fns)
                yield imgs, labels

        return generator()    
        
    def labels_to_sparse(self, labels, batch_size):
        
        loc = []
        val = []
        maxbound = 0

        for line_num, data in enumerate(labels):
            tmp_loc = zip(zip([line_num]*len(data), range(len(data))), data)
            _1, _2 = zip(*filter(lambda x: x[1] != 0, tmp_loc))
            loc += _1
            val += _2
            #loc.append(_1)
            #val.append(_2)

            maxbound = max(len(data), maxbound)

        return np.array(loc), np.array(val), np.array((batch_size, maxbound))

    def dense_to_labels(self, dense_labels):

        ret = []
        for line in dense_labels:
            # filter out blank index
            l = filter(lambda x : x != self.blank_idx, line)
            l = "".join(map(lambda x: self.mapping[x], l))
            ret.append(l)
        return ret
        

class Captcha(object):

    def __init__(self, datasrc, config):
        self.datasrc = datasrc
        self.config = config
        self._build_graph()
        self.sess = None
        self.saver = tf.train.Saver(max_to_keep=10)

    def _build_graph(self):
        
        # For dubug only
        self.assert_ops = []

        # Input figure
        self.input = tf.placeholder(tf.float32, (None, self.config.input_h, self.config.input_w, 3), name = "sequence")
        self.output = tf.sparse_placeholder(tf.int32)
        self.seq_len = tf.placeholder(tf.int32, (None, ), name = "sequence_length")
        #self.output = tf.placeholder(tf.float32, (None, self.config.output_len))
        self.keep_prob = tf.placeholder(tf.float32, shape = (), name = "keep_probablity")

        # Convolutional Layer:
        with tf.variable_scope("conv"):
            # 3x3 filter, 3 => 10
            conv1 = tf.get_variable("conv1", shape=(3, 3, 3, 10), dtype=tf.float32, \
                                     initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.relu(tf.nn.conv2d(self.input, conv1, strides = (1,1,1,1), padding = "SAME"))

            # 3x3 filter, 10 => 10
            conv2 = tf.get_variable("conv2", shape=(3, 3, 10, 10), dtype=tf.float32, \
                                     initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.nn.relu(tf.nn.conv2d(layer1, conv2, strides = (1,1,1,1), padding = "SAME"))
    
            # 1x1 filter, 10 => 10
            conv3 = tf.get_variable("conv3", shape=(1, 1, 10, 10), dtype=tf.float32, \
                                     initializer=tf.contrib.layers.xavier_initializer())
            layer3 = tf.nn.relu(tf.nn.conv2d(layer2, conv3, strides = (1,1,1,1), padding = "SAME"))
        
        ### Transform the picture
        layer3 = tf.transpose(layer3, (0, 2, 1, 3))
        layer3 = tf.reshape(layer3, (-1, self.config.input_w, self.config.input_h * 10))

        #rnn_input = tf.concat(tf.split(layer3, self.config.split_num, axis = 1), axis = -1)
        rnn_input = tf.stack(tf.split(layer3, self.config.split_num, axis = 1), axis = 1)
        #print(rnn_input)
        #sys.exit()
        rnn_size = self.config.input_w / self.config.split_num * self.config.input_h * 10
        rnn_input = tf.reshape(rnn_input, (-1, self.config.split_num, rnn_size))

        #self.assert_ops.append(tf.assert_equal(tf.shape(rnn_input), tf.constant([100, 16, 2560])))

        #print(rnn_input)
        #sys.exit()
        #Feeding CNN into RNN, to recognize number
        
        with tf.variable_scope("rnn"):
            cell_fw = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size), \
                       output_keep_prob = self.keep_prob)]*self.config.num_layer

            cell_bw = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size), \
                       output_keep_prob = self.keep_prob)]*self.config.num_layer

            cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw)
            cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_input, dtype = tf.float32)

        with tf.variable_scope("output_mapping"):
            weight = tf.get_variable("weight1", shape = (rnn_size * 2, self.config.output_dim + 1), \
                     dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer()\
            )
            bias = tf.get_variable("bias1", shape = (self.config.output_dim + 1, ), \
                     dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer()\
            )

            outputs = tf.concat(outputs, axis = -1)
            #print(outputs)
            #sys.exit()
            outputs = tf.reshape(outputs, (-1, rnn_size*2))
            outputs = tf.matmul(outputs, weight) + bias
            #print(outputs)
            #sys.exit()
            outputs =  tf.reshape(outputs, (-1, self.config.split_num, self.config.output_dim + 1))
            # CTC loss don't need softmax...
            # outputs = tf.nn.softmax(tf.reshape(outputs, (-1, 16, self.config.output_dim)))
        
        #self.assert_ops.append(tf.assert_equal(tf.shape(outputs), tf.constant([100, 16, 63])))
        #print(self.output, outputs)
        #sys.exit()
        self.tmp = outputs

        #self.cost = tf.nn.ctc_loss(self.output, outputs, self.seq_len, time_major = False)
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.output, outputs, self.seq_len, time_major = False))

        outputs = tf.transpose(outputs, (1, 0, 2))
        self.target, self.log_prob  = tf.nn.ctc_beam_search_decoder(outputs, self.seq_len, top_paths = 1, merge_repeated=False) 
        self.target = tf.sparse_tensor_to_dense(self.target[0])

    def train(self):
        
        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate, \
                    beta1 = 0.9, beta2 = 0.999).minimize(self.loss)

        #grads = optimizer.compute_gradients(self.loss)
        #for i, (g, v) in enumerate(grads):
        #    if g is not None:
        #        grads[i] = (tf.clip_by_norm(g, 5), v)
        #train_op = optimizer.apply_gradients(grads)

        self.sess = tf.Session()
        s = self.sess

        writer = tf.summary.FileWriter("./log", graph = s.graph)
        tf.summary.scalar("loss", self.loss)


        merged_summary = tf.summary.merge_all()
        cnt_total = 0
        s.run(tf.global_variables_initializer())

        for epoch in range(self.config.epoch_num):
            print("In epoch %d " %epoch)
            cnt = 0
        
            for img, label, seq_len in self.datasrc.get_iter(16, self.config.batch_size):
                #print(img.shape)
                #print(label[0].shape, label[1].shape, label[2].shape)
                #print(seq_len.shape)
                #print(tf.SparseTensorValue(*label))

                #data = s.run(self.assert_ops, feed_dict = { \
                #    self.input : img,
                #    self.output : tf.SparseTensorValue(*label),
                #    self.seq_len : seq_len,
                #    self.keep_prob : 1.0,
                #})

                #print(data)
                #sys.exit()
                #loss, data, summary = s.run([self.loss, self.tmp, merged_summary], feed_dict = { \
                #    self.input : img,
                #    self.output : tf.SparseTensorValue(*label),
                #    self.seq_len : [self.config.split_num]*self.config.batch_size,
                #    self.keep_prob : 1.0,
                #})
                #print(loss)
                #print(np.max(np.abs(data)))
                #sys.exit()

                loss, _, summary = s.run([self.loss, optimizer, merged_summary], feed_dict = { \
                    self.input : img,
                    self.output : tf.SparseTensorValue(*label),
                    self.seq_len : [self.config.split_num]*len(seq_len),
                #    self.seq_len : seq_len,
                    self.keep_prob : 1.0,
                })

                #print("loss %f" %loss)
                
                writer.add_summary(summary, cnt_total)
                sys.stdout.write("Current loss: %.3e, current batch: %d \r" %(loss,cnt))
                cnt += 1
                cnt_total += 1

            if epoch % 5 == 4:
                self.saver.save(s, "./log/model_epoch_%d.ckpt" %(epoch + 1))
        print("")

    def predict(self, input_src, model_path = None):
        
        if not self.sess:
            self.sess = tf.Session()
            self.saver.restore(self.sess, model_path)

        s = self.sess
        
        target, log_prob = s.run([self.target, self.log_prob], feed_dict = { \
            self.input: img,
            self.seq_len : [self.config.split_num] * len(img),
            self.keep_prob : 1.0,
        })

        return target, log_prob


if __name__ == "__main__":
    dr = DataReader("./data")

    #fns = ["0APguy.png"]
    #label_idx = map(lambda x: map(lambda y: dr.mapping.index(y)
    #                ,os.path.basename(x).split(".")[0]), fns)

    #label_idx_padded = map(lambda y: np.pad(y, (0, \
    #       16 - len(y)), \
    #       mode = "constant", \
    #       constant_values = dr.blank_idx), label_idx)

    #labels = np.stack(label_idx_padded)

    #print(labels)
    #print(dr.labels_to_sparse(labels, 1))

    #sys.exit()

    Config = namedtuple("NeuralNetworkConfig", "batch_size, input_h, input_w, rnn_hidden, output_dim, " \
                                               "output_len, learning_rate, num_layer, epoch_num, split_num, " \
    )
    conf = Config(batch_size = 100, input_h = 32, input_w = 128, rnn_hidden = 320, \
                  output_len = 6, output_dim = 62, learning_rate = 1e-3, num_layer = 1, \
                  epoch_num = 10, split_num = 16 \
                 )

    captcha = Captcha(dr, conf)
    #captcha.train()

    cnt_right = 0 
    cnt_total = 0

    for img, labels in dr.get_test(100):
        target, prob = captcha.predict(img, "./log/model_epoch_5.ckpt")
        #print(target)
        tgt = dr.dense_to_labels(target)
        for i, (a,b) in enumerate(zip(tgt, labels)):
            cnt_total += 1
            if a == b:
                cnt_right += 1
            else:
                print(target[i])
                print(a,b)
                plt.imshow(img[i,:,:,:])
                plt.show()
                sys.exit()
    print("Final accuracy: %.3f" %(cnt_right/float(cnt_total)))
