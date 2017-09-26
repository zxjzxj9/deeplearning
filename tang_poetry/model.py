#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import numpy as np
import os
import char_dict

"""
    This model implements a poem machine, 
    with the rhyme and tone constraint.

    @author: Victor (Xiao-Jie) Zhang
    @date: 2017/09/25

"""

class DeepPoet(object):
    
    def __init__(self, data_src, char_size, seq_len, batch_size, embedding_size):

        self.data_src = data_src
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.char_size = char_size
        self.embedding_size = embedding_size

        self.layer_num = 2
        
        self.sess = None
        self.init_state = None

        self._build_graph()
        self.saver = tf.train.Saver()

    def _build_graph(self):
        
        self.sent_src = tf.placeholder(tf.int32, shape = (None, self.seq_len), name = "input_sentence")
        self.sent_target = tf.placeholder(tf.int32, shape = (None, self.seq_len), name = "target_sentence")
        self.keep_prob = tf.placeholder(tf.float32, shape = (), name = "keep_probablity")

        self.inf_src = tf.placeholder(tf.int32, shape = (None, 1), name = "input_character")

        # https://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm
        self.inf_state = tf.placeholder(tf.float32, shape = (self.layer_num, 2, None, self.embedding_size), name = "input_char")

        with tf.variable_scope("embedding"):
            # Character embedding

            self.embedding = tf.get_variable("character_embedding", shape = (self.char_size, self.embedding_size), \
                                             dtype = tf.float32, initializer = tf.truncated_normal_initializer()\
                                             )

        input_data = tf.nn.embedding_lookup(self.embedding, self.sent_src)
        
        ###### for inf
        input_inf = tf.nn.embedding_lookup(self.embedding, self.inf_src)
        #input_inf = tf.squeeze(input_inf)

        cell_fw = []

        for layer_cnt in range(self.layer_num):
            with tf.variable_scope("rnn_layer_%d" %layer_cnt):
                cell_fw.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size, state_is_tuple = True), \
                               output_keep_prob = self.keep_prob)) 

        cell = tf.contrib.rnn.MultiRNNCell(cell_fw, state_is_tuple=True)

        layer_state = tf.unstack(self.inf_state, axis=0)
        #print(layer_state)

        rnn_tuple_state = tuple(
          [tf.nn.rnn_cell.LSTMStateTuple(layer_state[idx][0], layer_state[idx][1])
          for idx in range(self.layer_num)]
        )

        print(rnn_tuple_state)
        #print(inf_outputs)
        #print(cell.zero_state(1, tf.float32))

        outputs, state = tf.nn.dynamic_rnn(cell, input_data, dtype = tf.float32)

        inf_outputs, inf_state = tf.nn.dynamic_rnn(cell, input_inf, initial_state=rnn_tuple_state) 
        #inf_outputs, inf_state = cell(input_inf, rnn_tuple_state)
        #inf_outputs = input_inf
        #inf_state = self.inf_state

        with tf.variable_scope("output_mapping"):
            weight = tf.get_variable("weight1", shape = (self.embedding_size, self.char_size), \
                     dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer()\
            )
            bias = tf.get_variable("bias1", shape = (self.char_size, ), \
                     dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer()\
            )
            linear_out = tf.reshape(outputs, (-1, self.embedding_size))
            linear_out = tf.matmul(linear_out, weight) + bias

            inf_linear_out = tf.reshape(inf_outputs, (-1, self.embedding_size))
            inf_linear_out = tf.matmul(inf_linear_out, weight) + bias

        logits = tf.reshape(linear_out, (-1, self.seq_len, self.char_size))
        
        target = tf.one_hot(self.sent_target, depth = self.char_size, axis = -1)

        target = tf.cast(target, tf.float32)

        ce = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = target)
        # For training....
        self.loss = tf.reduce_mean(tf.reduce_mean(ce, axis = -1), axis = 0)
        # For inference...

        self.inf_prob = tf.nn.softmax(tf.reshape(inf_linear_out, (-1, 1, self.char_size)))
        self.inf_state_new = inf_state

    def train(self, save_path = None):
        
        self.sess = tf.Session()

        temp = set(tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999).minimize(self.loss)


        if save_path:
            self.saver.restore(self.sess, save_path)

        s = self.sess

        writer = tf.summary.FileWriter("./log", graph = s.graph)
        tf.summary.scalar("loss", self.loss)
        merged_summary = tf.summary.merge_all()
        cnt_total = 0

        if save_path: 
            s.run(tf.variables_initializer(set(tf.global_variables()) - temp))
        else:
            s.run(tf.global_variables_initializer())

        for epoch in range(200):
            for batch in self.data_src.gen_data():
                input_data = batch
                target_data = np.roll(batch, -1, axis = -1)
                target_data[:, -1] = 0
                loss, _, summary = s.run([self.loss, optimizer, merged_summary], feed_dict = {
                                                                      self.sent_src: input_data, \
                                                                      self.sent_target: target_data, \
                                                                      self.keep_prob : 1.0
                                                                   })
                writer.add_summary(summary, cnt_total)
                cnt_total += 1
                sys.stdout.write("In epoch %d, current loss is: %.3f\r" %(epoch, loss))

            if epoch % 10 == 9:
                self.saver.save(s, "./log/model_epoch_%d.ckpt" %(epoch + 1))

            print("")

    def keyword_summarize(self, inf_src, inf_dict, save_path=None):
        """
            Basically this module is like inference, but only generate sentence embedding,
            As well as the initial character.
        """

        if not self.sess:
            self.sess = tf.Session()
            self.saver.restore(self.sess, save_path)

        s = self.sess

        hidden_state = np.zeros([self.layer_num, 2, 1, self.embedding_size])

        prob_dist = None
        ret_char = None

        for char in inf_src.decode("utf-8"):
            #print(char)
            idx = inf_dict[char.encode("utf-8")]
            inf_src = np.expand_dims([idx], axis = -1)
              
            hidden_state_new, prob_dist = s.run([self.inf_state_new, self.inf_prob], feed_dict = { \
                self.inf_src: inf_src, \
                self.inf_state: hidden_state, \
                self.keep_prob : 1.0 \
            })

            for i in range(self.layer_num):
                hidden_state[i, 0, :, :] = hidden_state_new[i][0]
                hidden_state[i, 1, :, :] = hidden_state_new[i][1]
        
        while True:
            ret_char = np.random.choice(range(self.char_size), 1, p = np.squeeze(prob_dist))
            if ret_char != 0 and ret_char != inf_dict["\n"]:
                break

        return int(np.squeeze(ret_char)), hidden_state


    def inference(self, inf_src, inf_dict, save_path=None, hidden_state = None):
        """
            Only for 7-char Jueju
            ! Suppose batch_size is 1
        """

        if not self.sess:
            self.sess = tf.Session()
            self.saver.restore(self.sess, save_path)

        s = self.sess

        poem = [inf_src[0]]

        if hidden_state is None:
            hidden_state = np.zeros([self.layer_num, 2, len(inf_src), self.embedding_size])

        inf_src = np.expand_dims(inf_src, axis = -1)
        #print(inf_src)
        #print(hidden_state)
        #print(self.inf_state)

        for step in range(1, self.seq_len):

            hidden_state_new, prob_dist = s.run([self.inf_state_new, self.inf_prob], feed_dict = { \
                self.inf_src: inf_src, \
                self.inf_state: hidden_state, \
                self.keep_prob : 1.0 \
            })

            for i in range(self.layer_num):
                hidden_state[i, 0, :, :] = hidden_state_new[i][0]
                hidden_state[i, 1, :, :] = hidden_state_new[i][1]
            #print(hidden_state_new[1].shape)
            #print(hidden_state[:,0,:,:].shape)
            #sys.exit()
            #hidden_state[:, 0, :, :] = hidden_state_new[0]
            #hidden_state[:, 1, :, :] = hidden_state_new[1]
            #print(hidden_state)

            while True:
                nw = np.random.choice(range(self.char_size), 1, p = np.squeeze(prob_dist))
                t = nw[0]
                
                # Meet ending, focing it to be sparation character
                # Continue with the hidden state of last character
                if step % 8 == 7:
                    nw = [inf_dict["\n"]]
                    t = nw[0]
                    poem.append(t)
                    inf_src = np.expand_dims(nw, axis = -1)
                    break
        
                #print(inf_dict.id_to_char(t))
                #print(t)
                
                if t == 0: continue
                if t == inf_dict["\n"]: 
                    prob_dist = np.squeeze(prob_dist)
                    prob_dist[inf_dict["\n"]] = 0.0
                    prob_dist[0] = 0.0
                    prob_dist = prob_dist/np.sum(prob_dist)
                    continue
                ### Consider tone
                # 2,4,6 p z p r (optional)
                # 2,4,6 z p z r
                # 2,4,6 z p z 
                # 2,4,6 p z p r

                if (step%8 == 1 and step/8 == 0) or \
                   (step%8 == 5 and step/8 == 0) or \
                   (step%8 == 3 and step/8 == 1) or \
                   (step%8 == 3 and step/8 == 2) or \
                   (step%8 == 1 and step/8 == 3) or \
                   (step%8 == 5 and step/8 == 3):
                    if not inf_dict.is_ze_by_idx(t):
                        continue

                if (step%8 == 1 and step/8 == 1) or \
                   (step%8 == 5 and step/8 == 1) or \
                   (step%8 == 3 and step/8 == 0) or \
                   (step%8 == 3 and step/8 == 3) or \
                   (step%8 == 1 and step/8 == 2) or \
                   (step%8 == 5 and step/8 == 2):
                    if not inf_dict.is_ping_by_idx(t):
                        continue

                if step%8 == 6 and (step/8 == 1 or step/8 ==3):
                    if step > 8 and t == poem[step-8]: continue
                    if inf_dict.is_ze_by_idx(t):
                        continue
                    if inf_dict.is_ping_by_idx(poem[6]):
                        if not inf_dict.compare_rhyme_by_idx(t, poem[6]):
                            continue
                    if step/8 == 3:
                        if not inf_dict.compare_rhyme_by_idx(t, poem[14]):
                            continue

                poem.append(t)
                inf_src = np.expand_dims(nw, axis = -1)
                break

            print(inf_dict.get_sent(poem))
        return inf_dict.get_sent(poem)

if __name__ == "__main__":
    rdict = char_dict.RhymeDict("./rhyme.pkl")
    pl = char_dict.PoemLoader("./poem.pkl")
    pl.data_to_matrix(rdict)

    dp = DeepPoet(pl, rdict.data_len, pl.maxlen, 100, 256)

    #w, hs = dp.keyword_summarize("秋", rdict, "./log/model_epoch_30.ckpt")
    #print(dp.inference([w], rdict, "./log/model_epoch_30.ckpt", hs))
    #print(w)
    #print(rdict.id_to_char(w))
    #print(dp.keyword_summarize("冬天\n下雪", rdict, "./log/model_epoch_30.ckpt"))
    #dp.train("./log/model_epoch_70.ckpt")
    #dp.train()
    #print(dp.inference([rdict["秋"]], rdict))
    #print(dp.inference([rdict["秋"]], rdict, "./log/model_epoch_190.ckpt"))
    #print(dp.inference([rdict["喵"]], rdict, "./log/model_epoch_200.ckpt"))
    #print(dp.inference([rdict["春"]], rdict, "./log/model_epoch_100.ckpt"))
    #print(dp.inference([rdict["夏"]], rdict, "./log/model_epoch_200.ckpt"))
    #print(dp.inference([rdict["月"]], rdict, "./log/model_epoch_30.ckpt"))
    #print(dp.inference([rdict["明"]], rdict, "./log/model_epoch_10.ckpt"))
    #print(dp.inference([rdict["冬"]], rdict, "./log/model_epoch_30.ckpt"))
