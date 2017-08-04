#! /usr/bin/python

# This script implements GAN in Ian Goodfellow's original work
# http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

import tensorflow as tf
#from tensorflow.python import debug as tf_debug
import numpy as np
import sys

class DataIterator(object):
    # Suppose the data are in the form [(img, label)]
    def __init__(self, datasrc, batchsize = 10):
        self.data = datasrc
        self.size = len(self.data)
        self.bs = batchsize

    def __iter__(self):
        self.current = 0
        np.random.shuffle(self.data)
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        start = self.current
        end = self.current + self.bs
        self.current += self.bs
        if self.current > self.size: raise StopIteration
        return self.data[start:end]

class PriorIterator(object):
    def __init__(self, batch_size, prior_size):
        self.batch_size = batch_size
        self.prior_size = prior_size

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return np.random.normal(0.0, 1.0, size = (self.batch_size, self.prior_size)).astype(np.float32)

class GAN_MNIST(object):

    def __init__(self, datasrc, input_size, batch_size = 10,  noise = 0.1, prior_size = 100, hidden_size = 1200, dis_size = 240):
        self.datasrc = datasrc
        self.keep_prob = 1.0
        self.noise = noise
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dis_size = dis_size
        self.batch_size = batch_size
        self.prior_size = prior_size
        self.output_size = 10
        self.dis_vars = []
        self.gen_vars = []
        self._build_graph()

    def _build_graph(self):
        # Buid simple 3-layer mlp as the generator
        
        # generated prior 
        self.g_input = tf.placeholder(dtype = tf.float32, shape = (None, self.prior_size),  name = "GeneratorInput")

        layerg1 = tf.nn.relu(self._linear(self.g_input, self.prior_size, self.hidden_size, "gen_layer1", noisy = False))
        layerg2 = tf.nn.relu(self._linear(layerg1, self.hidden_size, self.hidden_size, "gen_layer2", noisy = False))
        #layerg3 = tf.nn.relu(self._linear(layerg2, self.input_size, self.input_size, "gen_layer3", noisy = False))
        layerg3 = tf.nn.sigmoid(self._linear(layerg2, self.hidden_size, self.input_size, "gen_layer4", noisy = False))

        #layerg3_logits = tf.nn.sigmoid(self._linear(layerg2, self.input_size, self.input_size, "gen_layer3", noisy = True))
        #layerg3 = tf.nn.sigmoid(layerg3_logits)
        
        # For inference only!
        self.image = layerg3 #tf.cast(layerg3*256, tf.int32)

        # layer3 --> layer1
        # discriminator part, for synthesis image (fake)
        layerd1_f = self._maxout(layerg3, self.input_size, self.dis_size, 5, "dis_layer1")
        #layerd1_f = tf.nn.dropout(layerd1_f, keep_prob = self.keep_prob)
        layerd2_f = self._maxout(layerd1_f, self.dis_size, self.dis_size, 5, "dis_layer2")
        #layerd2_f = tf.nn.dropout(layerd2_f, keep_prob = self.keep_prob)
        layerd3_f_logits = self._maxout(layerd2_f, self.dis_size, 1, 5, "dis_layer3")
        layerd3_f = tf.nn.sigmoid(layerd3_f_logits)
        #layerd3_f = tf.nn.sigmoid(self._linear(layerd2_f, self.input_size, 1, "dis_layer3"))

        self.fake_ratio = tf.reduce_mean(layerd3_f)

        gen_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = layerd3_f_logits, labels = tf.ones_like(layerd3_f_logits)))
        gen_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = layerd3_f_logits, labels = tf.zeros_like(layerd3_f_logits)))

        #print gen_loss1
        #sys.exit()
        #gen_loss1 = -tf.reduce_mean(tf.log(layerd3_f))
        #gen_loss2 = -0.5*tf.reduce_mean(tf.log(1.0 - layerd3_f))

        # discriminator part, for real image
        self.d_input = tf.placeholder(dtype = tf.float32, shape = (None, self.input_size), name = "DiscriminatorInput")
        layer_noise = tf.clip_by_value(self.d_input + tf.truncated_normal(tf.shape(self.d_input), stddev = self.noise), 0.0, 1.0)

        layerd1_r = self._maxout(layer_noise, self.input_size, self.dis_size, 5, "dis_layer1", reuse = True)
        #layerd1_r = tf.nn.dropout(layerd1_r, keep_prob = self.keep_prob)
        layerd2_r = self._maxout(layerd1_r, self.dis_size, self.dis_size, 5, "dis_layer2", reuse = True)
        #layerd2_r = tf.nn.dropout(layerd2_r, keep_prob = self.keep_prob)
        layerd3_r_logits = self._maxout(layerd2_r, self.dis_size, 1, 5, "dis_layer3", reuse = True)
        layerd3_r = tf.nn.sigmoid(layerd3_r_logits)

        #layerd3_r = tf.nn.sigmoid(self._linear(layerd2_r, self.input_size, 1, "dis_layer3", reuse = True))
        #dis_loss1 = -0.5*tf.reduce_mean(tf.log(layerd3_r))
        dis_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = layerd3_r_logits, labels = tf.ones_like(layerd3_r_logits)))
        
        self.total_loss1 = gen_loss1
        self.total_loss2 = dis_loss1 + gen_loss2

    def train(self):
        #gen1_opt = tf.train.AdamOptimizer(learning_rate = 1.0e-2).minimize(self.gen_loss1)
        #gen2_opt = tf.train.AdamOptimizer(learning_rate = 1.0e-2).minimize(self.gen_loss2)
        #dis_opt = tf.train.GradientDescentOptimizer(learning_rate = 1.0e-2).minimize(self.dis_loss1)
        #dis_opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.dis_loss1)
        #print(self.gen_vars)
        #sys.exit()
        #print(self.dis_vars)
        #sys.exit()

        #loss2_opt = tf.train.GradientDescentOptimizer(learning_rate = 1.0e-1).minimize(self.total_loss2, var_list = self.dis_vars)
        loss1_opt = tf.train.AdamOptimizer(learning_rate = 2.0e-3, beta1 = 0.5).minimize(self.total_loss1, var_list = self.gen_vars)
        loss2_opt = tf.train.AdamOptimizer(learning_rate = 2.0e-3, beta1 = 0.5).minimize(self.total_loss2, var_list = self.dis_vars)
        #loss2_opt = tf.train.GradientDescentOptimizer(learning_rate = 1.0e-4).minimize(self.total_loss2, var_list = self.dis_vars)

        config = tf.ConfigProto(
              device_count = {'GPU': 1}
        )

        s = tf.Session(config = config)
        # For debug
        # s = tf_debug.LocalCLIDebugWrapperSession(s)

        self.session = s
        s.run(tf.global_variables_initializer())

        total_epoch = 100

        for epoch in range(total_epoch):
            #if epoch < total_epoch/2: print("In epoch %d" %epoch)
            #else: print("In epoch %d (switched generator loss)" %epoch)
            print("In epoch %d" %epoch)

            loss1 = 0.0
            loss2 = 0.0
            cnt = 0
            fr = 0.0
            # Discriminator

            for prior, realbatch in zip(PriorIterator(self.batch_size, self.prior_size), DataIterator(self.datasrc, self.batch_size)):

                real_img, real_label = zip(*realbatch)
                real_img = np.vstack(real_img)

                #import matplotlib.pyplot as plt
                #plt.imshow(real_img[0].reshape((28,28)), cmap = "gray")
                #plt.show()
                 
                _, _, loss2, loss1, fake_ratio1 = s.run([loss2_opt, loss1_opt, self.total_loss2, self.total_loss1, self.fake_ratio], feed_dict = {self.g_input: prior, self.d_input: real_img})
                #_, loss2, fake_ratio1 = s.run([loss2_opt, loss1_opt, self.total_loss2, self.fake_ratio], feed_dict = {self.g_input: prior, self.d_input: real_img})
                #if loss1 > 2:
                #    for tmp in range(10):
                #        _, loss1, fake_ratio2 = s.run([loss1_opt, self.total_loss1, self.fake_ratio], feed_dict = {self.g_input: PriorIterator(self.batch_size, self.prior_size).next()})


                #print sum(sum(prior**2)), loss1
                #sys.stdout.write("Loss1 (Generator Only): %.3f, Loss2 (All): %.3f, Fake ratio: %.3f, %.3f\r" %(loss1, loss2, fake_ratio1, fake_ration2))
                sys.stdout.write("Loss1 (Generator Only): %.3f, Loss2 (All): %.3f, Fake ratio: %.3f\r" %(loss1, loss2, fake_ratio1))
                #fr = fake_ratio2
                cnt += 1

            ## Generator
            #for prior, _ in zip(self, range(cnt)):
            #    _, loss1, fake_ratio2 = s.run([loss1_opt, self.total_loss1, self.fake_ratio], feed_dict = {self.g_input: prior})
            #    sys.stdout.write("Loss1 (Generator Only): %.3f, Loss2 (All): %.3f, Fake ratio: %.3f\r" %(loss1, loss2, fake_ratio2))

            #if epoch + 1 == total_epoch:
            if (epoch+1)%10 ==0:
                import matplotlib.pyplot as plt
                graphs, axes = plt.subplots(nrows=5, ncols=5)
                axes = axes.flatten()

                figs = s.run(self.image, feed_dict = {self.g_input: PriorIterator(self.batch_size, self.prior_size).next()})
                for t in range(25):
                    axes[t].imshow(figs[t,:].reshape((28,28)),  cmap='gray')
                graphs.tight_layout()
                plt.show()

            print("")


    def _linear(self, layer_in, size_in, size_out, suffix = "", noisy = False, reuse = None):

        # add some noise to the layer, discriminator layer shall be reused
        with tf.variable_scope("linear_layer_%s" %suffix, reuse = reuse):
            # first layer:
            weight = tf.get_variable(name = "weight", dtype = tf.float32, shape = (size_in, size_out), \
                initializer = tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name = "bias", dtype = tf.float32, shape = (size_out, ), \
                initializer = tf.contrib.layers.xavier_initializer())

        layer_out = tf.add(tf.matmul(layer_in, weight), bias)
        if noisy: layer_out += tf.truncated_normal(tf.shape(layer_out), stddev = self.noise)

        if "dis" in suffix and (not reuse):
            self.dis_vars.append(weight)
            self.dis_vars.append(bias)
        
        if "gen" in suffix:
            self.gen_vars.append(weight)
            self.gen_vars.append(bias)

        return layer_out

    def _maxout(self, layer_in, size_in, size_out, pieces, suffix = "", noisy = False, reuse = None):
        with tf.variable_scope("maxout_layer_%s" %suffix, reuse = reuse):
            weight = tf.get_variable(name = "weight", dtype = tf.float32, shape = (size_in, size_out*pieces), \
                            initializer = tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name = "bias", dtype = tf.float32, shape = (size_out*pieces, ), \
                            initializer = tf.contrib.layers.xavier_initializer())

        layer_out = tf.reduce_max(tf.reshape(tf.add(tf.matmul(layer_in, weight), bias), (-1, size_out, pieces) ), axis = -1)
        if noisy: layer_out += tf.truncated_normal(tf.shape(layer_out), stddev = self.noise)

        if "dis" in suffix and (not reuse):
            self.dis_vars.append(weight)
            self.dis_vars.append(bias)
        
        if "gen" in suffix:
            self.gen_vars.append(weight)
            self.gen_vars.append(bias)

        return layer_out


if __name__ == "__main__":
    print tf.__version__
    sys.path.append("../") 
    import DataReader
    tdata = DataReader.ImageReader("../dataset/train-images-idx3-ubyte.gz").to_tensor()
    # For compatibility
    ldata = np.argmax(DataReader.LabelReader("../dataset/train-labels-idx1-ubyte.gz").to_tensor(), axis = -1)
    print tdata.shape
    print ldata.shape
    gm = GAN_MNIST(zip(tdata, ldata), tdata.shape[-1], batch_size = 100, noise = 0.10)
    gm.train()    

