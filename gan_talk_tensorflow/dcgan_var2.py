# coding: utf-8

import tensorflow as tf
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import sys
import glob
import os

from tensorflow.python.platform import flags
import argparse
#tf.app.flags.FLAGS = flags._FlagValues()
#tf.app.flags._global_parser = argparse.ArgumentParser()
tf.app.flags.DEFINE_integer(name="epochs", help="number of training epoches", default=1000)
tf.app.flags.DEFINE_integer(name="crop_height", help="image cropping height", default=500)
tf.app.flags.DEFINE_integer(name="crop_width", help="image cropping width", default=500)
tf.app.flags.DEFINE_integer(name="target_height", help="image resize height", default=64)
tf.app.flags.DEFINE_integer(name="target_width", help="image resize width", default=64)
tf.app.flags.DEFINE_integer(name="batch_size", help="image batchsize", default=128)
tf.app.flags.DEFINE_float(name="learning_rate", help="learning rate", default=2e-4)
tf.app.flags.DEFINE_float(name="prior_scaling", help="scale of prior", default=1e0)
tf.app.flags.DEFINE_boolean(name="is_training", help="whether the model is at training stage", default=True)
tf.app.flags.DEFINE_string(name="log_dir", help="The log directory", default="./log")
tf.app.flags.DEFINE_string(name="visible_gpu", help="which gpu is visible", default="0")


os.environ["CUDA_VISIBLE_DEVICES"]=tf.app.flags.FLAGS.visible_gpu

class DataProcess(object):
    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.label_batch, self.img_batch = self._process()
    def _process(self):
        
        def img_process(fn):
            img = tf.image.decode_image(tf.read_file(fn))
            cropped = tf.image.resize_image_with_crop_or_pad(img, tf.app.flags.FLAGS.crop_height, tf.app.flags.FLAGS.crop_width)
            new_img = tf.image.resize_images(cropped, (tf.app.flags.FLAGS.target_height, tf.app.flags.FLAGS.target_width), method = 
                                             tf.image.ResizeMethod.AREA)
            return fn, new_img

        filenames = tf.constant(glob.glob(os.path.join(self.src_dir,"*")))
        dataset = tf.data.Dataset.from_tensor_slices((filenames, ))
        dataset = dataset.map(img_process)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(tf.app.flags.FLAGS.batch_size)
        dataset = dataset.repeat(tf.app.flags.FLAGS.epochs)

        iterator = dataset.make_one_shot_iterator()

        labels, imgs = iterator.get_next()
        return labels, imgs
        
    def get_data(self):
        return self.label_batch, self.img_batch

#dp = DataProcess("../dataset/102flowers/jpg/")
#dp._process()

#with tf.Session() as s:
#    s.run(tf.global_variables_initializer())
#    s.run(tf.local_variables_initializer())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while True:
#            labels, imgs = s.run(dp.get_data(), feed_dict = {})
#            #print(np.min(imgs))
#            plt.imshow(imgs[0,:,:,:]/256.0)
#            plt.show()
#            break
#    except tf.errors.OutOfRangeError as e:
#        print("fetch data ended")
#    coord.request_stop()
#    coord.join(threads)

class Generator(object):
    """
    Generator in GAN, used to generate "fake" images
    Original paper: https://arxiv.org/pdf/1511.06434.pdf
    """
    def __init__(self):
        self._build_graph()
    def _build_graph(self):
        with tf.variable_scope("generator") as scope:
            print("### Print Generator Intermediate Parameter")
            self.prior = tf.placeholder(dtype=tf.float32, shape=(None, 100), name="prior_gen")
            self.is_training = tf.placeholder(dtype=tf.bool, shape = (), name="training_flag")
            prior_proj = tf.contrib.layers.fully_connected(inputs=self.prior, num_outputs=4*4*1024, 
                                                           activation_fn=None, scope="prior_projection")
            prior_proj = tf.contrib.layers.batch_norm(inputs=prior_proj, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                                  is_training= self.is_training, scope="bn0")
            conv0 = tf.reshape(prior_proj, (-1, 4, 4, 1024))
            conv1 = tf.contrib.layers.convolution2d_transpose(inputs=conv0, num_outputs=512, activation_fn=None,
                                                          kernel_size=(5,5), stride=(2,2), padding="SAME",scope="deconv1")
            conv1 = tf.contrib.layers.batch_norm(inputs=conv1, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                             is_training= self.is_training, scope="bn1")
            print(conv1.shape)
            conv2 = tf.contrib.layers.convolution2d_transpose(inputs=conv1, num_outputs=256, activation_fn=None,
                                                          kernel_size=(5,5), stride=(2,2), padding="SAME",scope="deconv2")
            conv2 = tf.contrib.layers.batch_norm(inputs=conv2, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                             is_training= self.is_training, scope="bn2")
            print(conv2.shape)
            conv3 = tf.contrib.layers.convolution2d_transpose(inputs=conv2, num_outputs=128, activation_fn=None,
                                                          kernel_size=(5,5), stride=(2,2), padding="SAME",scope="deconv3")
            conv3 = tf.contrib.layers.batch_norm(inputs=conv3, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                             is_training= self.is_training, scope="bn3")
            print(conv3.shape)
            conv4 = tf.contrib.layers.convolution2d_transpose(inputs=conv3, num_outputs=3, activation_fn=None,
                                                          kernel_size=(5,5), stride=(2,2), padding="SAME",scope="deconv4")
            self.gen_img = tf.nn.tanh(conv4)
            self.gen_img_out = tf.cast(x= tf.floor(self.gen_img*128.0 + 128.0), dtype=tf.int32)
            print(conv4.shape)
            print("### End Print Generator Intermediate Parameter")


# tf.reset_default_graph()
# g = Generator()

class Discriminator(object):
    """
    Discriminator in GAN, used to distinguish "fake" and "real" images
    """
    def __init__(self, img_gen):
        self._build_graph(img_gen)
    def _build_graph(self, image_gen):
        self.real_img = tf.placeholder(tf.float32, (None, 64, 64, 3), name="real_image")
        #real_img = (self.real_img - 128.0)/128.0
        real_img = self.real_img/255.0*2.0 - 1.0
        self.is_training = tf.placeholder(dtype=tf.bool, shape = (), name="training_flag")
        
        self.real_judge = self._discrim(real_img)
        #print(self.real_judge)
        self.fake_judge = self._discrim(image_gen.gen_img, reuse = True)
        #print(self.fake_judge)
    def _discrim(self, input_img, reuse = None):
        """
            This function will be called twice, 
            one for real images, and one for fake images.
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse: scope.reuse_variables()
            print("### Print Discriminator Intermediate Parameter")
            
            print(self.is_training)
            conv1 = tf.contrib.layers.convolution2d(inputs=input_img, num_outputs=128, padding="SAME",
                                                    kernel_size=(5,5), stride=(2,2), activation_fn=tf.nn.leaky_relu, scope = "conv1")
            #conv1 = tf.contrib.layers.batch_norm(inputs=conv1, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
            #                                 is_training= self.is_training, scope="bn1")
            print(conv1.shape)
            conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=256, padding="SAME",
                                                    kernel_size=(5,5), stride=(2,2), activation_fn=None, scope = "conv2")
            conv2 = tf.contrib.layers.batch_norm(inputs=conv2, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                             is_training= self.is_training, scope="bn2")
            print(conv2.shape)
            ###
            conv3 = tf.contrib.layers.convolution2d(inputs=conv2, num_outputs=512, padding="SAME",
                                                    kernel_size=(5,5), stride=(2,2), activation_fn=None, scope = "conv3")
            conv3 = tf.contrib.layers.batch_norm(inputs=conv3, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                             is_training= self.is_training, scope="bn3")
            print(conv3.shape)
            conv4 = tf.contrib.layers.convolution2d(inputs=conv3, num_outputs=1024, padding="SAME",
                                                    kernel_size=(5,5), stride=(2,2), activation_fn=None, scope = "conv4")
            conv4 = tf.contrib.layers.batch_norm(inputs=conv4, center=True, scale=True, activation_fn=tf.nn.leaky_relu, 
                                             is_training= self.is_training, scope="bn4")
            print(conv4.shape)
            ###
            conv5 = tf.contrib.layers.avg_pool2d(inputs=conv4, kernel_size=(4,4), stride=(4,4), padding="SAME", scope="avg_pool5")
            print(conv5.shape)
            print("### End Print Discriminator Intermediate Parameter")
            ### no need to perform sigmoid
            return tf.contrib.layers.fully_connected(inputs=tf.reshape(conv5,(-1, 1024)), num_outputs=1, 
                                                     activation_fn=None, scope="output_projection")
            
# tf.reset_default_graph()
# g = Generator()
# d = Discriminator(g)

# tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

# tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

class GANModel(object):
    def __init__(self, generator, discriminator, datasrc):
        self.generator = generator
        self.discriminator = discriminator
        self.datasrc = datasrc
        self.sess = None
        self.saver = tf.train.Saver()
    def train(self, model_path = None):
        self.sess = tf.Session()
        fake_result = self.discriminator.fake_judge
        real_result = self.discriminator.real_judge
        fake_rate = tf.reduce_mean(tf.cast(tf.nn.sigmoid(fake_result) > 0.5, tf.float32))
        #loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result, labels = tf.random_uniform(tf.shape(fake_result), minval = 0.9, maxval = 1.1)))
        #loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_result, labels = tf.random_uniform(tf.shape(real_result), minval = 0.9, maxval = 1.1)))\
        #       + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result, labels = tf.random_uniform(tf.shape(fake_result), minval = 0.0, maxval = 0.1)))

        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result , labels = tf.ones_like(fake_result)))
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_result , labels = tf.ones_like(real_result)))\
               + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_result , labels = tf.zeros_like(fake_result)))


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #update_ops = [k for k in update_ops1 if k.name.startswith("discriminator")]

        with tf.control_dependencies(update_ops):
            optim_g = tf.train.AdamOptimizer(tf.app.flags.FLAGS.learning_rate, beta1=0.5).minimize(loss_g, var_list =\
                                                                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
            optim_d = tf.train.AdamOptimizer(tf.app.flags.FLAGS.learning_rate, beta1=0.5).minimize(loss_d, var_list =\
                                                                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

        writer = tf.summary.FileWriter(tf.app.flags.FLAGS.log_dir, self.sess.graph)
        summary_g = tf.summary.scalar(name="generator_loss", tensor=loss_g)
        summary_d = tf.summary.scalar(name="discriminator_loss", tensor=loss_d)
        summary_fake_rate = tf.summary.scalar(name="fake_rate", tensor=fake_rate)
        
        if model_path:
            self.saver.restore(self.sess, model_path)
            
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        cnt_total = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        try:
            while True:
                labels, imgs = self.sess.run(self.datasrc.get_data(), feed_dict = {})
                # print(imgs.dtype)
                # print(imgs.shape)
                # First train discriminator
                # print(self.discriminator.is_training)
                # print(self.generator.is_training)
                # print(np.max(imgs/255.0*2 - 1.0), np.min(imgs/255.0*2 - 1.0))
                # sys.exit()

                loss_d_out, summary_d_out, _ = self.sess.run([loss_d, summary_d, optim_d], 
                                                                 feed_dict = {
                                                                      self.discriminator.real_img: imgs,
                                                                      self.generator.prior: \
                                                                           np.random.randn(tf.app.flags.FLAGS.batch_size, 100)* tf.app.flags.FLAGS.prior_scaling,
                                                                      self.discriminator.is_training: True,
                                                                      self.generator.is_training: True
                                                                })
                # Then train generator
                loss_g_out, summary_g_out, _ = self.sess.run([loss_g, summary_g, optim_g], 
                                                                 feed_dict = {
                                                                      self.discriminator.real_img: imgs,
                                                                      self.generator.prior: \
                                                                           np.random.randn(tf.app.flags.FLAGS.batch_size, 100)* tf.app.flags.FLAGS.prior_scaling,
                                                                      self.discriminator.is_training: True,
                                                                      self.generator.is_training: True,
                                                                })
                # Then evaluate the fake ratio
                fake_rate_out, summary_fake_rate_out = self.sess.run([fake_rate, summary_fake_rate],
                                                                        feed_dict = {
                                                                            self.generator.prior: \
                                                                                np.random.randn(tf.app.flags.FLAGS.batch_size, 100)* tf.app.flags.FLAGS.prior_scaling,
                                                                            self.discriminator.is_training: False,
                                                                            self.generator.is_training: False,
                                                                        })
                cnt_total += 1       
                writer.add_summary(summary_d_out, cnt_total)
                writer.add_summary(summary_g_out, cnt_total)
                writer.add_summary(summary_fake_rate_out, cnt_total)
                
                print("In batch %3d, Dicriminator Loss %.3f, Generator Loss %.3f, Fake Ratio %.3f\r" \
                      %(cnt_total, loss_d_out, loss_g_out, fake_rate_out), end="")
                # Save every 100 batches
                if cnt_total % 50 == 0:
                    self.saver.save(self.sess, os.path.join(tf.app.flags.FLAGS.log_dir, "model_%03d.ckpt" %(cnt_total//50)))
                    
        except tf.errors.OutOfRangeError as e:
            print("fetch data ended")
        coord.request_stop()
        coord.join(threads)
        
    def infer_gen(self, model_path = None, n_img = 1, prior=None):
        """
            After the training, now we can use generator images!
            n_img: number of images, if not given any prior
            prior: given priors, if None, then random generate
        """
        if not self.sess:
            self.sess = tf.Session()
        if not model_path:
            print("Invalid model path!")
            sys.exit()
        else:
            self.saver.restore(self.sess, model_path)
            
        if not prior:
            prior = np.random.randn(n_img, 100) * tf.app.flags.FLAGS.prior_scaling
             
        imgs = self.sess.run(self.generator.gen_img_out, feed_dict = {self.generator.prior: prior, self.generator.is_training: False})
        return imgs
            
    def infer_dis(self):
        """
            In fact, discriminator can be used to predict,
            but here we will not complete the code
        """
        pass

if __name__ == "__main__":
    #os.system("rm -rf ./log")
    #tf.reset_default_graph()
    dp = DataProcess("../dataset/102flowers/jpg/")
    g = Generator()
    d = Discriminator(g)
    gan = GANModel(generator=g, discriminator=d, datasrc=dp)
    gan.train()
    # For test only
    #imgs = gan.infer_gen(model_path="./log/model_001.ckpt", n_img = 1)
    #img = np.reshape(imgs, (64, 64, 3))
    #plt.imshow(img/256.0)
    #plt.show()

