#! /usr/bin/python

import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    a1 = tf.Variable(np.arange(10).astype(np.float32))
    a2 = tf.constant(2, tf.float32)
    s = tf.Session()
    init = tf.global_variables_initializer()
    s.run(init)
    print s.run(a1*a2)
