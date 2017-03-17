#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""

-- @author Victor Xiao-Jie Zhang
-- @date 2017/3/13
-- @info Class to read the MNIST Images
-- @version 0.1

"""

import gzip
import struct
import numpy as np

class ImageReader(object):
    
    def __init__(self, filename):
        if filename.endswith(".gz"):
            print("Reading gzip files:")
            self.fdata = gzip.open(filename, "rb")
        else:
            self.fdata = open(filename, "rb")

        self.magic = struct.unpack(">i", self.fdata.read(4))[0]
        #print("%x" %self.magic)
        if self.magic != 0x00000803:
            raise Exception("Magic Number Error! Check if this file is MNIST Image file!")
        self.nimage = struct.unpack(">i", self.fdata.read(4))[0]
        print("Image numbers: {0}".format(self.nimage))
        self.nrows = struct.unpack(">i", self.fdata.read(4))[0]
        self.ncols = struct.unpack(">i", self.fdata.read(4))[0]
        print("Image size: {0} x {1}".format(self.nrows,self.ncols))

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        tmp = self.fdata.read(self.nrows*self.ncols)
        if tmp == "": raise StopIteration
        #image = np.array(struct.unpack("{0}B", tmp), dtype=np.byte). \
        #                reshape((self.nrows, self.ncols))
        # notice we shall normalize the input
        image = np.array(struct.unpack("{0}B".format(self.nrows*self.ncols), tmp), dtype=np.float32) / 256.0
        #print np.max(image), np.min(image)
        return image

    def to_tensor(self):
        return np.array(list(self)).astype(np.float32)

    def __del__(self):
        self.fdata.close()
