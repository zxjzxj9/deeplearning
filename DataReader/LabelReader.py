#! /usr/bin/env python
#-*- coding: utf-8 -*-


"""

-- @author Victor Xiao-Jie Zhang
-- @date 2017/3/13
-- @info Class to read the MNIST Labels
-- @version 0.1

"""

import gzip
import struct
import numpy as np

class LabelReader(object):

    def __init__(self, filename):
        if filename.endswith(".gz"):
            print("Reading gzip files:")
            self.fdata = gzip.open(filename, "rb")
        else:
            self.fdata = open(filename, "rb")
        
        self.magic = struct.unpack(">i", self.fdata.read(4))[0]
        #print("%x" %self.magic)
        if self.magic != 0x00000801:
            raise Exception("Magic Number Error! Check if this file is MNIST Image file!")
        self.nlabel = struct.unpack(">i", self.fdata.read(4))[0]
        print("Label numbers: {0}".format(self.nlabel))


    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        tmp = self.fdata.read(1)
        if tmp == "": raise StopIteration
        idx = struct.unpack("b", tmp)[0]
        ret = np.zeros(10, dtype=np.int32)
        ret[idx] = 1
        return ret

    def to_tensor(self):
        return np.array(list(self)).astype(np.float32)

    def __del__(self):
        self.fdata.close()
    
