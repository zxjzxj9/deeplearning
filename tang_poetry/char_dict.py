#! /usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import re
import random
import math
import numpy as np
import cPickle as pkl

class CharDict(object):

    """
        Data Source is from Quan TangShi (Full Tang poems)
    """

    #start_regex = re.compile(r"\s+卷\d+_\d+\s+【(\w+)】\s+(\w+).*", re.UNICODE)
    start_regex = re.compile(r"\s+卷\d+_\d+\s+【(.*)】(.*)", re.UNICODE)
    split_regex = re.compile(u"[，|。]", re.UNICODE)
    
    def __init__(self, input_file):
        data_iter = self.full_tang_iter(input_file)
        self.data = {}
        for poem in data_iter:
            ret = CharDict.parse(poem[0], poem[1], poem[2])
            if ret:
                self.data[(ret[0], ret[1])] = ret[2]

    def full_tang_iter(self, input_file):
        with open(input_file, "r") as fin:
            data = []
            title = ""
            author = ""
            while True:
                line = fin.readline()
                #print line
                if not line: break
                d = self.start_regex.match(line)
                if d:
                    #print(d)
                    if data: yield (title, author, data)
                    title = d.group(1)
                    author = d.group(2)
                    data = []
                elif "-" not in line and line.strip():
                    data.append(line)

    @staticmethod
    def parse(title, author, data):
        """
            This method plays as a filter,
            Firstly we need 5-characters 
        """
        if len(data) > 4:
            return None

        #print(data[0])
        #if len(data[0].decode('utf-8')) != 12:
        #    return None
        data = map(lambda x : x.strip(), data)
        sent_words = CharDict.split_regex.split("".join(data).decode('utf-8'))
        sent_words = filter(lambda x: x, sent_words)

        data = "\n".join(sent_words)

        if len(data) == 63 and  u"（" not in data: 
            return title, author, data[:31].encode('utf-8')

        if len(data) != 31 or u"（" in data:
            return None

        #print(data)

        return title, author, data.encode('utf-8')

    def dump(self, data_dist):
        print("Total data length: %d" %len(self.data))
        with open(data_dist, "wb") as fout:
            pkl.dump(self.data, fout)


class RhymeDict(object):
    """
        Data Source is from PingShuiYun (PingShui Rhymes)
    """
    def __init__(self, data_src):
        with open(data_src, "r") as fin:
             self.rdict = pkl.load(fin)
        # Add separation
        self.rdict["\n"] = {None, None}
        # Notice: Include unknown chars
        self.data_len = len(self.rdict) + 1
        #print("Character Dictionary Length: %d" %self.data_len)
        print("Character Dictionary Length: %d (including separating enter)" %self.data_len)
        self.mapping = dict(zip(self.rdict.keys(), range(1, self.data_len)))
        self.inv_mapping = dict(zip(range(1, self.data_len), self.rdict.keys()))

    def char_to_id(self, char):
        if char in self.rdict:
            return self.mapping[char]
        else:
            return 0

    def id_to_char(self, idx):
        if idx in self.inv_mapping:
            return self.inv_mapping[idx]
        else:
            return "<UNK>"

    def get_sent(self, sent):
        return "".join([self.id_to_char(x) for x in sent]).decode('utf-8')

    def get_char_attr(self, char):
        if char in self.rdict:
            return self.rdict[char]
        else:
            return None

    def is_ze_by_idx(self, idx):
        return self.get_char_attr(self.id_to_char(idx))[0] == "入声"

    def is_ping_by_idx(self, idx):
        return not self.is_ze_by_idx(idx)

    def compare_rhyme_by_idx(self, idx1, idx2):
        return self.get_char_attr(self.id_to_char(idx1))[1] == \
               self.get_char_attr(self.id_to_char(idx2))[1]

    def get_char_attr_by_idx(self, idx):
        return self.get_char_attr(self.id_to_char(idx))

    def __getitem__(self, char):
        return self.char_to_id(char)

class PoemLoader(object):
    """
        Notice all the poem is string!
        string --> unicode --> string
              decode      encode
        use string to index, and unicode to iterate!
    """

    def __init__(self, data_src, batch_size = 100):
        with open(data_src, "rb") as fin:
            self.data = pkl.load(fin)
        self.batch_size = batch_size

    def data_to_matrix(self, word_dict):
        
        self.mat = []
        for sent in self.data.values():
            #print(sent)
            self.mat.append([ word_dict[idx.encode('utf-8')] for idx in sent.decode('utf-8')])
        self.mat = np.array(self.mat)
        self.maxlen = self.mat.shape[1]

    def gen_data(self):
        idx = range(len(self.data))
        random.shuffle(idx)

        def iterator():
            maxbatch = int(math.ceil(len(self.data)/float(self.batch_size)))
            for i in range(maxbatch):
                start = i*self.batch_size
                end = min(len(self.data), (i+1)*self.batch_size)
                yield self.mat[idx[start:end], :]

        return iterator()

if __name__ == "__main__":
    rdict = RhymeDict("./rhyme.pkl")
    pl = PoemLoader("./poem.pkl")
    pl.data_to_matrix(rdict)
    print(rdict.data_len)
    print(pl.mat[3, :])
    print(rdict.get_sent(pl.mat[3, :]))
    print(rdict.is_ping_by_idx(rdict["春"]))
    print(rdict.compare_rhyme_by_idx(rdict["闲"], rdict["寒"]))
    #cdict = CharDict("./full_tang_poetry.txt")
    #cdict.dump("./poem.pkl")
