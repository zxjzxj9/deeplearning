#! /usr/bin/env python

import os
import sys
from collections import Counter, Iterable

"""
    Create a vocabulary for the give corpus
"""

class VocabDict(object):
    """
        Standard library for searching word, three default tokens:
        <go>    0 
        <eos>   1
        <unk>   2
    """
    def __init__(self, word_list, min_rank = 0, max_rank = 10000, freq_cut_off = None):
        self.voc = Counter(word_list).most_common(max_rank)
        min_rank_del = self.voc.most_common(min_rank)

        for k, v in min_rank_del:
            self.voc.pop(k)

        if freq_cut_off != None:
            for k, v in self.voc.items():
                if v  < freq_cut_off: self.voc.pop(k)

        self.voc_dict = dict(zip(self.voc.keys(), range(3, 3+len(self.voc))))
        self.voc_dict["<go>"] = 0
        self.voc_dict["<eos>"] = 1
        self.voc_dict["<unk>"] = 2
        self.inv_voc_dict = {v:k for k,v in self.voc_dict.items()}

    def __getitem__(self, idx):
        if idx in self.voc_dict: return self.voc_dict[idx]
        return 2

    def conv(self, src):
        if isinstance(src, Iterable):
            return [self[idx] for idx in src]
        else:
            return self[idx]

    def _inv_get(self, idx):
        if idx in self.inv_voc_dict:
            return self.inv_voc_dict[idx]
        else:
            raise IndexError("Invalid Index: {}, no such index".format(idx))

    def inv_conv(self, src):
        if isinstance(src, Iterable):
            return [self._inv_get[idx] for idx in src]
        else:
            return self._inv_get[src]
        
