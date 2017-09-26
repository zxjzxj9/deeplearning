#! /usr/bin/env python
#-*- coding: utf-8 -*- 

import requests
import pickle as pkl
import os
from lxml.etree import HTML
import sys

#reload(sys)  
#sys.setdefaultencoding('utf8')

"""
    Crawl the rhyme dict from website http://sou-yun.com/QR.aspx
    Notice: character multi-pronounciation haven't beeen considered
"""

class RhymeCrawler(object):
    
    website = "http://sou-yun.com/QR.aspx"
    root = "http://sou-yun.com/"
    dt = {}

    def __init__(self):
        r = requests.get(self.website)
        #page = HTML(r.contents.encode(r.encoding))
        page = HTML(r.text)
        self.page_list = page.xpath("//*[@id='FullListPanel']/div/div/a/@href")

    def crawl(self):
        for p in self.page_list:
            r = requests.get(os.path.join(self.root, p))
            page = HTML(r.text)

            rn = page.xpath("//div[@class='char']/span[@class='rhymeName']/text()")[0].encode('utf-8')
            comment = page.xpath("//div[@class='char']/span[@class='comment']/text()")[0].encode('utf-8')
            
            #print(type(rn))
            #print(rn.decode('utf8'))
            #print(type(rn.decode('utf8')))

            #print(sys.getdefaultencoding())
            #print(type(rn))
            #sys.exit()

            print(u"Rhyme Name is %s, comment is %s".encode('utf-8') %(rn, comment))
            #sys.exit()

            for t in page.xpath("//div[@class='char']/a/text()"):
                self.dt[t.encode('utf-8')] = [comment, rn]

    def dump(self):
        """
            Dump the character's attribute as the following dict:

            {"Character" :  ["p|s|q|r", "rhyme"]}
            
            where psqr stands for four kinds of tones:

            Ping    --->   Ping Tone
            Shang   --->   Ping Tone
            Qu      --->   Ping Tone
            Ru      --->   Ze   Tone
        """
        #print(self.dt)
        with open("rhyme.pkl" ,"w") as rout:
            pkl.dump(self.dt, rout)

if __name__ == "__main__":
    rc = RhymeCrawler()
    rc.crawl()
    rc.dump()


