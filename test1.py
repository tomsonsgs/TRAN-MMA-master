# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:13:12 2019

@author: tang
"""

import numpy
import copy
#a=[[1,2],[3,4]]
#for b in a:
#    b[0]=9
#print(a)
#import argparse
#parser=argparse.ArgumentParser()
#parser.add_argument('-enc',type=str)
#opt=parser.parse_args()
#print(opt.enc)
#import torch
class a:
    def __init__(self,a):
      self.a=a
      self.b=[12,3]   
class b:
    def __init__(self,a):
      self.c=9
n=a(3)
b.c=n.b
n.b=[56,4]
print(b.c)
#print(isinstance(n,a))
#print(isinstance(n,b))
#b=a(8)
#c=copy.deepcopy(b)
##torch.save(b,open('bn.bin','wb'))
#a=torch.load('bn.bin','wb')
#class a(object):
#    def __init__(self):
#        self.g=0
#class b(a):
#    def __init__(self):
#        super(a, self).__init__()
#v=a()
#vb=b()
#print(type(v)==type(vb))
#print(type(vb) is a)
