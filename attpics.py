#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 07:41:58 2019

@author: tom
"""
# coding=utf-8
from __future__ import print_function

import argparse
from itertools import chain
import trainwiki
import six.moves.cPickle as pickle
from six.moves import xrange as range
from six.moves import input
import traceback

import numpy as np
import time
import os
import sys

#import torch
#from torch.autograd import Variable
#
#import evaluation
#from asdl import *
#from asdl.asdl import ASDLGrammar
#from common.registerable import Registrable
#from components.dataset import Dataset, Example
#from common.utils import update_args, init_arg_parser
#from datasets import *
#from datasets.wikisql.dataset import get_action_infos
#from model import nn_utils, utils
#import tqdm
#from model.parser import Parser
#from model.utils import GloveHelper, get_parser_class
#import copy
#from pytorch_transformers import *
#if six.PY3:
#    # import additional packages for wikisql dataset (works only under Python 3)
#    from model.wikisql.dataset import WikiSqlExample, WikiSqlTable, TableColumn
#    from model.wikisql.parser import WikiSqlParser
#    from datasets.wikisql.dataset import Query, DBEngine
#def init_config():
#    args = arg_parser.parse_args()
#    trainwiki.update(args)
#    trainwiki.updatetest(args)
#    # seed the RNG
#    torch.manual_seed(args.seed)
#    if args.cuda:
#        torch.cuda.manual_seed(args.seed)
#    np.random.seed(int(args.seed * 13 / 7))
#
#    return args    
#path1='decodes/wikisql/'+'wikitest.decode'
#path2='decodes/wikisql/'+'wikitest.decode1'
#arg_parser = init_arg_parser()
#args = init_config()
#    
#print(args, file=sys.stderr)
#if args.mode == 'train':
#        pass
#elif args.mode == 'test':
#    test_set = (Dataset.from_bin_file(args.test_file)).examples
#    decode_results = pickle.load(open(path1, 'rb'))
#    decode_results1 = pickle.load(open(path2, 'rb'))
#   
#def evaluate_dataset(examples, decode_results,decode_results1, fast_mode=True):
#        global cachepredict
#        global cachetrue
#        correct_array = []
#        oracle_array = []
#        cachepredict=[]
#        cachetrue=[]
#        allstats=[]
#        number=0
#
#        for example, hyp_list,atts,hyp_list1,atts1 in tqdm.tqdm(zip(examples, decode_results[0],decode_results[1],decode_results1[0],decode_results1[1])):
#            number+=1
#            filename='attpic/haha'+str(number)+'.txt'
#            
#            if fast_mode:
#                hyp_list = hyp_list[:1]
#                att=atts[:1]
#                hyp_list1 = hyp_list1[:1]
#                att1=atts1[:1]
##                ast=attss[:1]
#            if hyp_list:
#                if(hyp_list[0].tree.sort_removedup_self().to_string()==example.tgt_ast.sort_removedup_self().to_string()):
#                  if(hyp_list1[0].tree.sort_removedup_self().to_string()!=example.tgt_ast.sort_removedup_self().to_string()):
#                    file=open(filename,'w',encoding='utf8') 
#                    def logs(source):
#                        tmp=str(source)+'\n'
#                        if(type(source)==np.ndarray):
##                            print('gh')
#                            tmp=tmp.replace(' ',',')
#                        file.write(tmp)
##                    print(example.src_sent)
#                    logs(example.src_sent)
##                    tree=Tree.fromstring(str(nlp.parse(' '.join(example.src_sent))))
#                    
##                    print([item.name for item in example.table.header])
#                    logs([item.name for item in example.table.header])
##                    print(type(att[0][1][0]))
#                    logs(att[0][1][0])
#                    logs(hyp_list[0].actions)
#                    logs([a.action for a in example.tgt_actions])
#                    
#                    for action,at,ats in zip(hyp_list[0].actions,att[0][0],att[0][0]):
##                    if(show and number>3):
##                        if np.linalg.norm(at[-1]-at[0])>0.5:
#                             logs(example.src_sent) 
#                             logs(action)
#                             logs(at)
#                    logs('hahahah')
#                    logs(att1[0][1][0])
#                    logs(hyp_list1[0].actions)
#                    logs([a.action for a in example.tgt_actions])
#                    
#                    for action,at,ats in zip(hyp_list1[0].actions,att1[0][0],att1[0][0]):
##                    if(show and number>3):
##                        if np.linalg.norm(at[-1]-at[0])>0.5:
#                             logs(example.src_sent) 
#                             logs(action)
#                             logs(at)
##                             print(ats)
##                    a=input('jk')
##                    tree.draw() 
#                    file.close()
#evaluate_dataset(test_set,decode_results,decode_results1)
def plotandsave(x1,filename) :
    h=x1.shape[0]
    w=x1.shape[1]  
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
fig=plt.figure(figsize=(4,4),dpi=100)
ax=fig.add_subplot(131)
#filename='attpic/'+'hah'+str(1)+'.png'                 
#x1=None
#x2=None
y1=[[9.6998870e-04,9.1004940e-03,8.3634806e-01,4.7535527e-02,2.3504786e-02
,8.2541265e-02]
,[5.2753957e-09,2.0472803e-03,8.6948401e-01,2.0598317e-03,3.7566433e-03
,1.2265216e-01]
,[1.7321455e-04,3.6292754e-02,3.3654305e-01,1.4397605e-01,5.5177052e-02
,4.2783794e-01]
,[1.1034801e-14,4.9862310e-06,9.9231601e-01,6.5579261e-03,1.1198113e-03
,1.2694999e-06]
,[6.5195176e-11,3.1255414e-14,9.9998724e-01,1.2734134e-05,1.0426947e-10
,7.4663644e-09]
,[2.4697385e-08,6.7954903e-10,4.9978055e-02,9.5002151e-01,4.1699624e-07
,6.4978032e-12]
,[3.8797024e-04,5.5915923e-03,6.0967588e-01,3.6113146e-01,1.8571369e-02
,4.6416251e-03]
,[9.9455160e-01,9.2497544e-08,1.1339599e-16,5.4482529e-03,3.6507071e-08
,2.4443275e-14]
,[2.0638020e-06,1.6218249e-07,3.8913299e-06,9.9999321e-01,7.6687843e-07
,1.7298886e-14]
,[3.9401298e-06,4.9928213e-03,8.3514327e-01,1.5552324e-01,4.3367338e-03
,1.0438029e-07]
,[5.2026206e-01,5.5223227e-02,1.4369285e-01,2.7513480e-01,5.3710183e-03
,3.1601486e-04]]
y1=np.array(y1)
y1=y1.transpose([1,0])
y2=[[1.1594588e-12,2.5747770e-12,3.1203585e-11,5.4956217e-07,2.8658548e-01
,7.1341395e-01,5.4337381e-12,9.4988326e-17,6.7793228e-17,2.3685290e-15
,2.1779333e-15]]
y2=np.array(y2)
y1=y1*y2
y2=np.sum(y1,-1,keepdims=True)
ax.grid(True)
tp=ax.imshow(y1,cmap=plt.cm.get_cmap('gray').reversed(),vmin=0.0,vmax=1.0)
plt.xticks([])
plt.yticks([])
ax=fig.add_subplot(132)
ax.grid(True)
ax.imshow(y2,cmap=plt.cm.get_cmap('gray').reversed(),vmin=0.0,vmax=1.0)
plt.colorbar(tp,ax=ax,shrink=0.2)
plt.xticks([])
plt.yticks([])
#fig.tight_layout()
plt.show()
#plt.savefig('tmpp.png')
#plt.colorbar(shrink=0.93)
#y2=None
# 
#plotandsave(x1,filename)              