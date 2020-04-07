#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:02:11 2019

@author: tom
"""

from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import torch
sen='which province did grey. jk 34.5 hjk'
nlp = StanfordCoreNLP('/home/tom/stanford-nlp-python/libs')
print(nlp.pos_tag(sen))
print(nlp.ner(sen))
def constructbatchforextra(examples):
    cs=[]
    ps=[]
    ns=[]
    ts=[]
    tins=[]
    f1s=[]
    for example in examples:
        c,p,n,t,tin,f1= example.contains,example.pos,example.ner,example.types,example.tins,example.F1
        cs.append(c)
        ps.append(p)
        ns.append(n)
        ts.append(t)
        tins.append(tin)
        f1s.append(f1)
    maxlen=max([adj.shape[0] for adj in cs])+1
    maxlen1=max([adj.shape[0] for adj in ts])
    batchsize=len(examples)
    css=np.zeros([batchsize,maxlen])
    pss=np.zeros([batchsize,maxlen])
    nss=np.zeros([batchsize,maxlen])
    tss=np.zeros([batchsize,maxlen1])
    tinss=np.zeros([batchsize,maxlen1])
    f1ss=np.zeros([batchsize,maxlen-1,maxlen1])
    for i in range(batchsize):
        templen=cs[i].shape[0]
        css[i][:templen]=cs[i]
        pss[i][:templen]=ps[i]
        nss[i][:templen]=ns[i]
        templen=ts[i].shape[0]
        tss[i][:templen]=ts[i]
        tinss[i][:templen]=tins[i]
        f1ss[i][:cs[i].shape[0],:templen]=f1s[i]
    def trans(a):
        return torch.Tensor(a).cuda().float()
    def transs(a):
        return torch.Tensor(a).cuda().long()
    return trans(css),transs(pss),transs(nss),transs(tss),transs(tinss),trans(f1ss)
def sentencetoextra_message(sentence,table,tabletypes,vocab1,vocab2,vocab3,train=True):
    re1=nlp.pos_tag(' '.join(sentence))
    re1=[item[1] for item in re1]
    re2=nlp.ner(' '.join(sentence))
    re2=[item[1] for item in re2]
    contains=np.zeros(len(sentence))
    pos=np.zeros(len(sentence))
    ner=np.zeros(len(sentence))
    tabletype=np.zeros(len(table))
    tablein=np.zeros(len(table))
    F1=np.zeros([len(sentence),len(table)])
    if(len(re1)==len(sentence)):
     for idx,item in enumerate(re1):
         if item in vocab1:
            pos[idx]=vocab1[item]
         else:
             if train:
              vocab1[item]=len(vocab1)
             pos[idx]=vocab1.get(item,1)
    if(len(re2)==len(sentence)):
     for idx,item in enumerate(re2):
         if item in vocab2:
            ner[idx]=vocab2[item]
         else:
             if train:
              vocab2[item]=len(vocab2)
             ner[idx]=vocab2.get(item,1)
    for idx,item in enumerate(sentence):
        if item in [te  for tem in table for te in tem]:
            contains[idx]=1
    for idx,item in enumerate(tabletypes):
         if item in vocab3:
            tabletype[idx]=vocab3[item]
         else:
             if train:
              vocab3[item]=len(vocab3)
             tabletype[idx]=vocab3.get(item,1)
    for idx,items in enumerate(table):
          if ' '.join(items) in ' '.join(sentence):
              tablein[idx]=1
          elif any([item in sentence for item in items]):
              tablein[idx]=2
          else:pass
    def isin(tem1,tem2):
        for tem in tem1:
            if tem in tem2 or tem2 in tem:
                return True
        return False
    for idx,item1 in enumerate(table):
        for idy,item2 in enumerate(sentence):
            if(isin(item1,item2)):
                F1[idy,idx]=1
#    print(re1)
#    print(re2)
#    print(table)
#    print(tabletypes)
#    print(tablein)
#    print(F1)
    return contains,pos,ner,tabletype,tablein,F1   
def sentencetoadj(sentence,edgevocab=None):
    results=nlp.dependency_parse(' '.join(sentence))
#    print(results)
    mainnode=results[0][2]-1
    adj=np.zeros([len(sentence),len(sentence)])
    edge=np.zeros([len(sentence),len(sentence)])
    maxlen=max([item[1] for item in results]+[item[2] for item in results])
    if maxlen!=len(sentence):
#        print(sentence)
#        print(results)
#        a=input('gh')
#        print(11)
        return 0,adj,edge,edgevocab,1
    if edgevocab is None:
        edgevocab={'null':0}
    for item in results[1:]:
#        try:
        adj[item[1]-1][item[2]-1]=1
#        except:
#            print(sentence)
#            print(results)
#            a=input('gh')
        if(item[0] not in edgevocab):
            edgevocab[item[0]]=len(edgevocab)
        edge[item[1]-1][item[2]-1]=edgevocab[item[0]]
#    print(adj.shape)
    return mainnode,adj,edge,edgevocab,0
def constructbatch(examples):
    mainnodes=[]
    adjs=[]
    edges=[]
    for sen in examples:
        mainnode,adj,edge=sen.mainnode,sen.adj,sen.edge
        mainnodes.append(mainnode)
        adjs.append(adj)
        edges.append(edge)
    maxlen=max([adj.shape[0] for adj in adjs])
    batchsize=len(examples)
    adj=np.zeros([batchsize,maxlen,maxlen])
    edge=np.zeros([batchsize,maxlen,maxlen])
    mask=np.zeros([batchsize,maxlen])
    for i in range(batchsize):
        templen=adjs[i].shape[0]
        adj[i][:templen,:templen]=adjs[i]
        edge[i][:templen,:templen]=edges[i]
        mask[i][:templen]=1
    def trans(a):
        return torch.Tensor(a).cuda().float()
    def transs(a):
        return torch.Tensor(a).cuda().long()
    return torch.Tensor(np.array(mainnodes)).cuda().long(),trans(adj),trans(adj.transpose([0,2,1])),transs(edge),transs(edge.transpose([0,2,1])),trans(mask)
#print(sentencetoadj(sen))