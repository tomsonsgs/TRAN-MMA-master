# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from six.moves import xrange
class myGCN(torch.nn.Module):
    def __init__(self, input_size,edge_size,vocab_size):#hid1,hid2
        super(myGCN, self).__init__()
        self.K=3
        self.emb=torch.nn.Embedding(vocab_size,edge_size)
        self.linear1=torch.nn.Linear(input_size+edge_size,edge_size,bias=False)
        self.linear2=torch.nn.Linear(input_size+edge_size,input_size)
        self.linear3=torch.nn.Linear(input_size+edge_size,edge_size,bias=False)
        self.linear4=torch.nn.Linear(input_size+edge_size,input_size)
        self.linear5=torch.nn.Linear(input_size*2,input_size)
        self.linear6=torch.nn.Linear(input_size*2,input_size)
    def forward(self, inputs,mains,adj1,adj2,edge1,edge2,mask=None):#b,s1,hid1;b,s2,hid2;b,s1;b,s2;
        length=inputs.size(1)
        innumber=torch.sum(adj1>0,-1).float().unsqueeze(-1)+0.0000000001#b,s,1
        outnumber=torch.sum(adj2>0,-1).float().unsqueeze(-1)+0.0000000001
        edge1=self.emb(edge1)#b,s,s,h
        edge2=self.emb(edge2)
        adj1=adj1.unsqueeze(-1)#b,s,s,1
        adj2=adj2.unsqueeze(-1)
        hid1=inputs
        hid2=inputs
        for i in range(self.K):#a,b x b,c ==sum(a,b,1 *1,b,c,-2)
            hid1e=hid1.unsqueeze(1).repeat(1,length,1,1)
            hid1e=torch.cat([hid1e,edge1],-1)#b,s,s,2h
            hid2e=hid2.unsqueeze(1).repeat(1,length,1,1)
            hid2e=torch.cat([hid2e,edge2],-1)#b,s,s,2h            
            hid1e=torch.tanh(self.linear1(torch.sum(adj1*hid1e,-2)/innumber))
            hid1=torch.tanh(self.linear2(torch.cat([hid1e,hid1],-1)))
            hid2e=torch.tanh(self.linear3(torch.sum(adj2*hid2e,-2)/outnumber))
            hid2=torch.tanh(self.linear4(torch.cat([hid2e,hid2],-1)))
        fused=torch.tanh(self.linear5(torch.cat([hid1,hid2],-1)))  #b,s,h  
        return fused,fused[torch.arange(0,inputs.size(0)),mains]
#        return torch.tanh(self.linear6(torch.cat([fused,inputs],-1))),fused[torch.arange(0,inputs.size(0)),mains]#b,s1,hid1,gao ji suo ying
class cross_att(torch.nn.Module):
    def __init__(self, input_size,hid_size):#hid1,hid2
        super(cross_att, self).__init__()
        self.linear1=torch.nn.Linear(input_size,hid_size,bias=False)
        self.linear2=torch.nn.Linear(input_size+hid_size,input_size)
    def forward(self, in1,in2,mask2=None):#b,s1,hid1;b,s2,hid2;b,s1;b,s2;
        new_in1=self.linear1(in1)#b,s1,hid2
        cross_att=torch.matmul(new_in1,in2.transpose(1,2))#b,s1,s2
        
        
        if type(mask2)!=type(None):
         mask2=mask2.unsqueeze(-2).repeat(1,in1.size(1),1)#b,s1,s2   
         cross_att.data.masked_fill_(mask2,-float('inf'))
        att=torch.softmax(cross_att,-1)#b,s1,s2
        fused=torch.matmul(att,in2)#b,s1,hid2
        fused=torch.cat([in1,fused],-1)
        fused=torch.tanh(self.linear2(fused))
        return fused#b,s1,hid1
class cross_att5(torch.nn.Module):
    def __init__(self, input_size,hid_size):#hid1,hid2
        super(cross_att5, self).__init__()
        self.linear1=torch.nn.Linear(4*hid_size,hid_size)
        self.linear2=torch.nn.Linear(hid_size,1)
    def forward(self, in1,in2,probs,mask2=None):#b,s1,hid1;b,s2,hid2;b,s1;b,s2;
#        new_in1=self.linear1(in1)#b,s1,hid2
        in1=in1.unsqueeze(1).repeat(1,in2.size(1),1)
        cross_att=self.linear2(torch.tanh(self.linear1(torch.cat([in1,in2,torch.abs(in1-in2),in1*in2],-1)))).squeeze(-1)#b,s1,s2
        
        
        if type(mask2)!=type(None):
#         mask2=mask2.unsqueeze(-2).repeat(1,in1.size(1),1)#b,s1,s2   
         cross_att.data.masked_fill_(mask2,-float('inf'))
        att=torch.softmax(cross_att,-1)#b,s1,s2
        fused=torch.matmul(att.unsqueeze(1),in2)#b,s1,hid2
#        fused=torch.cat([in1,fused],-1)
#        fused=torch.tanh(self.linear2(fused))
        return fused.squeeze(1),att#b,s1,hid1
class cross_atts(torch.nn.Module):
    def __init__(self, input_size,hid_size):#hid1,hid2
        super(cross_atts, self).__init__()
#        self.linear1=torch.nn.Linear(input_size,hid_size,bias=False)
    def forward(self, in1,in2,mask2=None):#b,s1,hid1;b,s2,s3,hid2;b,s1;b,s2;
#        new_in1=self.linear1(in1)#b,s1,hid2
        in1=in1/(torch.sum(in1*in1,-1).unsqueeze(-1)**0.5+0.0000000000001)
        in2=in2/(torch.sum(in2*in2,-1).unsqueeze(-1)**0.5+0.0000000000001)
        in1=in1.unsqueeze(2).unsqueeze(2)
        in2=in2.unsqueeze(1)
        cross_att=torch.sum(in1*in2,-1)#b,s1,s2,s3
        cross_att,_=torch.max(cross_att,-1)#b,s1,s2,0~2
        cross_att=cross_att+1
        if type(mask2)!=type(None):
         mask2=mask2.unsqueeze(-2).repeat(1,in1.size(1),1)#b,s1,s2   
         cross_att.data.masked_fill_(mask2,-float(0))
        att=(cross_att>1.999).float()
#        att=cross_att/torch.sum(cross_att,-1).unsqueeze(-1)#b,s1,s2
        return att     
class cross_attss(torch.nn.Module):
    def __init__(self, input_size,hid_size):#hid1,hid2
        super(cross_attss, self).__init__()
        self.linear1=torch.nn.Linear(input_size,hid_size,bias=False)
#        self.linear2=torch.nn.Linear(input_size+hid_size,input_size)
    def forward(self, in1,in2,mask2=None):#b,s1,hid1;b,s2,hid2;b,s1;b,s2;
        new_in1=self.linear1(in1)#b,s1,hid2
        cross_att=torch.matmul(new_in1,in2.transpose(1,2))#b,s1,s2
        
        
        if type(mask2)!=type(None):
         mask2=mask2.unsqueeze(-2).repeat(1,in1.size(1),1)#b,s1,s2   
         cross_att.data.masked_fill_(mask2,-float('inf'))
        att=torch.softmax(cross_att,-1)#b,s1,s2
#        fused=torch.matmul(att,in2)#b,s1,hid2
#        fused=torch.cat([in1,fused],-1)
#        fused=torch.tanh(self.linear2(fused))
        return att#b,s1,hid1    
class multi_rnn(torch.nn.Module):
    def __init__(self, input_size, hid_size, num):#hid1,hid2
        super(multi_rnn, self).__init__()
        self.num=num
        self.hid_size=hid_size
        self.drop=torch.nn.Dropout(0.3)
        rnns=[torch.nn.LSTMCell(input_size, hid_size)]+[torch.nn.LSTMCell(hid_size, hid_size) for _ in range(num-1)]
        self.rnns=torch.nn.ModuleList(rnns)
    def get_zeros(self,batch_size):
        return torch.cat([torch.zeros(batch_size,self.hid_size).cuda()]*self.num,-1),torch.cat([torch.zeros(batch_size,self.hid_size).cuda()]*self.num,-1)
    def forward(self, inputs,h,c):#b,s1,hid1;b,s2,hid2;b,s1;b,s2;
        mid=inputs
        output=[]
        cmem=[]
#        print(h.size())
        h=torch.split(h,self.hid_size,-1)
#        print(h)
        c=torch.split(c,self.hid_size,-1)
        for i in range(self.num):
           mid,ctemp=self.rnns[i](mid,(h[i],c[i])) 
#           if(i<self.num-1):
#            mid=self.drop(mid)
           output.append(mid)
           cmem.append(ctemp)
        return torch.cat(output,-1),torch.cat(cmem,-1)#h,c        
class multi_rnn_withcontext(torch.nn.Module):
    def __init__(self, input_size, hid_size, num):#h id1,hid2
        super(multi_rnn_withcontext, self).__init__()
        self.num=num
        self.hid_size=hid_size
        self.drop=torch.nn.Dropout(0.3)
        rnns=[torch.nn.LSTMCell(input_size, hid_size)]+[torch.nn.LSTMCell(hid_size, hid_size) for _ in range(num-1)]
        self.rnns=torch.nn.ModuleList(rnns)
        self.linear = nn.Linear(hid_size*2, hid_size, bias=False)
#        self.linear1 = nn.Linear(hid_size*2, hid_size, bias=False)
#        self.gru=nn.GRUCell(hid_size, hid_size)
        self.cross=cross_att5(hid_size, hid_size)
    def get_zeros(self,batch_size):
        return torch.cat([torch.zeros(batch_size,self.hid_size).cuda()]*self.num,-1),torch.cat([torch.zeros(batch_size,self.hid_size).cuda()]*self.num,-1)
    def forward(self, inputs,h,c,global_state,src_encodings, src_encodings_att_linear,probs,masks=None):#b,s1,hid1;b,s2,hid2;b,s1;b,s2;
        mid=inputs
        output=[]
        cmem=[]
        atts=[]
#        print(h.size())
        h=torch.split(h,self.hid_size,-1)
#        print(h)
        c=torch.split(c,self.hid_size,-1)
#        tmp=[]
        for i in range(self.num):
           mid,ctemp=self.rnns[i](mid,(h[i],c[i])) 
#           mids=torch.tanh(self.linear1(torch.cat([mid,global_state],-1)))
           
#           if(i<self.num-1):
#            mid=self.drop(mid)
           output.append(mid)
#           ctx_t, alpha_t = dot_prod_attention(mid,
#                                                     src_encodings, src_encodings_att_linear,
#                                                     mask=masks)
           ctx_t, alpha_t = self.cross(mid,src_encodings,probs, mask2=masks)
           atts.append(alpha_t)
#           tmp.append(ctx_t)
           if(i<self.num-1):
               pass
#            mid = F.tanh(self.linear(torch.cat([mid, ctx_t], 1))) 
#            mid=self.gru(ctx_t,mid)
#            mid=self.drop(mid)
           cmem.append(ctemp)
#        global_state=self.gru(torch.cat(tmp,-1),global_state)
        atts=torch.stack(atts,1).cpu().data.numpy()
        return torch.cat(output,-1),torch.cat(cmem,-1),ctx_t,alpha_t,global_state,atts#h,c          
class AttGRU(torch.nn.Module):
    def __init__(self, input_size,hid_size):
        super(AttGRU, self).__init__()
        self.w_r=torch.nn.Linear(input_size,hid_size)
        self.w_c=torch.nn.Linear(input_size,hid_size)
        self.u_r=torch.nn.Linear(hid_size,hid_size)
        self.u_c=torch.nn.Linear(hid_size,hid_size)

    def forward(self, inputs,state, att):
        r=torch.sigmoid(self.w_r(inputs)+self.u_r(state))
        c=torch.tanh(self.w_c(inputs)+self.u_c(state*r))
        new_h=att*c+(1-att)*state
        return new_h
class ParentGRU(torch.nn.Module):
    def __init__(self, input_size,hid_size):
        super(ParentGRU, self).__init__()
        self.w_a=torch.nn.Linear(input_size,hid_size)        
        self.w_r=torch.nn.Linear(input_size,hid_size)
        self.w_r1=torch.nn.Linear(input_size,hid_size)
        self.w_c=torch.nn.Linear(input_size,hid_size)
        self.u_a=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.u_r=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.u_r1=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.u_c=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.p_a=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.p_r=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.p_r1=torch.nn.Linear(hid_size,hid_size,bias=False)
        self.p_c=torch.nn.Linear(hid_size,hid_size,bias=False)
    def forward(self, inputs,state, parent_state):
        att=torch.sigmoid(self.w_a(inputs)+self.u_a(state)+self.p_a(parent_state))
        r=torch.sigmoid(self.w_r(inputs)+self.u_r(state)+self.p_r(parent_state))
        r1=torch.sigmoid(self.w_r1(inputs)+self.u_r1(state)+self.p_r1(parent_state))
        c=torch.tanh(self.w_c(inputs)+self.u_c(state*r)+self.p_c(parent_state*r1))
        new_h=att*c+(1-att)*state
        return new_h
class DMN(torch.nn.Module):
    def __init__(self, input_size,hid_size,num=3):#hid;src
        super(DMN, self).__init__()
        self.num=num
#        self.attgru=AttGRU(hid_size,hid_size)
        self.gru=torch.nn.GRUCell(hid_size,input_size)
        self.feed=torch.nn.Sequential(torch.nn.Linear(2*input_size+hid_size+1,hid_size),torch.nn.Tanh(),torch.nn.Linear(hid_size,1))
    def forward(self, hid_vec,src_enc, src_mask,last_used):
        #b,hid1;b,s,hid2;b,s;b,s
        h_0= hid_vec
        hid_vec=h_0.unsqueeze(1).repeat(1,src_enc.size(1),1)#b,s,hid1
        for j in range(self.num):
            hid_new=h_0.unsqueeze(1).repeat(1,src_enc.size(1),1)#b,s,hid1
            new_vec=torch.cat([hid_vec,hid_new,src_enc,last_used.unsqueeze(-1)],-1)
            tmp=self.feed(new_vec).squeeze(-1)
            if type(src_mask)!=type(None):
             tmp.data.masked_fill_(src_mask,-float('inf'))
            att=torch.softmax(tmp,-1)#b,s
            h0=torch.matmul(att.unsqueeze(-2),src_enc).squeeze(1)#b,1,hid_size
#            h0=torch.zeros(src_enc.size(0),src_enc.size(2)).float().cuda()#b,hid2
#            
#            for i in range(src_enc.size(1)):
#                h0=self.attgru(src_enc[:,i,:],h0,att[:,i].unsqueeze(-1))
            h_0=self.gru(h0,h_0)
#        return att,h_0, last_used+att
        return att,h_0, last_used

def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


def length_array_to_mask_tensor(length_array, cuda=False, valid_entry_has_mask_one=False):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.zeros((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        if valid_entry_has_mask_one:
            mask[i][:seq_len] = 1
        else:
            mask[i][seq_len:] = 1

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in xrange(batch_size)])

    return sents_t


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]
def word2ids(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab._convert_token_to_id(w) for w in s] for s in sents]
    else:
        return [vocab._convert_token_to_id(w) for w in sents]

def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2word[w] for w in s] for s in sents]
    else:
        return [vocab.id2word[w] for w in sents]


def to_input_variable(sequences, vocab, cuda=False, training=True, append_boundary_sym=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    word_ids = word2id(sequences, vocab)
    sents_t = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=(not training), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var
def to_input_variable1(sequences, vocab, bert_model,cuda=False, training=True, append_boundary_sym=True):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        sequences = [['[CLS]'] + seq + ['[SEP]'] for seq in sequences]

        
    word_ids = word2id(sequences, vocab)
    maxlen=max([len(item) for item in word_ids])
    newword_ids=[]
    mask_ids=[]
    for item in word_ids:
       newword_ids.append(item+[0]*(maxlen-len(item))) 
       mask_ids.append([1]*len(item)+[0]*(maxlen-len(item)))
    sents_var = Variable(torch.LongTensor(newword_ids), volatile=(not training), requires_grad=False)
    sents_var_mask = Variable(torch.FloatTensor(mask_ids), volatile=(not training), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
        sents_var_mask = sents_var_mask.cuda()
#    sents_t = input_transpose(word_ids, vocab['<pad>'])
    outputs=bert_model(sents_var,attention_mask=sents_var_mask)
    outputs=outputs[0]
#    print(outputs.size())
#    a=input('haha')

    return outputs[:,1:,:]
def to_input_variable2(sequences,tables ,tokenizer, bert_model,cuda=False, training=True, append_boundary_sym=True):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    beforquestions=[]
    tmptokens=[]
    for seq in sequences:
      beforquestion=[]
      tmptoken=[]
      for se in seq:
          tmplength=len(tmptoken)
          tmp=tokenizer._tokenize(se)
#          tmp=[se]
          tmptoken.extend(tmp)
          beforquestion.append([tmplength,len(tmptoken)-1])
      tmptokens.append(tmptoken)
      beforquestions.append(beforquestion)
    maxquestionlen=max([len(item) for item in sequences])
    src_enc=torch.zeros(len(sequences),maxquestionlen+1,768).cuda().float()
    tmpheads=[]
    headlens=[]
    for table in tables:
        tmphead=[]
        headlen=[]
        content=[item.tokens for item in table.header]
        for ite in content:
            tmph=[]
            for it in ite:
                tmph.extend(tokenizer._tokenize(it))
#                tmph.extend([it])
            headlen.append(len(tmph))
            tmphead.append(tmph)
        tmpheads.append(tmphead)
        headlens.append(headlen)
    maxheadlen=max([len(item) for item in tmpheads])
    maxtokenlen=max([ite for item in headlens for ite in item])
    for item in headlens:
        item.extend([1]*(maxheadlen-len(item)))
    src_head=torch.zeros(len(sequences),maxheadlen,maxtokenlen,768).cuda().float()
    newsample=[]
    befortables=[]
    typeids=[]
    for i in range(len(sequences)):
        befortable=[]
        tmp=[]
        typeid=[]
        tmp.append('[CLS]')
        typeid.append(0)
        tmp.extend(tmptokens[i])
        typeid.extend([0]*len(tmptokens[i]))
        tmp.append('[SEP]')
        typeid.append(0)
        for j in range(len(tmpheads[i])):
            tmplen=len(tmp)
            tmp.extend(tmpheads[i][j])
            typeid.extend([1]*len(tmpheads[i][j]))
            tmp.append('[SEP]')
            if(j<len(tmpheads[i])-1):typeid.append(0)
            else:typeid.append(1)
            befortable.append([tmplen,len(tmp)-2])
        newsample.append(tmp)
        befortables.append(befortable)
        typeids.append(typeid)
#    if append_boundary_sym:
#        sequences = [['[CLS]'] + seq + ['[SEP]'] for seq in sequences]
#    print(sequences[0])
#    print(newsample) 
#    print(beforquestions[0])
#    print(befortables[0])
#    print(tmpheads[0])
#    a=input('hahah')
    word_ids = word2ids(newsample, tokenizer)
    maxlen=max([len(item) for item in word_ids])
    newword_ids=[]
    mask_ids=[]
    for u in typeids:
        u.extend([1]*(maxlen-len(u)))
#    print('dfdff'+str(maxheadlen))
#    print('dfddff'+str(maxtokenlen))
#    print('fdf'+str(maxquestionlen))
#    print('gh'+str(maxlen))
    for item in word_ids:
       newword_ids.append(item+[0]*(maxlen-len(item))) 
       mask_ids.append([1]*len(item)+[0]*(maxlen-len(item)))
       
    sents_var = Variable(torch.LongTensor(newword_ids), volatile=(not training), requires_grad=False)
    sents_var_mask = Variable(torch.FloatTensor(mask_ids), volatile=(not training), requires_grad=False)
    sents_var_mask1 = Variable(torch.LongTensor(typeids), volatile=(not training), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
        sents_var_mask = sents_var_mask.cuda()
        sents_var_mask1=sents_var_mask1.cuda()
#    sents_t = input_transpose(word_ids, vocab['<pad>'])
    outputs=bert_model(sents_var,token_type_ids=sents_var_mask1,attention_mask=sents_var_mask)
    outputs=outputs[0]
    real=outputs[:,1:,:]
    for idb,item in enumerate(sequences):
        for idx in range(len(item)):
            src_enc[idb][idx,:]=torch.sum(real[idb][beforquestions[idb][idx][0]:beforquestions[idb][idx][1]+1,:],0)
    for idb,item in enumerate(tmpheads):
        for idx,ite in enumerate(item):
            src_head[idb][idx][:len(ite),:]=outputs[idb][befortables[idb][idx][0]:befortables[idb][idx][1]+1,:]
#    print(outputs.size())
#    a=input('haha')
    
    return src_enc,src_head,headlens
def variable_constr(x, v, cuda=False):
    return Variable(torch.cuda.x(v)) if cuda else Variable(torch.x(v))


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in xrange(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
       source: https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def uniform_init(lower, upper, params):
    for p in params:
        p.data.uniform_(lower, upper)


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal(p.data)


def identity(x):
    return x


class LabelSmoothing(nn.Module):
    """Implement label smoothing.

    Reference: the annotated transformer
    """

    def __init__(self, smoothing, tgt_vocab_size, ignore_indices=None):
        if ignore_indices is None: ignore_indices = []

        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        smoothing_value = smoothing / float(tgt_vocab_size - 1 - len(ignore_indices))
        one_hot = torch.zeros((tgt_vocab_size,)).fill_(smoothing_value)
        for idx in ignore_indices:
            one_hot[idx] = 0.

        self.confidence = 1.0 - smoothing
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, model_prob, target):
        # (batch_size, *, tgt_vocab_size)
        dim = list(model_prob.size())[:-1] + [1]
        true_dist = Variable(self.one_hot, requires_grad=False).repeat(*dim)
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        # true_dist = model_prob.data.clone()
        # true_dist.fill_(self.smoothing / (model_prob.size(1) - 1))  # FIXME: no label smoothing for <pad> <s> and </s>
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return self.criterion(model_prob, true_dist).sum(dim=-1)


class FeedForward(nn.Module):
    """Feed forward neural network adapted from AllenNLP"""

    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super(FeedForward, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore

        self.activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        self.linear_layers = nn.ModuleList(linear_layers)
        dropout_layers = [nn.Dropout(p=value) for value in dropout]
        self.dropout = nn.ModuleList(dropout_layers)
        self.output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def forward(self, x):
        output = x
        for layer, activation, dropout in zip(self.linear_layers, self.activations, self.dropout):
            output = dropout(activation(layer(output)))
        return output
