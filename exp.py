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

import torch
from torch.autograd import Variable

import evaluation
from asdl import *
from asdl.asdl import ASDLGrammar
from common.registerable import Registrable
from components.dataset import Dataset, Example
from common.utils import update_args, init_arg_parser
from datasets import *
from datasets.wikisql.dataset import get_action_infos
from model import nn_utils, utils
import tqdm
from model.parser import Parser
from model.utils import GloveHelper, get_parser_class
import copy
from pytorch_transformers import *
if six.PY3:
    # import additional packages for wikisql dataset (works only under Python 3)
    from model.wikisql.dataset import WikiSqlExample, WikiSqlTable, TableColumn
    from model.wikisql.parser import WikiSqlParser
    from datasets.wikisql.dataset import Query, DBEngine


def init_config():
    args = arg_parser.parse_args()
    trainwiki.update(args)
    trainwiki.updatetest(args)
    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def train(args):
    """Maximum Likelihood Estimation"""
    tokenizer=BertTokenizer.from_pretrained('./pretrained_models/bert-base-uncased-vocab.txt')
    bertmodels=BertModel.from_pretrained('./pretrained_models/base-uncased/')
    print(len(tokenizer.vocab))
    
    # load in train/dev set
    tmpvocab={'null':0}
    tmpvocab1={'null':0,'unk':1}
    tmpvocab2={'null':0,'unk':1}
    tmpvocab3={'null':0,'unk':1}
    train_set = Dataset.from_bin_file(args.train_file)
    from dependency import sentencetoadj,sentencetoextra_message
    valid=0
#    for example in tqdm.tqdm(train_set.examples):
##        print(example.src_sent)
##        example.mainnode,example.adj,example.edge,_,isv=sentencetoadj(example.src_sent,tmpvocab)
#        example.contains,example.pos,example.ner,example.types,example.tins,example.F1=sentencetoextra_message(example.src_sent,[item.tokens for item in example.table.header],[item.type for item in example.table.header],tmpvocab1,tmpvocab2,tmpvocab3,True)
##        valid+=isv
##        print(example.src_sent)
###        print( example.contains,example.pos,example.ner,example.types)
##        a=input('gh')
#    print('bukey',valid)
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])
#    for example in tqdm.tqdm(dev_set.examples):
##        print(example.src_sent)
##        example.mainnode,example.adj,example.edge,_,_=sentencetoadj(example.src_sent,tmpvocab)
#        example.contains,example.pos,example.ner,example.types,example.tins,example.F1=sentencetoextra_message(example.src_sent,[item.tokens for item in example.table.header],[item.type for item in example.table.header],tmpvocab1,tmpvocab2,tmpvocab3,False)
    vocab = pickle.load(open(args.vocab, 'rb'))
    print(len(vocab.source))
    vocab.source.copyandmerge(tokenizer)
    print(len(vocab.source))
#    tokenizer.update(vocab.source.word2id)
#    print(len(tokenizer.vocab))
#    print(tokenizer.vocab['metodiev'])
#    bertmodels.resize_token_embeddings(len(vocab.source))
    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)

    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    model = parser_cls(args, vocab, transition_system,tmpvocab,tmpvocab1,tmpvocab2,tmpvocab3)

    model.train()
    model.tokenizer=tokenizer
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    if args.cuda: model.cuda()



    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    elif args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    # load pre-trained word embedding (optional)
    if args.glove_embed_path:
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path)
        glove_embedding.load_to(model.src_embed, vocab.source)
    print([name for name,_ in model.named_parameters()])
    model.bert_model=bertmodels
#    print([name for name,_ in model.named_parameters()])
    model.train()
    if args.cuda: model.cuda()
#    return 0
#    a=input('haha')
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
#    parameters=[p for name,p in model.named_parameters() if 'bert_model' not in name or 'embeddings' in name]
    parameters=[p for name,p in model.named_parameters() if 'bert_model' not in name]
    parameters1=[p for name,p in model.named_parameters() if 'bert_model' in name]
    optimizer = optimizer_cls(parameters, lr=args.lr)
    optimizer1 = optimizer_cls(parameters1, lr=0.00001)
    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)
    is_better = False
    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    report_loss1=0
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        if epoch>40:break
        epoch += 1
        epoch_begin = time.time()
        model.train()
        for batch_examples in tqdm.tqdm(train_set.batch_iter(batch_size=args.batch_size, shuffle=True)):
            def process(header,src_sent,tokenizer):
                length1=len(header)
                flat_src=[]
                for item in src_sent:
                    flat_src.extend(tokenizer._tokenize(item))
                flat=[token for item in header for token in item.tokens]
                flat_head=[]
                for item in flat:
                    flat_head.extend(tokenizer._tokenize(item))
#                length2=len(flat)+length1+len(src_sent)
                length2=len(flat_head)+length1+len(flat_src)
                print(src_sent)
                print([item.tokens for item in header])
                print(flat_src)
                print(flat)
                a=input('hahaha')
                return length2<130
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step and process(e.table.header,e.src_sent,tokenizer)]
            train_iter += 1
            optimizer.zero_grad()
            optimizer1.zero_grad()
#            params1=model.named_parameters()
#            print([param for param,_ in params1])
#            params=model.rnns.named_parameters()
#            print([param for param,_ in params])
#            print([type(param.grad) for _,param in model.rnns.named_parameters()])
#            a=input('ghh;')
            ret_val,_ = model.score(batch_examples)
            loss = -ret_val[0]
            loss1=ret_val[1]
            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_loss1 += 1.0*torch.sum(ret_val[2])
            report_examples += len(batch_examples)
            loss = torch.mean(loss)+0*loss1+0*torch.mean(ret_val[2])

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data.item()
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            loss.backward()
#            print([type(param.grad) for _,param in model.rnns.named_parameters()])
#            
#            print([type(param.grad) for param in model.parameters()])
#            a=input('ghh;')
            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()
            optimizer1.step()
            loss=None
            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f,coverage loss=%.5f' % (train_iter, report_loss / report_examples,report_loss1 / report_examples)
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.
                report_loss1=0

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if args.dev_file and epoch>=6:
#            a=input('gh')
            if epoch % args.valid_every_epoch == 0:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)
                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
                print('[Epoch %d] begin validation2' % epoch, file=sys.stderr)
#                eval_start = time.time()
#                eval_results = evaluation.evaluate(dev_set.examples[:2000], model, evaluator, args,
#                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
#                dev_score = eval_results[evaluator.default_metric]
#
#                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
#                                    epoch, eval_results,
#                                    evaluator.default_metric,
#                                    dev_score,
#                                    time.time() - eval_start), file=sys.stderr)
#                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
#                history_dev_scores.append(dev_score)

        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                a=input('hj')
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args):
    tmpvocab={'null':0}
    tmpvocab1={'null':0,'unk':1}
    tmpvocab2={'null':0,'unk':1}
    tmpvocab3={'null':0,'unk':1}
    train_set = Dataset.from_bin_file(args.train_file)
    from dependency import sentencetoadj,sentencetoextra_message
    valid=0
    for example in tqdm.tqdm(train_set.examples):
#        print(example.src_sent)
        example.mainnode,example.adj,example.edge,_,isv=sentencetoadj(example.src_sent,tmpvocab)
        example.contains,example.pos,example.ner,example.types,example.tins,example.F1=sentencetoextra_message(example.src_sent,[item.tokens for item in example.table.header],[item.type for item in example.table.header],tmpvocab1,tmpvocab2,tmpvocab3,True)
        valid+=isv
#        print(example.adj)
#        a=input('gh')
    print('bukey',valid)
#    if args.dev_file:
#        dev_set = Dataset.from_bin_file(args.dev_file)
#    else: dev_set = Dataset(examples=[])
    test_set = Dataset.from_bin_file(args.test_file)
    for example in tqdm.tqdm(test_set.examples):
#        print(example.src_sent)
        example.mainnode,example.adj,example.edge,_,_=sentencetoadj(example.src_sent,tmpvocab)
        example.contains,example.pos,example.ner,example.types,example.tins,example.F1=sentencetoextra_message(example.src_sent,[item.tokens for item in example.table.header],[item.type for item in example.table.header],tmpvocab1,tmpvocab2,tmpvocab3,False)
    bertmodels=BertModel.from_pretrained('./pretrained_models/base-uncased/')
    tokenizer=BertTokenizer.from_pretrained('./pretrained_models/bert-base-uncased-vocab.txt')
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda,bert=bertmodels,tmpv=tmpvocab,v1=tmpvocab1,v2=tmpvocab2,v3=tmpvocab3)
    parser.tokenizer=tokenizer
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))
def train_rl(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model
    train_set = Dataset.from_bin_file(args.train_file)
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang
    def getnew(model,e,shows):
        example=copy.deepcopy(e)
        hyps,extra = model.sample(example.src_sent, context=example.table,show=shows)
        if(len(hyps)==0):
            return e
        actions=hyps[0].actions
        example.tgt_actions = get_action_infos(example.src_sent, actions, force_copy=True)
        example.actions=actions
        example.extra=extra
        if(shows):print('haha')
        return example
    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    
    parser.train()
    model=parser
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    if args.cuda: model.cuda()
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    for e in train_set.examples:
                e.actions=[item.action for item in e.tgt_actions]
    train_iter=1 
    if(True):
                parser.eval()
#                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(test_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    train_iter, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)
    a=input('df')
    print(len(train_set.examples)/64) 
    bestscore=0            
    for _ in range(100): 
     model.train()   
     show=False    
     for batch_examples in tqdm.tqdm(list(train_set.batch_iter(batch_size=args.batch_size, shuffle=False))[0:]):
            model.eval()
#            if(train_iter>=23):
#                show=True
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            newbatch_examples=[getnew(parser,e,show) for e in batch_examples]
#            print(newbatch_examples[0].extra)
#            model.train() 
            means=[]
            model.eval()
            for example in batch_examples:               
             hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
             if(len(hyps)==0):
                 means.append(None)
             else:means.append(hyps[0])
            import asdl
            weight=asdl.transition_system.get_scores(newbatch_examples,batch_examples,means)
#            if(train_iter>1000):
#            for i in range(len(batch_examples)):
#                print('true'+str(batch_examples[i].actions))
#                print('chou'+str(newbatch_examples[i].actions))
#                print('max'+str(means[i].actions))
#                print(weight[i])
#                a=input('hj')
#            if(train_iter>=22):
#             print(weight)
##            a=input('jj')
            train_iter += 1
            model.train()
            optimizer.zero_grad()

            ret_val,extra = model.score(batch_examples,weights=weight)
            ret_val1,extra1 = model.score(batch_examples)
#            print([torch.exp(item) for item in extra[0]])
#            a=input('gh')
            loss = -ret_val[0]
            loss1= -ret_val1[0]

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
#            report_loss += loss_val
#            report_examples += len(batch_examples)
            loss = torch.mean(loss)+torch.mean(loss1)*0.0

#            if args.sup_attention:
#                att_probs = ret_val[1]
#                if att_probs:
#                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
#                    sup_att_loss_val = sup_att_loss.data.item()
##                    report_sup_att_loss += sup_att_loss_val
#
#                    loss += sup_att_loss

            loss.backward()
#
#            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
#
            optimizer.step()
#            print(train_iter)
            if train_iter % 200 == 0:
                parser.eval()
#                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(test_set.examples[:2000], model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]
                if bestscore < dev_score:
                    bestscore =dev_score
                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    train_iter, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)
                print('hhah'+str(bestscore))
            
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise RuntimeError('unknown mode')
