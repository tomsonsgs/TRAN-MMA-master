from __future__ import print_function
from asdl.transition_system import GenTokenAction, TransitionSystem, ApplyRuleAction, ReduceAction,score_acts
import sys, traceback
import numpy as np
from common.registerable import Registrable
import tqdm
cachepredict=[]
cachetrue=[]
from dependency import nlp
from nltk.tree import Tree
@Registrable.register('default_evaluator')
class Evaluator(object):
    def __init__(self, transition_system=None, args=None):
        self.transition_system = transition_system
        self.default_metric = 'accuracy'

    def is_hyp_correct(self, example, hyp):
        return self.transition_system.compare_ast(hyp.tree, example.tgt_ast)#this func is at

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        global cachepredict
        global cachetrue
        correct_array = []
        oracle_array = []
        cachepredict=[]
        cachetrue=[]
        allstats=[]
        for example, hyp_list,atts in tqdm.tqdm(zip(examples, decode_results[0],decode_results[1])):
            if fast_mode:
                hyp_list = hyp_list[:1]
                att=atts[:1]
#                ast=attss[:1]
            if hyp_list:
                if(hyp_list[0].tree.sort_removedup_self().to_string()!=example.tgt_ast.sort_removedup_self().to_string()):
                    print(example.src_sent)
##                    tree=Tree.fromstring(str(nlp.parse(' '.join(example.src_sent))))
#                    
                    print([item.name for item in example.table.header])
                    print(att[0][1][0])
                    print(hyp_list[0].actions)
                    print([a.action for a in example.tgt_actions])
#                    
                    for action,at,ats in zip(hyp_list[0].actions,att[0][0],att[0][0]):
#                    if(show and number>3):
#                        if np.linalg.norm(at[-1]-at[0])>0.5:
                             print(example.src_sent) 
                             print(action)
                             print(at)
#                             print(ats)
#                             a=input('jk')
#                    tree.draw()         
                    a=input('jk')
#                 show=False
#                 number=0
#                 for at in att[0]:
#                     if np.linalg.norm(at[-1]-at[0])>0.5:
#                         show=True
#                         number+=1
#                 for action,at in zip(hyp_list[0].actions,att[0]):
#                    if(show and number>3):
#                        if np.linalg.norm(at[-1]-at[0])>0.5:
#                             print(example.src_sent) 
#                             print(action)
#                             print(at)
#                             a=input('jk')
#                for hyp_id, hyp in enumerate(hyp_list):
#                    try:
#                        is_correct = self.is_hyp_correct(example, hyp)
#                    except:
#                        is_correct = False
#
#                        print('-' * 60, file=sys.stdout)
#                        print('Error in evaluating Example %s, hyp %d {{ %s }}' % (example.idx, hyp_id, hyp.code),
#                              file=sys.stdout)
#
#                        print('example id: %s, hypothesis id: %d' % (example.idx, hyp_id), file=sys.stdout)
#                        traceback.print_exc(file=sys.stdout)
#                        print('-' * 60, file=sys.stdout)
#
#                    hyp.is_correct = is_correct
#
#                correct_array.append(hyp_list[0].is_correct)
#                correct_array.append(hyp_list[0].tree.to_string()==example.tgt_ast.to_string())
                correct_array.append(hyp_list[0].tree.sort_removedup_self().to_string()==example.tgt_ast.sort_removedup_self().to_string())
#                hyp_list[0].is_correct)
#                cachepredict.append(hyp_list[0].tree)
#                cachetrue.append(example.tgt_ast)
#                print(hyp_list[0].actions)
#                print([a.action for a in example.tgt_actions])
#                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
#                print(hyp_list[0].tree.to_string())
#                print(example.tgt_ast.to_string())
#                print(score_acts(hyp_list[0].actions,self.transition_system.get_actions(example.tgt_ast)))
#                p=input('gg')
                oracle_array.append(hyp_list[0].tree.sort_removedup_self().to_string()==example.tgt_ast.sort_removedup_self().to_string())
                allstats.append(self.finemet(hyp_list[0].tree,example.tgt_ast,example,oracle_array[-1]))
#                allstats.append([False,False,False])
            else:
                correct_array.append(False)
                oracle_array.append(False)
                allstats.append([False,False,False])

        acc = np.average(correct_array)
        allacc=np.mean(np.array(allstats),0)
        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc,allaccs=allacc)

        return eval_results


@Registrable.register('cached_evaluator')
class CachedExactMatchEvaluator(Evaluator):
    def is_hyp_correct(self, example, hyp):
        raise hyp.is_correct

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        if fast_mode:
            acc = sum(hyps[0].is_correct for hyps in decode_results if len(hyps) > 0) / float(len(examples))
            return acc

        acc_array = []
        oracle_array = []
        for hyp_list in decode_results:
            acc_array.append(hyp_list[0].is_correct if hyp_list else False)
            oracle_array.append(any(hyp.is_correct for hyp in hyp_list))

        return dict(accuracy=np.average(acc_array),
                    oracle_array=np.average(oracle_array))
