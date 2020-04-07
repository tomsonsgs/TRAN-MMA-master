# coding=utf-8
#import copy
import numpy as np
def score_acts(actions1,actions2):
    mask1=[0]*len(actions1)
    mask2=[0]*len(actions2)
    leng=len(actions1)+len(actions2)+0.0000001
    join=0
    for i,item1 in enumerate(actions1):
       for j,item2 in enumerate(actions2):
           if item1==item2 and mask1[i]==0 and mask2[j]==0:
               mask1[i]=1
               mask2[j]=1
               join+=1
    return 2*join/leng
def get_single_score(ai,bi,ci,method='action+no_weight',alpha=1):
    length=len(bi.actions)
    if(ci==None):
        return [0.1]*length
    if(len(ai.actions)!=len(bi.actions)):
        return [0.1]*length
    else: 
        count=0
        weight=[]
        for idx,(ais,bis) in enumerate(zip(ai.actions,bi.actions)):
            if ais==bis:
                if(len(ai.actions)!=len(ci.actions)):
                  weight.append(0.1)
                else:
                  if(ais==ci.actions[idx]):
                      weight.append(0.1)
                  else:  weight.append(1.0)
            else:
                count+=1
                weight.append(1.0)
        if count==1 or count==2:
          return weight
    return [0.1]*length
#    delta=score_acts(ai.actions,bi.actions)
#    -score_acts(ci.actions,bi.actions)
#    delta=alpha*delta
#    return [delta]*length
def return_index(a,alist,mask=None):
    isin=False
    if(mask):
     for i,item in enumerate(alist):
        if item==a and mask[i]==0:
            isin=True
            break
    else:
     for i,item in enumerate(alist):
        if item==a:
            isin=True
            break
    if isin:return i
    else:return -1
def score_acts_3(actions1,actions2,actions3):#a1 sample;a2,max;a3,golden
#    mask1=[0]*len(actions1)
    mask2=[0]*len(actions2)
    mask3=[0]*len(actions3)
#    leng=len(actions1)+len(actions2)+0.0000001
    join=[]
    for i,item1 in enumerate(actions1[::-1]):
        index1=return_index(item1,actions2,mask2)
        if index1>=0:
            mask2[index1]=1
        index2=return_index(item1,actions3,mask3)
        if index2>=0:
            mask3[index2]=1
        if index1==-1 and index2==-1:
            r=-1
        elif index1==-1 and index2!=-1:
            r=1
        elif index1!=-1 and index2!=-1:
            r=0.3
        else:
            r=-1
        join.append(r)
    return np.array(join[::-1])
def get_scores(a,b,c):
    weights=[None for _ in range(len(c))]
    for idx,(ai,bi,ci) in enumerate(zip(a,b,c)):
        weights[idx]=get_single_score(ai,bi,ci)
    return weights
class Action(object):
    pass


class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()


class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token
    def __eq__(self, other):
        return isinstance(other, GenTokenAction) and self.token == other.token

class ReduceAction(Action):
   def __repr__(self):
       return 'Reduce'
   def __eq__(self, other):
        return isinstance(other, ReduceAction)

class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast):
        """
        generate action sequence given the ASDL Syntax Tree
        """

        actions = []

        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            # is a composite field
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions = self.get_actions(field.value)
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions = self.get_actions(val)
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions = self.get_actions(field.value)

                    # if an optional field is filled, then do not need Reduce action
                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        field_actions.append(ReduceAction())
            else:  # is a primitive field
                field_actions = self.get_primitive_field_actions(field)

                # if an optional field is filled, then do not need Reduce action
                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    # reduce action
                    field_actions.append(ReduceAction())

            actions.extend(field_actions)

        return actions

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]

    @staticmethod
    def get_class_by_lang(lang):
        if lang == 'python':
            from .lang.py.py_transition_system import PythonTransitionSystem
            return PythonTransitionSystem
        elif lang == 'python3':
            from .lang.py3.py3_transition_system import Python3TransitionSystem
            return Python3TransitionSystem
        elif lang == 'lambda_dcs':
            from .lang.lambda_dcs.lambda_dcs_transition_system import LambdaCalculusTransitionSystem
            return LambdaCalculusTransitionSystem
        elif lang == 'prolog':
            from .lang.prolog.prolog_transition_system import PrologTransitionSystem
            return PrologTransitionSystem
        elif lang == 'wikisql':
            from .lang.sql.sql_transition_system import SqlTransitionSystem
            return SqlTransitionSystem

        raise ValueError('unknown language %s' % lang)
