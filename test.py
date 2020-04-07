# -*- coding: utf-8 -*-


import json
import sys 
import torch
import numpy as np
import argparse
from test1 import a
from common.utils import update_args, init_arg_parser
#from common.utils import update_args, init_arg_parser
from common.registerable import Registrable
from asdl.lang.sql.sql_transition_system import SqlTransitionSystem,sql_query_to_asdl_ast,asdl_ast_to_sql_query
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree
from asdl.transition_system import GenTokenAction, TransitionSystem, ApplyRuleAction, ReduceAction,score_acts
#a=test1.numpy.array(1)
#from asdl_ast import RealizedField, AbstractSyntaxTree
#from transition_system import GenTokenAction, TransitionSystem, ApplyRuleAction, ReduceAction
from datasets.wikisql.lib.query import Query
from datasets.wikisql.lib.dbengine import DBEngine
data_file = './data_model/wikisql/train.jsonl'
#def init_config():
#    args = arg_parser.parse_args()
#
#    # seed the RNG
#    torch.manual_seed(args.seed)
#    if args.cuda:
#        torch.cuda.manual_seed(args.seed)
#    np.random.seed(int(args.seed * 13 / 7))
#
#    return args
#arg_parser = argparse.ArgumentParser()
#arg_parser.add_argument('-no_parent_production_embe',default=False, action='store_true',
#                            help='Do not use embedding of parent ASDL production to update decoder LSTM state')
#args = arg_parser.parse_args()
##args = init_config()
##args=init_config()
#print(args.no_parent_production_embe)
tmp=[]
engine = DBEngine('./data_model/wikisql/data/train.db')
grammar = ASDLGrammar.from_text(open('./asdl/lang/sql/sql_asdl.txt').read())
transition_system = SqlTransitionSystem(grammar)
if(True):
    from asdl.hypothesis import Hypothesis
    for ids,line in enumerate(open(data_file,encoding='utf-8')):
        example = json.loads(line)
        print(example['sql'])
        query = Query.from_dict(example['sql']).lower()
        print(query)
        asdl_ast = sql_query_to_asdl_ast(query, grammar)
        asdl_ast.sanity_check()
        print(asdl_ast.to_string())
#        asdl_ast.sort_removedup_self()
#        print(asdl_ast.to_string())
#        a=input('fff')
        actions = transition_system.get_actions(asdl_ast)
        tmp.append(actions)
        hyp = Hypothesis()
        print(actions)
        for action in actions:
            hyp.apply_action(action)
        print(hyp.tree)
#        a=input('fff')
#        if asdl_ast_to_sql_query(hyp.tree) != asdl_ast_to_sql_query(asdl_ast):
        if(True):
             hyp_query = asdl_ast_to_sql_query(hyp.tree)
#             make sure the execution result is the same
             hyp_query_result = engine.execute_query(example['table_id'], hyp_query)
             ref_result = engine.execute_query(example['table_id'], query)
             print(query)
             print(ref_result)
             assert hyp_query_result == ref_result
        query_reconstr = asdl_ast_to_sql_query(asdl_ast)
        assert query == query_reconstr
        if(ids>10):break
        print(query)
