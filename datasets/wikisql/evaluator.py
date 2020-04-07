import sys
import traceback

from components.evaluator import Evaluator
from common.registerable import Registrable
from datasets.wikisql.lib.query import Query
from datasets.wikisql.lib.dbengine import DBEngine
from datasets.wikisql.utils import detokenize_query
from asdl.lang.sql.sql_transition_system import asdl_ast_to_sql_query
import tqdm

@Registrable.register('wikisql_evaluator')
class WikiSQLEvaluator(Evaluator):
    def __init__(self, transition_system, args):
        super(WikiSQLEvaluator, self).__init__(transition_system=transition_system)

        print(f'load evaluation database {args.sql_db_file}', file=sys.stderr)
        self.execution_engine = DBEngine(args.sql_db_file)
        self.answer_prune = args.answer_prune

    def is_hyp_correct(self, example, hyp):
        hyp_query = asdl_ast_to_sql_query(hyp.tree)
        detokenized_hyp_query = detokenize_query(hyp_query, example.meta, example.table)

        hyp_answer = self.execution_engine.execute_query(example.meta['table_id'], detokenized_hyp_query, lower=True)

        ref_query = Query.from_tokenized_dict(example.meta['query'])
        ref_answer = self.execution_engine.execute_query(example.meta['table_id'], ref_query, lower=True)

        result = ref_answer == hyp_answer

        return result
    def finemet(self, tree1,tree2, example,passed):
        hyp_query = asdl_ast_to_sql_query(tree1)
        detokenized_hyp_query=hyp_query
#        detokenized_hyp_query = detokenize_query(hyp_query, example.meta, example.table).lower()
        c1=detokenized_hyp_query.conditions
        c1=sorted(c1,key=lambda x:str(x))
        

        ref_query = asdl_ast_to_sql_query(tree2)
        c2=ref_query.conditions
        c2=sorted(c2,key=lambda x:str(x))
        issim=True
        if(len(c1)==len(c2)):
            for i in range(len(c1)):
                if str(c1[i])==str(c2[i]):
                    issim=True
                else:
                    issim=False
                    break
        else:issim=False
        if passed and not issim:
         print(detokenized_hyp_query)
         print(ref_query)
         print(c1)
         print(c2)
        
        
#         print(result)
         a=input('haha')
        result = [detokenized_hyp_query.sel_index==ref_query.sel_index,detokenized_hyp_query.agg_index==ref_query.agg_index,issim] 
        return result
    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
#        for example, hyp_list in tqdm.tqdm(zip(examples, decode_results)):
#            if(hyp_list):
#                print(hyp_list[0].actions)
#                print([a.action for a in example.tgt_actions])
#        print('jkhff')
        self.answer_prune=True
        if self.answer_prune:
            filtered_decode_results = []
            for example, hyp_list in tqdm.tqdm(zip(examples, decode_results[0])):
                pruned_hyps = []
                if hyp_list:
                    for hyp_id, hyp in enumerate(hyp_list):
                        try:
                            # check if it is executable
                            detokenized_hyp_query = detokenize_query(hyp.code, example.meta, example.table)
                            hyp_answer = self.execution_engine.execute_query(example.meta['table_id'],
                                                                             detokenized_hyp_query, lower=True)
                            if len(hyp_answer) == 0:
                                continue

                            pruned_hyps.append(hyp)
                            if fast_mode: break
                        except:
                            print("Exception in converting tree to code:", file=sys.stdout)
                            print('-' * 60, file=sys.stdout)
                            print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                                      ' '.join(
                                                                                                          example.src_sent),
                                                                                                      example.tgt_code,
                                                                                                      hyp_id,
                                                                                                      hyp.tree.to_string()),
                                  file=sys.stdout)
                            print()
                            print(hyp.code)
                            traceback.print_exc(file=sys.stdout)
                            print('-' * 60, file=sys.stdout)

                filtered_decode_results.append(pruned_hyps)

            decode_results = [filtered_decode_results,decode_results[1]]

        eval_results = Evaluator.evaluate_dataset(self, examples, decode_results, fast_mode)

        return eval_results
