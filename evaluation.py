# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    is_wikisql = args.parser == 'wikisql_parser'

    decode_results = []
    decodeatt_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if is_wikisql:
            hyps,atts = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size,or_example=example)
#            hyps,atts = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size,or_example=example,doubletest=False)
#            hyps = model.sample(example.src_sent, context=example.table)
#            print(hyps[0].actions)
#            
#            h=input('hhh')
        else:
            hyps,atts = model.parse(example.src_sent, context=None, beam_size=args.beam_size,or_example=example)
        decoded_hyps = []
        decoded_atts=[]
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                decoded_hyps.append(hyp)
                decoded_atts.append([atts[0][hyp_id],atts[1][hyp_id]])
            except:
                if verbose:
                    pass
#                    print("Exception in converting tree to code:", file=sys.stdout)
#                    print('-' * 60, file=sys.stdout)
#                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
#                                                                                             ' '.join(example.src_sent),
#                                                                                             example.tgt_code,
#                                                                                             hyp_id,
#                                                                                             hyp.tree.to_string()), file=sys.stdout)
#                    if got_code:
#                        print()
#                        print(hyp.code)
#                    traceback.print_exc(file=sys.stdout)
#                    print('-' * 60, file=sys.stdout)

        count += 1

        decode_results.append(decoded_hyps)
        decodeatt_results.append(decoded_atts)

    if was_training: model.train()

    return decode_results,decodeatt_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=True):
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)
#    eval_result=None
    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
