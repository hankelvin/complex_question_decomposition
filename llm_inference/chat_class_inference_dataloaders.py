from collections import defaultdict
import json, re, random


### For RankLLM ###
def load_rank_llm_data(args, dp):
    import sys
    sys.path.append('tools')
    from rank_llm.src.rank_llm.data import Query, Candidate, Request

    dataset, split = args.input_file
    fp = '{}/01_unified_format/UNIFIED_{}_{}.jsonl'.format(dp, dataset, split)
    ranker_requests = {}
    with open(fp, encoding = 'utf-8') as f:
        for l in f: 
            line = json.loads(l)
            cq_id = line['id']
            qs = line['text']['qs']
            num_hops = line['num_hops']
            if type(qs) == list: qs = qs[0]
            elif type(qs) == str: pass
            else: raise NotImplementedError(f'🚨\t\tUnknown type: {type(qs)}')
            r = Request(Query(text = qs, qid = cq_id))
            
            index_start = 1
            if dataset in ['musique']: 
                key = 'paragraphs'
                for c in line['original_info'][key]:
                    c_id = cq_id + f'_doc{c["idx"]+index_start}'
                    if args.input_file[1] != 'test':
                        c_score = 1.0 if c['is_supporting'] else 0.0 
                    else: c_score = 0.0
                    candidate = Candidate(docid = c_id, score = c_score, 
                                          doc = {'title': c['title'], 'text': c['paragraph_text']})
                    r.candidates.append(candidate)
            
            elif dataset in ['hotpotqa_fullwiki', 'hotpotqa_distractor']:
                key = 'context'
                
                if split == 'validation': # distractor setting
                    supporting_facts = [t[0].strip().lower() for t in line['original_info']['supporting_facts']]
                    titles = [i[0] for i in line['original_info'][key]]
                    paragraphs = [' '.join(i[1]) for i in line['original_info'][key]]
                else: # fullwiki setting
                    supporting_facts = [t.strip().lower() for t in line['original_info']['supporting_facts']['title']]
                    titles = line['original_info'][key]['title']
                    paragraphs = [' '.join(s) for s in line['original_info'][key]['sentences']]
                assert len(titles) == len(paragraphs), f'🚨\t\tMismatched lengths: {len(titles)} vs {len(paragraphs)}'
                
                for i, (t, p) in enumerate(zip(titles, paragraphs)):
                    c_id = cq_id + f'_doc{i+index_start}'
                    c_score = 1.0 if t.strip().lower() in supporting_facts else 0.0
                    candidate = Candidate(docid = c_id, score = c_score, 
                                          doc = {'title': t, 'text': p})
                    r.candidates.append(candidate)

            else: raise NotImplementedError
            r.num_hops = num_hops # to be used for evaluation
            r.perm_order = list(range(len(r.candidates)))
            ranker_requests[cq_id] = r

            if args.do_trial and len(ranker_requests) >= args.trial_num: break

            if len(r.candidates) != 20: print('HERE 1', cq_id, len(r.candidates))

    print(f'🔮\tData for reranker task loaded with {len(ranker_requests)} requests.')

    return ranker_requests
            

def load_rank_llm_dqg_data(args):
    if 'clean_sysname' not in globals():
        from chat_class_inference_utils import clean_sysname
    import sys, glob, pandas as pd
    random.seed(54506)
    sys.path.append('tools')
    from rank_llm.src.rank_llm.data import Query, Candidate, Request

    dataset, split = args.input_file

    if args.device == 'mps': fps = glob.glob(f'results/decomp_qg/*{dataset}_{split}*')
    else:                    fps = glob.glob(f'results/decomp_qg/*{dataset}_{split}*')
    
    # 1. exclude large models and STV models, and if not of the same prompt version/model_lineup generation
    fps = sorted([fp for fp in fps if 'large' not in fp and 'STV' not in fp])
    if args.prompt_version > 1:     fps = sorted([fp for fp in fps if f'_prmptv{args.prompt_version}' in fp])
    else:                           fps = sorted([fp for fp in fps if f'_prmptv' not in fp])
    
    if args.model_lineup > 1:       fps = sorted([fp for fp in fps if f'_modelline-{args.prompt_version}' in fp])
    else:                           fps = sorted([fp for fp in fps if f'_modelline' not in fp])
    
    # 2. filter based on args.rerank_dqg_models
    print('\t🔍rerank_dqg_models', args.rerank_dqg_models)
    if args.rerank_dqg_models:
        fps_keep = []
        for fp in fps:
            if any([re.search(rf'_{split}_{m}', fp) for m in args.rerank_dqg_models]):
                if 'rerank_syndqg' in fp: continue # these datasets are handled via load_rank_llm_syndqg_class_data
                fps_keep.append(fp)
        fps = fps_keep
    print(f'\t🔮Found {len(fps)} DQG result folders for {dataset} {split}')

    model_outs_dict = defaultdict(dict)
    gold_dict = {}
    op_key_gold, op_key_pred = 'operators_tgt', 'operators_gen'
    systems = set()
    for i, fp in enumerate(fps):
        sysname = fp.split(f'_{split}_')[-1]
        sysname = clean_sysname(sysname)
        systems.add(sysname)
        print('\t🚧Loading Rerank DQG data for:', sysname)
        try:
            df_gold = pd.read_csv(fp+'/text_labels.csv')
            df_gold.columns = [f'gold_{i}' if i in ['decomposition', 'operators'] else i for i in df_gold.columns]
            df_pred = pd.read_csv(fp+'/text_predictions.csv')
            df_pred.columns = [f'pred_{i}' if i in ['decomposition', 'operators'] else i for i in df_pred.columns]
            assert len(df_gold) == len(df_pred), f'🚨\t\tMismatched lengths: {len(df_gold)} vs {len(df_pred)}'

            df_holder = pd.concat([df_gold[['question_id', 'question_text', 
                                            'gold_decomposition', 'operators_tgt']], df_pred], axis=1)
            
            for line_i, line in df_holder.iterrows():
                cq_id = line['question_id']

                if cq_id not in gold_dict:
                    cq = line['question_text']
                    sqs_list = line['gold_decomposition'].split(' ;')
                    r = Request(Query(text = cq, qid = cq_id))
                    # same format as supervised set-up
                    
                    if dataset == 'breakhigh':
                        r.operators_gold = eval(line[op_key_gold])
                        if len(r.operators_gold) < len(sqs_list): # issue with parse of gold operators for gemma
                            print('\t🟥Mismatched (short) gold_operators-sqs lengths:', len(r.operators_gold), len(sqs_list), cq_id, sysname, r.operators_gold)
                            continue
                        if  args.prompt_version == 1:
                            sqs_list = ''.join([args.qpos_tokens[i] + r.operators_gold[i] + q for i, q in enumerate(sqs_list)])
                        elif args.prompt_version > 1:
                            sqs_list = ' '.join([f"{args.qpos_tokens[i]} {r.operators_gold[i]} {q}" for i, q in enumerate(sqs_list)])

                    else: 
                        if   args.prompt_version == 1:
                            sqs_list = ''.join([args.qpos_tokens[i] + q for i,q in enumerate(sqs_list)])
                        elif args.prompt_version > 1:
                            sqs_list = ' '.join([f"{args.qpos_tokens[i]} {q}" for i,q in enumerate(sqs_list)])
                    
                    gold_dict[cq_id] = {'question': cq, 'decomposition': sqs_list, 'request': r}

                sqs_list = line['pred_decomposition'].strip()
                if sqs_list == '': pass # certain LLM may have degenerate outputs for some questions
                else: 
                    sqs_list = sqs_list.split(' ;')
                    if dataset == 'breakhigh':
                        r.operators_pred = eval(line[op_key_pred])
                        if len(r.operators_pred) < len(sqs_list): 
                            print('🚨Mismatched (short) operator_pred-sqs lengths:', 
                                len(r.operators_pred), len(sqs_list), cq_id, sysname, r.operators_pred, sqs_list)
                            short = len(sqs_list) - len(r.operators_pred)
                            r.operators_pred = r.operators_pred + ['[none]']*short
                        elif len(r.operators_pred) > len(sqs_list): 
                            print('🚨Mismatched (excess) operator_pred-sqs lengths:', 
                                len(r.operators_pred), len(sqs_list), cq_id, sysname, r.operators_pred, sqs_list)
                            r.operators_pred = r.operators_pred[:len(sqs_list)]
                        
                        if args.prompt_version == 1:
                            sqs_list = ''.join([args.qpos_tokens[i] + r.operators_pred[i] + q for i, q in enumerate(sqs_list)])
                        elif args.prompt_version > 1:
                            sqs_list = ' '.join([f"{args.qpos_tokens[i]} {r.operators_pred[i]} {q}" for i, q in enumerate(sqs_list)])
                    
                    else: 
                        if args.prompt_version == 1:
                            sqs_list = ''.join([args.qpos_tokens[i] + q for i,q in enumerate(sqs_list)])
                        elif args.prompt_version > 1:
                            sqs_list = ' '.join([f"{args.qpos_tokens[i]} {q}" for i, q in enumerate(sqs_list)])
                
                model_outs_dict[cq_id][sysname] = sqs_list

        except Exception as e: 
            print('\t\t🚨Error in loading:', e, sysname, fp)
            raise ValueError(f'Error loading data for {cq_id} from {sysname}: {e}')
            continue

    # Some systems may not have generated outputs for all questions
    # we will add a generic generation error message for such cases
    error_gen_msg_missing   = '[ERROR]: MODEL DID NOT GENERATE PREDICTION'
    error_gen_msg_long      = '[ERROR]: MODEL GENERATED EXCESSIVELY LONG PREDICTION'
    fail_ctr = defaultdict(int)
    for cq_id, model_outs in model_outs_dict.items():
        missing  = systems - set(model_outs.keys())
        for sysname in missing: 
            print(f'🚨\tNOTE: Missing output from {sysname} for {cq_id} setting to:', error_gen_msg_missing)
            model_outs[sysname] = error_gen_msg_missing 
            fail_ctr[sysname] += 1

    ranker_requests = {}
    num_cands = []
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained('nvidia/Llama3-ChatQA-1.5-8B') # not gated, llama3-based
    for cq_id, model_outs in model_outs_dict.items():
        num_cands.append(len(model_outs))
        r = gold_dict[cq_id]['request']
        sysnames = list(model_outs.keys())
        # shuffle to randomise order of systems presented for reranking
        random.shuffle(sysnames)
        for sysname in sysnames:
            sqs_pred = model_outs[sysname]
            CUT_OFF = 150
            toks = t.encode(sqs_pred)
            if len(toks) > CUT_OFF: 
                print(f'\t\t🚨Truncated prediction for {cq_id} from {sysname} from:', sqs_pred)
                text = error_gen_msg_long
                print(f'\t\t🚨to:', text)
                fail_ctr[sysname] += 1
            else: text = sqs_pred
            candidate = Candidate(docid = f'sys-{sysname}', score = 0.0, 
                                    doc = {'text': text})
            r.candidates.append(candidate)
        
        r.perm_order = list(range(len(r.candidates)))
        ranker_requests[cq_id] = r
        if args.do_trial and len(ranker_requests) >= args.trial_num: break
    
    print('\t🟥 FAILED DISTRIBUTION', fail_ctr)
    assert len(set(num_cands)) == 1, f'🚨\t\tMismatched number of votes cast: {len(set(num_cands))}'
    num_cands = num_cands[0]

    print(f'🔮\tData for DQG reranker task loaded with {len(ranker_requests)} requests. #{num_cands} votes cast.')
    return ranker_requests, num_cands, gold_dict


def load_rank_llm_syndqg_class_data(args, shuffle):
    import sys
    random.seed(54506)
    sys.path.append('tools')
    from rank_llm.src.rank_llm.data import Query, Candidate, Request

    dataset, split = args.input_file
    num_ranks   = args.num_ranks
    print('\t🔍Loading synthetic DQG data for:', dataset, split, num_ranks)
    fps = [f'data/01_unified_format/UNIFIED_RANK_{dataset}-{num_ranks}_{split}.jsonl',]

    gold_dict = {}
    ranker_requests = {}
    num_cands = []
    for fp in fps:
        with open(fp, encoding = 'utf-8') as f:
            src_data = [json.loads(l) for l in f]
        for idx, line in enumerate(src_data):
            if 'splade' in dataset: idx = line['query_id']
            topic       = line['query']
            arguments   = line['paragraphs']
            scores      = line['scores']
            if shuffle: random.shuffle(arguments)

            ######################################################################################################
            # merge the SQs into a single string (see load_rank_llm_dqg_data above)
            for cand_id, sqs_list in enumerate(arguments):
                if dataset == 'breakhigh':
                    r.operators_pred = eval(line['operators']) if type(line['operators']) != list else line['operators']
                    if len(r.operators_pred) < len(sqs_list): 
                        print('🚨Mismatched (short) operator_pred-sqs lengths:', 
                            len(r.operators_pred), len(sqs_list), idx, r.operators_pred, sqs_list)
                        short = len(sqs_list) - len(r.operators_pred)
                        r.operators_pred = r.operators_pred + ['[none]']*short
                    elif len(r.operators_pred) > len(sqs_list): 
                        print('🚨Mismatched (excess) operator_pred-sqs lengths:', 
                            len(r.operators_pred), len(sqs_list), idx, r.operators_pred, sqs_list)
                        r.operators_pred = r.operators_pred[:len(sqs_list)]
                    
                    if  args.prompt_version  == 1:
                        sqs_list = ''.join([args.qpos_tokens[i] + r.operators_pred[i] + q for i, q in enumerate(sqs_list)])
                    elif args.prompt_version  > 1:
                        sqs_list = ' '.join([f"{args.qpos_tokens[i]} {r.operators_pred[i]} {q}" for i, q in enumerate(sqs_list)])
                
                else:                         
                    try: 
                        if   args.prompt_version  == 1:
                            sqs_list = ''.join([args.qpos_tokens[i] + q for i,q in enumerate(sqs_list)])
                        elif args.prompt_version  > 1:
                            sqs_list = ' '.join([f"{args.qpos_tokens[i]} {q}" for i,q in enumerate(sqs_list)])

                    except: print('Error parsing sqs:', idx, sqs_list, )
                
                arguments[cand_id] = sqs_list
            ######################################################################################################

            r = Request(Query(text = topic, qid = idx))
            gold_dict[idx] = {'question':    topic, 
                            'decomposition': arguments, 
                            'request':       r, 
                            'original_info': line}
            
            num_cands.append(len(arguments))
            r.perm_order = list(range(len(arguments)))
            if args.rerank_shuffle:
                random.shuffle(r.perm_order)
        
            for aid in r.perm_order:
                a = arguments[aid]
                docid = f'sys-{aid}'
                if 'splade' in dataset: docid = line["doc_ids"][aid]
                candidate = Candidate(docid = docid, score = float(scores[aid]), 
                                        doc = {'text': a})
                r.candidates.append(candidate)
            
            ranker_requests[idx] = r
            if args.do_trial and len(ranker_requests) >= args.trial_num: break

    assert len(set(num_cands)) == 1, f'🚨\t\tMismatched number of votes cast: {len(set(num_cands))}'
    num_cands = num_cands[0]

    print(f'🔮\tData for DQG reranker task loaded with {len(ranker_requests)} requests. #{num_cands} votes cast.')
    return ranker_requests, num_cands, gold_dict


def load_decomp_qg_llm_data(args, dp, bypass_do_trial = False):
    dataset, split = args.input_file
    c1 = split == 'test'
    c2 = args.test_only == 'phase3_val_as_test' and split == 'validation'
    if args.cross_domain and (c1 or c2):
        dataset = args.cross_domain
    fp = '{}/01_unified_format/UNIFIED_{}_{}.jsonl'.format(dp, dataset, split)
    decomp_qg_requests = {}
    with open(fp, encoding = 'utf-8') as f:
        for l in f:
            line = json.loads(l)
            idx = line['id']
            decomp_qg_requests[idx] = line

            if dataset == 'breakhigh':
                if split == 'test': continue
                line['operators'] = eval(line['original_info']['operators'])
            
            line['dataset'] = dataset
            
            if args.do_trial and not bypass_do_trial and len(decomp_qg_requests) >= args.trial_num: break
    
    print(f'🔮\tData for decomp_qg task loaded with {len(decomp_qg_requests)} requests.')
    return decomp_qg_requests


break_operators_definitions = {
 '[aggregate]':     'Return [the aggregate] of [SQX_i]',
 '[arithmetic]':    'Return the [sum/difference/division] of [SQX_i] and [SQX_j]',
 '[boolean]':       'Return [if / is] [SQX_i] [meeting a condition] [SQX_j]',
 '[comparative]':   'Return [SQX_i] where [SQX_j] [is more / less] [a number]',
 '[comparison]':    'Return [SQX_i] where [SQX_j] [is more / less] [a number]',
 '[discard]':       'Return [SQX_i] besides [SQX_j]',
 '[filter]':        'Return [subset of SQX_i] [meeting a condition]',
 '[group]':         'Return [the aggregate] of [SQX_i] for each [SQX_j]',
 '[intersection]':  'Return [a set with a certain relation] in both [SQX_i] and [SQX_j]',
 '[project]':       'Return [a set with a certain relation] of [SQX_i]',
 '[select]':        'Return [set of entities]',
 '[sort]':          'Return [SQX_i] sorted by [SQX_j]',
 '[superlative]':   'Return [SQX_i] where [SQX_j] is [highest / lowest]',
 '[union]':         'Return [SQX_i], [SQX_j]'}