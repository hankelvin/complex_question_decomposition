import json, tqdm, os
os.environ['TOKENIZERS_PARALLELISM'] = "false"


def main(args):
    from chat_class_inference_utils import (load_model, load_model_rankllm,      
                                            update_args_with_ranker_args)
    from task_utils_decomp_qg import (save_out_decomp_qg_outputs, 
                                        save_out_calibration_data)
    
    os.environ['HUGGINGFACE_HUB_CACHE'] = f'llm_models/{args.model_name}'
    dp = 'data'

    # 1. load data 
    gold_dict = None
    c_do_rank_llm_syndqg_class = args.rerank_task and ('dqg' not in args.task or 'syndqg' in args.task) and args.task != 'rerank'
    if args.task.startswith('rerank'):
        from chat_class_inference_dataloaders import (load_rank_llm_data, load_rank_llm_dqg_data,
                                                      load_rank_llm_syndqg_class_data)
        if args.task in ['rerank']: 
            inference_data = load_rank_llm_data(args, dp)
            args.num_cands = 10 # NOTE: affects window size in rankllm
        
        elif args.task in ['rerank_dqg_pairwise', 'rerank_dqg_listwise', ] or args.task in args.rerank_dqg_tasks:
            inference_data, args.num_cands, gold_dict = load_rank_llm_dqg_data(args)
        
        elif c_do_rank_llm_syndqg_class:
            inference_data, args.num_cands, gold_dict = load_rank_llm_syndqg_class_data(args, shuffle = False)

        else: raise NotImplementedError
    
    elif args.task == 'decomp_qg':
        from chat_class_inference_dataloaders import load_decomp_qg_llm_data
        inference_data = load_decomp_qg_llm_data(args, dp)
    
    else: raise NotImplementedError

    # 2. load model/pipeline 
    if args.task.startswith('rerank'): 
        args                 = update_args_with_ranker_args(args)
        pipeline_model, args = load_model_rankllm(args)
    else: 
        tokenizer, pipeline_model, device, args = load_model(args)
    print('🔮\tModel Loaded:', args.model_name)

    # 3. inference 
    batches = [list(inference_data.keys())[i:i+args.bsz] \
               for i in range(0, len(inference_data), args.bsz)]

    for __, bnums in enumerate(tqdm.tqdm(batches)):
        batch = [inference_data[b] for b in bnums]

        with torch.no_grad():
            tgts = None
            if args.task.startswith('rerank'):
                preds = pipeline_batch_step_rerank(args, pipeline_model, bnums, batch)
            else: 
                # preds, tgts, scores, prompt_repr = pipeline_batch_step(args, pipeline_model, batch)
                preds, tgts = pipeline_batch_step(args, pipeline_model, batch)

        if args.do_print_check: print('*'*50)
        assert len(bnums) == len(preds) == len(batch), \
            f'🚨\t\tMISMATCHED LENGTHS: {len(bnums)} vs {len(preds)} vs {len(batch)}'
        
        # 4. recover predictions
        for i, (idx, prediction) in enumerate(zip(bnums, preds)):
            
            if args.task.startswith('rerank'): 
                batch_instance = batch[i] # Request object
                inference_data = postprocess_one_step_rerank(args, batch_instance, 
                                                             prediction, idx, inference_data)
                
            elif args.task == 'decomp_qg':
                inference_data[idx].pop('original_info')
                inference_data[idx]['pred_subquestions'] = prediction
                inference_data[idx]['gold_subquestions'] = tgts[i]
            
            if args.do_print_check: print(f'\n{idx}\t{inference_data[idx]}')
        if args.do_print_check: print('*'*50, '\n')

    # 5. evaluation
    if args.task.startswith('rerank'):
        args.tokenizer = pipeline_model._agent._tokenizer # see rerank.py and then rank_listwise_os_llm.py
    else: args.tokenizer = pipeline_model.tokenizer
    if args.task == 'rerank':
        import sys 
        sys.path.append('..')
        from tools.beam_retriever.utils.utils_test_model import calculate_em_f1
        for idx, result_instance in inference_data.items():
            r = result_instance['rerank_products']
            preds = [rr-1 for rr in r['response']] # adjust to be 0-index (like gold_supports)
            preds = preds[:result_instance['num_hops']]
            r['f1'], r['em'] = calculate_em_f1(preds, r['gold_supports'])
            r['pred_supports'] = preds
            result_instance.pop('docid_map')

            if r['calibrated_response'] is not None:
                preds = [rr-1 for rr in r['calibrated_response']] # adjust to be 0-index (like gold_supports)
                preds = preds[:result_instance['num_hops']]
                r['f1_calib'], r['em_calib'] = calculate_em_f1(preds, r['gold_supports'])
                r['pred_supports_calib'] = preds

    elif args.task in ['decomp_qg', 'rerank_dqg_pairwise', 'rerank_dqg_listwise']:
        save_out_decomp_qg_outputs(args, inference_data, gold_dict)
        save_out_calibration_data(args, inference_data)

    elif c_do_rank_llm_syndqg_class or args.task in args.rerank_dqg_tasks:
        save_out_calibration_data(args, inference_data)

    else: raise NotImplementedError

    
    print('\t\t🔮 Prediction Complete')
    return inference_data, args
    
def pipeline_batch_step(args, pipeline_model, batch):
    '''
    for LLM inference outside of reranking task
    
    '''
    if 'prepare_decomp_qg_prompt' not in globals():
        from task_utils_decomp_qg import prepare_decomp_qg_prompt
    if 'get_nvidia_llama3_formatted_input' not in globals():
        from chat_class_inference_utils import get_nvidia_llama3_formatted_input

    if args.task == 'decomp_qg':
        assert len(batch) == 1
        messages, tgt, cq = prepare_decomp_qg_prompt(args, batch[0])
    
    else: raise NotImplementedError
    
    gen_args = args.gen_args
    if args.model_name == 'llama':
        terminators = [pipeline_model.tokenizer.eos_token_id,
                        pipeline_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        gen_args = gen_args | {'eos_token_id': terminators, 
                        # in llama, the pad_token_id is not set (rec: use the eos_token_id)
                        'pad_token_id': pipeline_model.tokenizer.eos_token_id,}
    elif args.model_name == ['phi3', 'gritlm_gen']:
        gen_args = gen_args | {'pad_token_id': pipeline_model.tokenizer.pad_token_id,}
    elif args.model_name in ['mistral', 'qwen', 'nvidia_llama3']:
        # in mistral, the pad_token_id is not set (rec: use the eos_token_id)
        gen_args = gen_args | {'pad_token_id': pipeline_model.tokenizer.eos_token_id,}
        
    # a. prepare prompt and run inference
    if args.model_name == 'nvidia_llama3':
        prompt = get_nvidia_llama3_formatted_input(messages = messages, context = '',
                                                    add_generation_prompt = True)
    else: 
        prompt  = pipeline_model.tokenizer.apply_chat_template(messages, tokenize = False, 
                                                               add_generation_prompt = True)   
    if args.do_print_check:
        print('*'*50)
        print('PROMPT:', prompt) 
        print('*'*50)
    
    # b. add other gen_args controls
    if 'do_sample' not in gen_args: gen_args['do_sample'] = False

    outputs = pipeline_model(prompt, **gen_args)
    # remove the prompt portion
    preds = [output["generated_text"][len(prompt):] for output in outputs]
    assert len(preds) == 1
    tgts = [tgt]

    return preds, tgts

def pipeline_batch_step_rerank(args, pipeline_model, bnums, batch):
    '''
    for LLM inference on the reranking task
    # see https://github.com/castorini/rank_llm/blob/main/src/rank_llm/retrieve_and_rerank.py
    
    '''
    rargs = args.ranker_args
    rerank_results = pipeline_model.rerank_batch(
            requests = batch,
            rank_end = rargs['top_k_candidates'],
            window_size = min(rargs['window_size'], rargs['top_k_candidates']),
            shuffle_candidates = rargs['shuffle_candidates'],
            logging = rargs['print_prompts_responses'],
            step = rargs['step_size'])
    

    return rerank_results

def postprocess_one_step_rerank(args, batch_instance, 
                                prediction, idx, inference_data):
    if 'give_rank_docmap_v1' not in globals():
        from chat_class_inference_utils import give_rank_docmap_v1
    if 'give_rank_docmap_v2' not in globals():
        from chat_class_inference_utils import give_rank_docmap_v2

    # recover the gold support info:
    # 1. inside Request under 'candidates' attribute
    # 2. each Candidate object has a 'score' attribute
    # 3. docid is of the form f'{cq_id}_doc{sq_idx}'
    # NOTE: corrected 30/08/24
    gold_supports   = None
    ranking_calibrated = None
    perm_order      = prediction.perm_order
    c1 = args.input_file[0] in ['breakhigh', 'musique', 'hotpot_distractor']
    if args.task == 'rerank' and c1:
        gold_supports = [i for i,c in enumerate(batch_instance.candidates) if c.score > 0]
    else: 
        best_to_worst   = list(reversed(range(len(perm_order))))
        gold_supports   = [perm_order.index(i) for i in best_to_worst]
    
    # NOTE: prediction is a Result object. It is initialised with a set of documents in .candidates
    # as the sliding window moves for a given Request, the .candidates attribute is updated and sorted.
    c_have_calibrated   = getattr(prediction, 'candidates_calibrated', None) is not None
    c_splade100         = 'splade100' in args.input_file[0]
    if args.task == 'rerank':
        ranking, docid_map = give_rank_docmap_v1(prediction.candidates, batch_instance.candidates)
        if c_have_calibrated:
            ranking_calibrated, docid_map_calibrated = \
                give_rank_docmap_v1(prediction.candidates_calibrated, batch_instance.candidates)
            assert docid_map_calibrated == docid_map, (docid_map_calibrated, docid_map)

    elif args.rerank_task and args.task != 'rerank':
        ranking, docid_map = give_rank_docmap_v2(prediction.candidates, batch_instance.candidates,
                                                 skip_sys_prefix = c_splade100)
        if c_have_calibrated:
            ranking_calibrated, docid_map_calibrated = \
                give_rank_docmap_v2(prediction.candidates_calibrated, batch_instance.candidates,
                                    skip_sys_prefix = c_splade100)
            assert docid_map_calibrated == docid_map, (docid_map_calibrated, docid_map)
        

    # ranking = ' > '.join([f'[{str(rank).zfill(2)}]' for rank in __ranking]) # ensure zfill(2)
    # NOTE: corrected 30/08/24
    # rerank_dqg_tasks have model_names instead of numerical docids
    if args.task not in args.rerank_dqg_tasks: 
        ranking     = [int(i) for i in ranking]
            
    
    inference_data[idx] = {'query'          : inference_data[idx].query.text,
                            'rerank_products': {'prompt':           [r.prompt   for r in prediction.ranking_exec_summary], 
                                                # NOTE: r.response gives the label that is according to the order presented to the model
                                                'prompt_response':  [r.response for r in prediction.ranking_exec_summary],
                                                # this is the ranking prediction (docids, after reordering docs from least to most)
                                                # in other words, this is the **predicted** "perm_order" 
                                                # (for those with GT, and ordered least to most but shuffled) 
                                                # or the "preference" (for those without GT)
                                                'response':         ranking, 
                                                # if task = 'rerank', this is the indices of the relevant docs/passages 
                                                # else, this is the positions of the docs ordered by quality (descending)
                                                'gold_supports':    gold_supports, 
                                                'perm_order'   :    perm_order,
                                                },
                            'num_hops'       : getattr(batch_instance, 'num_hops', None),
                            'time(sec)'      : prediction.time,
                            # note: docid_map to be 1-indexed, to follow what's done in prompt
                            'docid_map'      : docid_map,
                            'stance'         : getattr(inference_data[idx].query, 'stance', None)}   

    return inference_data

if __name__ == '__main__':
    ############################ ARGS ############################
    from chat_class_inference_utils import give_save_path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',             type = str, nargs = '*', 
                                                    default = ['musique', 'validation'])
    parser.add_argument('--model_name',             type = str, default = 'llama')
    parser.add_argument('--load_ckpt_path', help = 'The path of the ckpt to load the model weights from.', 
                                                    default = None, type = str)
    parser.add_argument('--model_lineup',           type = int, default = 2)
    parser.add_argument('--prompt_version',         type = int, default = 2)
    parser.add_argument('--model_size',             type = str, default = '')
    parser.add_argument('--hf_token',               type = str, default = '')
    parser.add_argument('--save_loc',               type = str, default = 'results')
    parser.add_argument('--task',                   type = str, default = 'rerank', 
                                help = '"rerank":       reranking task; \
                                "decomp_qg":            decompose question into sub-questions; \
                                "rerank_dqg_pairwise":  rerank a set of N CQ-to-SQ decompositions; window 2, step 1 \
                                "rerank_dqg_{ds}_pairwise":  rerank a set of N CQ-to-SQ decompositions; window 2, step 1 \
                                "rerank_dqg_listwise":  rerank a set of N CQ-to-SQ decompositions; window N, step N  \
                                "rerank_dqg_{ds}_listwise":  rerank a set of N CQ-to-SQ decompositions; window N, step N  \
                                "select_dqg_listwise":  given a list of N CQ-to-SQ decompositions, return the index for the best  \
                                "rerank_syndqggpt4o-{ds}_listwise": rerank a set of N arguments; window N, step 1 \
                                ')
    parser.add_argument('--num_few_shot_examples',  type = int, default = 2, 
                                help = 'the n-shot examples to use for few-shot learning in rankllm')  
    parser.add_argument('--num_ranks',              type = int, default = 4)
    parser.add_argument('--decomp_qg_args',         type = str, default = ['5', 'True'], nargs = '*')
    parser.add_argument('--rerank_dqg_models',      type = str, default = [], nargs= '*',
                                help = 'task-specific control parameters e.g. ["gemma", "llama", "phi", "qwen"]')
    parser.add_argument('--do_print_check',         type = bool, default = False)
    parser.add_argument('--do_trial',               type = bool, default = False)
    parser.add_argument('--rerank_constrained_bs',  type = str, default = ['True', '1'], nargs = '*',
                                help = 'Whether to run constrained beam search for reranking and how many beams')
    args = parser.parse_args()
    args.trial_num = 10
    # args.do_trial  = True
    ############################ ARGS ############################
    
    import torch
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available(): args.device = 'mps'
    # for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    
    for i,v in enumerate(args.rerank_constrained_bs):
        args.rerank_constrained_bs[i] = eval(v)

    args.rerank_shuffle = False
    #############################################

    ### INPUT FILE ###
    if args.input_file[0] == 'hotpotqa_fullwiki' and args.input_file[1] == 'validation':
        args.input_file[0] = 'hotpotqa_distractor'

    ### LLM SETTINGS ###
    modelcookbook = {   'ra': 'rank_zephyr',    'gr': 'gritlm_gen',
                        'll': 'llama',          '31': '31llama',
                        'ph': 'phi3',           'ol': 'olmo',
                        'ge': 'gemma',          'mi': 'mistral',
                        'qw': 'qwen',           'nv': 'nvidia_llama3',
                        'co': 'commandr_plus',  'ay': 'aya',
                        'fl': 'flan'}
    if all(len(s) == 2 for s in args.rerank_dqg_models):
        args.rerank_dqg_models = [modelcookbook[m] for m in args.rerank_dqg_models]
    if len(args.model_name) == 2:
        args.model_name = modelcookbook[args.model_name]
    if args.model_name not in ['llama', 'flan']: args.model_size = ''
    print('\t🔍Model to be used:', args.model_name, args.model_size)

    ### TASK SETTINGS ###
    args.qpos_tokens = [f'[SQ{i+1}]' for i in range(50)]
    if args.task != 'decomp_qg': args.decomp_qg_args = {} 
    args.rerank_with_score_scheme = False
    # if args.task == 'rerank_dqg_pairwise': raise KeyError('\t🚨 "rerank_dqg_pairwise" performs poorly.')
    if args.task.startswith('rerank'): 
        args.gen_args = {'max_new_tokens': 50, 'do_sample': False, 'top_p': None, 'top_k': None, 'temperature': 0.0}
        args.bsz = 1000000
        args.rerank_with_score_scheme = True

    elif args.task == 'decomp_qg':
        args.decomp_qg_args = {key: eval(args.decomp_qg_args[i]) for i, key in enumerate(['n_shot', 'cot'])}
        args.decomp_qg_args = {'n_shot': 5, 'cot': True}
        args.gen_args = {'max_new_tokens': 200, 'do_sample': False, 'top_p': None, 'top_k': None, 'temperature': 0.0}
        if args.decomp_qg_args['cot']: args.gen_args['max_new_tokens'] = 500
        args.bsz = 1
        assert args.bsz == 1, '🚨\t\tBatch size must be 1.'
        args.user_prompt = "Complex question: "
        args.asst_prompt = "Decomposed sub-questions: " if not args.decomp_qg_args['cot'] else ''
        args.rerank_constrained_bs = [False, None]
        assert args.rerank_constrained_bs[0] == False,   '🚨\t\tConstrained beam search not supported for decomp_qg.'

    ### RANK CALIBRATION ###
    args.ds_sizes = {'musique':          {'validation': 2417,  'test': 2459},
                     'breakhigh':        {'validation': 3130,  'test': 3195},}

    C_LIST = ['rerank_dqg_pairwise', 'rerank_dqg_listwise', 'rerank']
    args.rerank_dqg_tasks = []
    for ds in ['syndqggpt4o']:
        for setting in ['pairwise', 'listwise']: 
            # for synthethic dqg calibration 
            ds_strs = [ds] if ds != 'syndqggpt4o' else [f'{ds}-{dd}' \
                                    for dd in ['breakhigh', 'musique']]
            for ds_str in ds_strs: 
                C_LIST.append(f'rerank_{ds_str}_{setting}')
                if ds in ['syndqggpt4o']: args.rerank_dqg_tasks.append(f'rerank_{ds_str}_{setting}')
    for ds in ['breakhigh', 'musique']:
        C_LIST.append(f'rerank_dqg_{ds}_{setting}')
        for setting in ['pairwise', 'listwise']: args.rerank_dqg_tasks.append(f'rerank_dqg_{ds}_{setting}')
    args.rerank_task = args.task in C_LIST
    if args.task not in args.rerank_dqg_tasks: args.num_few_shot_examples = 0

    ### PREP FP! ###
    args = give_save_path(args)
    
    ### RUN! ###
    inference_data, args = main(args)
    
    ### SAVE! ###
    save_fp = os.path.join(args.save_path, 'model_outputs.jsonl')
    with open(save_fp, mode = 'w', encoding = 'utf-8') as f:
        for k, v in inference_data.items():
            # no reraank products for decomp_qg
            if 'rerank_products' in v: v['rerank_products'].pop('prompt_repr', None)
            try: f.write(json.dumps({k: v}) + '\n')
            except: print('Error writing', k, v)
    print('\t\t🔥 Prediction saved to:', save_fp)

    save_fp = save_fp.replace('_outputs.jsonl', '_args.jsonl')
    with open(save_fp, 'w') as f:
        delattr(args, 'tokenizer')
        args.gen_args.pop('constraints') if 'constraints' in args.gen_args else None
        save_args = {k: v for k, v in vars(args).items() if type(v) \
                     in [str, int, bool, float, dict, list, tuple, set, None]}
        for k, v in save_args.items():
            try: f.write(json.dumps({k:v}) + '\n')
            except: f.write(json.dumps({k:str(v)}) + '\n')
    print('\t\t🔥 Model args saved to:', save_fp)