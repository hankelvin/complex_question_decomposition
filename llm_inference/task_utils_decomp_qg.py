import json, os, re, tqdm
import torch

##### Loading prompt support data #####
fp = 'data/02_support_data/2wiki_breakhigh_musique_prompt_cands.json'
with open(fp, encoding = 'utf-8') as f:
    PROMPT_CANDS = json.load(f)

fp = 'data/02_support_data/breakhigh_hotpot_ans.jsonl'
with open(fp, encoding = 'utf-8') as f:
    PROMPT_CANDS['breakhigh_hotpotqa'] = [json.loads(l) for l in f]
#######################################

def prepare_decomp_qg_prompt(args, line, repro = True, split = None):
    dataset, split = args.input_file
    c1 = split == 'test'
    c2 = args.test_only == 'phase3_val_as_test' and split == 'validation'
    if args.cross_domain and (c1 or c2):
        dataset = args.cross_domain
    n_shot = args.decomp_qg_args['n_shot']
    cot    = args.decomp_qg_args['cot']

    messages = give_prefix_decomp_qg(args.model_name, prompt_version = args.prompt_version, 
                                     n_shot = n_shot, cot = cot, )

    if n_shot > 0:
        examples = give_n_shot_examples(dataset, n_shot)  
        for i, exline in enumerate(examples):
            if i == 0:
                content = 'Let\'s do some examples of how to decompose the complex question into simpler sub-questions:'
                content += f'\n{args.user_prompt}' + exline['cq']
            else: content = f'{args.user_prompt}' + exline['cq']
            
            sqs             = exline['sq_list']
            sqs_ans         = exline['sq_ans_list']
            sqs_ans_dict    = {i+1: a for i, a in enumerate(sqs_ans)} if sqs_ans else None
            
            if exline.get('dataset', None) == 'breakhigh':
                operators = [f'[{o.lower()}]' for o in exline.get('operators', '')]
                sq_sequence = ''.join([args.qpos_tokens[i] + operators[i] + q for i, q in enumerate(sqs)])
                if args.prompt_version > 1 and not repro:
                    sq_sequence =''.join([f"{args.qpos_tokens[i]} {operators[i]} {q}" for i, q in enumerate(sqs)])
            else: 
                sq_sequence = ''.join([args.qpos_tokens[i] + q for i,q in enumerate(sqs)])
                if args.prompt_version > 1 and not repro:
                    sq_sequence =''.join([f"{args.qpos_tokens[i]} {q}" for i, q in enumerate(sqs)])
            
            cot_content = produce_cot_one_example(args, sqs, sqs_ans_dict, dataset) if cot else ''
            messages.extend([{'role': 'user', 'content': content},
                             {'role': 'assistant', 'content': f'{args.asst_prompt}{cot_content}{sq_sequence}'}])
    
    #####################
    cq = line['text']['qs']
    assert len(cq) == 1, f'🚨\t\tExpected 1 complex question, found {len(cq)}'
    cq = cq[0]
    messages.extend([{'role': 'user', 'content': f'{args.user_prompt}{cq}'},])
    
    if split in ['test'] or dataset.startswith('hotpotqa'):
        tgt = '[NONE]'
    else: 
        sqs     = line['decomp_qs']['text']['qs_var']
        sqs_ans = line['decomp_qs']['text'].get('as', None)
        sqs_ans_dict = {i+1: a for i, a in enumerate(sqs_ans)} if sqs_ans else None

        if line.get('dataset', None) == 'breakhigh':
            operators = [f'[{o.lower()}]' for o in line.get('operators', '')]
            if getattr(args, 'use_dqg_llmgen', False):
                short = len(sqs) - len(operators)
                operators += ['[none]'] * short
            sq_sequence = ''.join([args.qpos_tokens[i] + operators[i] + q for i, q in enumerate(sqs)])
            if args.prompt_version > 1 and not repro:
                sq_sequence =''.join([f"{args.qpos_tokens[i]} {operators[i]} {q}" for i, q in enumerate(sqs)])
        else: 
            sq_sequence = ''.join([args.qpos_tokens[i] + q for i,q in enumerate(sqs)])
            if args.prompt_version > 1 and not repro:
                sq_sequence =''.join([f"{args.qpos_tokens[i]} {q}" for i, q in enumerate(sqs)])

        cot_content = produce_cot_one_example(args, sqs, sqs_ans_dict, dataset) if cot else ''
        tgt = f'{args.asst_prompt}{cot_content}{sq_sequence}'

    return messages, tgt, cq

def produce_cot_one_example(args, sqs, sqs_ans_dict, dataset):
    # NOTE: answers for prompts can be used for use_dqg_llmgen (the CoT and few-shot examples are drawn from a 
    # set of prompt examples which are not touched by use_dqg_llmgen (see give_n_shot_examples) )
    cot_content  = ''
    for sq_pos, sq in enumerate(sqs):
        if dataset not in ['breakhigh'] and not sq.endswith('?'): sq += '?'
        
        if sq_pos == 0: 
            cot_content += f'Let\'s think step-by-step. To decompose this complex question, first, I will want to ask: "{sq}". '
        
        elif sq_pos > 0 and sq_pos < len(sqs) - 1: 
            cot_content += 'Then, '
            addline = give_cot_reference_prev_answers(args, sq, sq_pos, sqs_ans_dict) 
            if addline: cot_content += addline
            if sqs_ans_dict is not None:
                answer_var = re.findall(r'(?:#)(\d+)', sq)
                answer_var = [int(v) for v in answer_var if int(v) < sq_pos+1]
                for av in answer_var: 
                    # llm generated decompositions might have answer var errors
                    if getattr(args, 'use_dqg_llmgen', False):
                        try: sq = sq.replace(f'#{av}', sqs_ans_dict[av])
                        except: pass
                    else: sq = sq.replace(f'#{av}', sqs_ans_dict[av])
            
            cot_content += f'I will look for the answer to: "{sq}". '
        
        elif sq_pos == len(sqs) - 1: 
            cot_content += f'Finally, '
            addline = give_cot_reference_prev_answers(args, sq, sq_pos, sqs_ans_dict) 
            
            if addline: cot_content += addline
            else: cot_content += 'given the answers to the previous questions,'

            if sqs_ans_dict is not None:
                answer_var = re.findall(r'(?:#)(\d+)', sq)
                answer_var = [int(v) for v in answer_var if int(v) < sq_pos+1]
                for av in answer_var: 
                    if getattr(args, 'use_dqg_llmgen', False):
                        try: sq = sq.replace(f'#{av}', sqs_ans_dict[av])
                        except: pass
                    else: sq = sq.replace(f'#{av}', sqs_ans_dict[av])   
            
            cot_content += f'I will be able to get to the complex question\'s answer by answering this question: "{sq}". '

    cot_content += 'Therefore the sequence of sub-questions should be: '
    return cot_content

def give_cot_reference_prev_answers(args, sq, sq_pos, sqs_ans_dict):
    answer_var = re.findall(r'(?:#)(\d+)', sq)
    answer_var = [int(v) for v in answer_var if int(v) < sq_pos+1]
    str_plural = 's' if len(answer_var) > 1 else ''
    addline = ''
    if answer_var:
        addline += f'based on the answer{str_plural} to the previous question{str_plural} '
        for av in answer_var: 
            if sqs_ans_dict is not None: 
                # llm generated decompositions might have answer var errors
                if getattr(args, 'use_dqg_llmgen', False):
                    try:    addline += f'#{av} ({sqs_ans_dict[av]}), '
                    except: addline += f'#{av}, '
                else: addline += f'#{av} ({sqs_ans_dict[av]}), '
            else: addline += f'#{av}, '
    return addline

def give_prefix_decomp_qg(model_name, prompt_version = 2, n_shot = 5, cot = True):

    if model_name in ['mistral', 'gemma']:
        messages = []
    else: 
        messages = [{'role': 'system',
                    'content': '''You are an intelligent assistant that can decompose a complex question into simpler sub-questions.''',},]
    if prompt_version == 1:
        messages += [
        {'role': 'user',
        'content': f'''I will provide you with a single complex multi-hop question. Decompose it into a set of simpler sub-questions so that it will be easier to identify and retrieve the information for answering them instead of the complex question. It is important that the sub-questions are logically connected and that they cover all the necessary information to answer the original complex question. Please provide the sub-questions in a single line, in the order that you would like them to be answered. They should be separated from each other by a ";". When you need to refer to the answer of a preceding sub-question, use a variable (e.g. "#1") for referring to that. A sub-question should not refer to the answer of a sub-question that comes after it. Limit your reply to only the sequence of sub-questions. Start your reply by immediately giving the sub-questions, and stop immediately after; do not say anything else.'''},
        {'role': 'assistant',
        'content': '''I understand the instructions and I will decompose the complex question into simpler sub-questions.'''},]
    
    elif prompt_version == 2:
        if cot: 
            cot_str = 'Start your reply with your reasoning (in less than 50 words), and then immediately give the sequence of sub-questions, and stop immediately after; do not say anything else. ' 
        else: 
            cot_str = 'Start your reply by immediately giving the sequence of sub-questions, and stop immediately after; do not say anything else. '
        
        if n_shot == 0:
            cot_str += '''Limit your reasoning to less than 50 words. It is very important that you prefix the start of each sub-question with a "[SQX]" marker that indicates the position number (X) of the sub-question. i.e. "[SQ1] ... [SQ2] ... [SQ3] ... "; the spans marked "..." in the example is where the sub-questions should be. '''

        messages += [
        {'role': 'user',
        'content': f'''I will provide you with one complex multi-hop question. Decompose it into a set of simpler sub-questions so that it will be easier to identify and retrieve the information for answering them instead of the complex question. It is important that the sub-questions (i) are not ambiguously worded, (ii) are logically connected, and (iii) that they cover all the necessary information and steps to answer the original complex question. It is very important that the decompositions must be as simple as possible, i.e. (i) there should only be as many sub-questions as absolutely necessary; and (ii) each sub-question and its answer should cover only a single atomic fact. Please provide the sub-questions in a single line, in the order that you would like them to be answered. They should be separated from each other by a ";". 
        
        When composing a sub-question, you must ensure that answers of earlier sub-questions are properly referred to whenever possible; when you need to do this, use a variable (e.g. "#1") for referring to the previous answer. It is very important to get the numbering for the answer variable correct. A sub-question should never refer to the answer of a sub-question that comes after it. 
        
        {cot_str}'''},

        {'role': 'assistant',
        'content': '''I understand the instructions and I will decompose the complex question into simpler sub-questions.'''},]


    return messages

def give_suffix():
    pass 

def give_n_shot_examples(dataset, n_shot, qa_task = False, legacy = True):
    if dataset in ['hotpotqa_fullwiki', 'hotpotqa_distractor']:
        dataset_key = 'musique'
    elif dataset == 'breakhigh' and qa_task == True:
        dataset_key = 'breakhigh_hotpotqa'
    else: dataset_key = dataset
    
    examples = [c for c in PROMPT_CANDS[dataset_key] if len(c['sq_list']) > 1]
    
    if dataset == 'breakhigh':
        if legacy: # used in the roundtrip_filtering step for the paper
            examples = [c for c in PROMPT_CANDS[dataset_key] if 'HOTPOT_' in c['id']]
        else: 
            examples = [c for c in examples if 'HOTPOT_' in c['id']]
    examples = examples[:n_shot]
    
    return examples

def save_out_decomp_qg_outputs(args, inference_data, gold_dict):
    c1 = args.task in ['decomp_qg']
    c2 = args.task in ['rerank_dqg_pairwise', 'rerank_dqg_listwise']
    # write a test_out.txt file (for use in evaluate_decomp_qg)
    test_out_fp = os.path.join(args.save_path, 'test_out.txt')
    if c2: args.asst_prompt = args.user_prompt = None
    with open(test_out_fp, mode = 'w+', encoding = 'utf-8') as f:
        for idx, result_instance in inference_data.items():
            gold_idx = re.sub(r'_perm(\d)+', '', idx)
            if c1: 
                src = result_instance['text']['qs'][0]
                tgt = result_instance['gold_subquestions'].strip()
                pred = result_instance['pred_subquestions'].strip()
            elif c2:
                src = gold_dict[gold_idx]['question'].strip()
                tgt = gold_dict[gold_idx]['decomposition'].strip()
                r = result_instance['rerank_products']
                ranks = re.findall(r'(?:\[)(\w+)(?:\]?)', r['response'])
                top_ranked = ranks[0]
                pred = result_instance['docid_map'][top_ranked][1]['text']
            # remove newlines, otherwise throws off prepare_for_dqg_eval lines
            for elem in [idx, src, tgt, pred]:
                elem = elem.replace('\n', ' ').replace('\t', ' ')
            f.write(f'{idx}\t{src}\t{tgt}\t{pred}\n')
    
    import sys
    sys.path.append('..')
    from evaluation.evaluate_dqg import prepare_for_dqg_eval
    df_holder = prepare_for_dqg_eval(args, hpe = False)
    
    for src_type, df in df_holder.items():
        df_label, df_pred = df['label'], df['pred']

        fp = os.path.join(args.save_path, f'{src_type}_labels.csv')
        df_label.to_csv(fp, index = False)
        
        fp = os.path.join(args.save_path, f'{src_type}_predictions.csv')
        df_pred.to_csv(fp, index = False)
        print('\t\t🔥 Decomp QG prep for evaluation complete, saved to:', args.save_path)

def strip_start_trailing_whitespace(line):
    return 

def save_out_calibration_data(args, inference_data):
    if args.use_ranking_calibration['prod_cal_scores']:
        calibration_savepath = os.path.join(args.calibration_savepath, f'perm_order_scores_{args.num_cands}cands')
        if not os.path.exists(calibration_savepath): os.makedirs(calibration_savepath)

        holder_settings = {'contraints_pred_seq':       args.contraints_pred_seq,  
                           'pred_seq_num_pos':          args.pred_seq_num_pos,
                           'pred_seq_num_idxes':        args.pred_seq_num_idxes,
                           'rank_pred_seq':             args.rank_pred_seq,
                           'rank_pred_seq_tokens':      args.rank_pred_seq_tokens, 
                           'rank_pred_seq_tokens_dec':  args.rank_pred_seq_tokens_dec,
                           'tokenizer_vocab':           args.tokenizer.vocab}
        fname = f'perm_order_scores_for_calibration_{args.num_cands}cands_settings.json'
        with open(os.path.join(calibration_savepath, fname), 'w+') as f:
            json.dump(holder_settings, f)
        
        print('\t\t🔥 Start saving calibration data...')
        for __, (idx, result_instance) in enumerate(tqdm.tqdm(inference_data.copy().items())):
            holder = {}
            if type(result_instance) != dict: 
                print(f'\t\t🚨{idx} does not have perm_order, not able to proceed in saving it')
                inference_data.pop(idx)
                continue
            
            holder['idx']        = idx
            holder['perm_order'] = result_instance['rerank_products']['perm_order']
            holder['scores']     = result_instance['rerank_products']['scores']
            holder['prompt_repr']= result_instance['rerank_products']['prompt_repr']
            
            if 'scores' in result_instance['rerank_products']: # remove (not serializable)
                result_instance['rerank_products'].pop('scores') 
        
            fname = f'perm_order_scores_for_calibration_{args.num_cands}cands_idx{idx}.pt'
            save_path = os.path.join(calibration_savepath, fname)
            torch.save(holder, save_path)
        print(f'\t\t🔥 Calibration data saved to: {calibration_savepath}')
