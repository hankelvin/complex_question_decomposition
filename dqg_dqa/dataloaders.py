import random, json, re, torch, os
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

def prep_data(args, tokenizer):
    if args.padding_side == 'left':
        
        from llm_inference.task_utils_decomp_qg import prepare_decomp_qg_prompt
        from llm_inference.task_utils_qa import prepare_qa_prompt
        from llm_inference.chat_class_inference_dataloaders import load_decomp_qg_llm_data
    
    if args.chat_model:
        dp = 'data'
    else: 
        dp = 'data/01_unified_format/'
    if 'zeroshotCoT' in args.task: pos = 2
    else: pos = 1
    datasets = [args.task.split('_')[pos]]
    # hotpotqa has a suffix "_fullwiki"
    datasets = [ds if ds != 'hotpotqa' else 'hotpotqa_fullwiki' for ds in datasets]
    original_input_file = '{}/UNIFIED_{}_{}.jsonl'
    
    dataholder = defaultdict(list)
    SPLITS = ['validation', 'test'] if args.c_dqg_hotpotqa_bypass else ['validation', 'train', 'test']
    if args.roundtrip_filtering_qa: SPLITS = ['train'] 
    if args.test_only == 'phase3_val_as_test': SPLITS = ['validation'] 
    
    
    for dataset in datasets:
        other_dataset = None
        if args.cross_domain != False: other_dataset = args.cross_domain

        for split in SPLITS:    

            c_dqg_llmgen    = args.use_dqg_llmgen   is not None and split == 'train'
            c_dqg_llmgen_qa = args.do_dqg_llmgen_qa is not None and split == 'validation'
            
            if  args.padding_side == 'right':
                if args.qa_task or c_dqg_llmgen or c_dqg_llmgen_qa: raise NotImplementedError

                dataholder_split = do_one_dataset_split(original_input_file, dp, dataset, split, other_dataset)      
            
            elif args.padding_side == 'left':
                
                # load replacement DQGs 
                if c_dqg_llmgen:    
                    dp_llmgen = 'results/decomp_qg'
                    df_llmgen = give_dqg_llmgen(args, dp_llmgen, dataset, split)
                elif c_dqg_llmgen_qa:               
                    dp_llmgen = args.dqg_llmgen_qa_validation_dp[args.do_dqg_llmgen_qa]
                    hotpot_holder = give_hotpot_holder(split_to_use = 'validation' \
                                        if split in ['test', 'validation'] else 'train')
                    df_llmgen = give_dqg_llmgen(args, dp_llmgen, dataset, split)
                 
                dataholder_split = []
                args.input_file = [dataset, split]
                decomp_qg_requests = load_decomp_qg_llm_data(args, dp, bypass_do_trial = True)

                PICKS = random.sample(list(range(len(decomp_qg_requests))), 3)   
                ORIGINAL_SIZE = len(decomp_qg_requests)
                for i, (idx, line) in enumerate(decomp_qg_requests.items()):
                    if args.do_trial and len(dataholder_split) >= 10: continue

                    c1 = dataset        == 'breakhigh' and not  args.cross_domain
                    c2 = other_dataset  == 'breakhigh' and      args.cross_domain 
                    c3 = split in ['test'] or (split == 'validation' and args.test_only == 'phase3_val_as_test')
                    c_breakhigh = c1 or (c2 and c3)
                    if c_dqg_llmgen_qa and c_breakhigh:
                        if 'HOTPOT_' not in idx: continue
                        hp_idx = re.search(r'(?:HOTPOT_train_|HOTPOT_dev_)(.+)', idx)
                        hp_oinfo = hotpot_holder[hp_idx.group(1)]['original_info']
                        line['original_info'].update(hp_oinfo)

                    if c_dqg_llmgen or c_dqg_llmgen_qa:
                        
                        if args.use_dqg_rtfilt and idx not in df_llmgen: continue
                        
                        # replace the decomposition
                        new_sqs_list    = df_llmgen[idx]['decomposition']

                        if i < 2: print('\t🟧CHECK use_dqg_llmgen 1.1 (BEFORE)', line['decomp_qs']['text']['qs_var'])
                        line['decomp_qs']['text']['qs_var'] = [sq.strip() for sq in new_sqs_list.split(' ;')]
                        if i < 2: print('\t🟧CHECK use_dqg_llmgen 1.2 (AFTER)', line['decomp_qs']['text']['qs_var'])

                        if c_breakhigh:
                            new_operators   = df_llmgen[idx]['operators_gen']
                            if type(new_operators) == str: 
                                new_operators = eval(new_operators)
                                new_operators = [o[1:] if o.startswith('[') else o for o in new_operators]
                                new_operators = [o[:-1] if o.endswith(']') else o for o in new_operators]

                            if i < 2: print('\t🟧CHECK use_dqg_llmgen 2.1 (BEFORE)', line['operators'])
                            line['operators'] = new_operators
                            if i < 2: print('\t🟧CHECK use_dqg_llmgen 2.1 (AFTER)', line['operators'])

                    entry = {}
                    sq_holder = None
                    if args.qa_task:
                        cq_messages, sq_messages, sq_messages_successive, tgt, cq = prepare_qa_prompt(args, line, split = split)
                        
                        # a. prepare prompt for CQ run 
                        cqm_prompt  = tokenizer.apply_chat_template(cq_messages, tokenize = False, add_generation_prompt = True)  
                        entry['src'] = tokenizer.encode(cqm_prompt)

                        # b. prepare prompt for SQ run 
                        sqm_prompt  = tokenizer.apply_chat_template(sq_messages, tokenize = False, add_generation_prompt = True)  
                        entry['src_sq_str'] = sqm_prompt

                        # c. prepare prompt for SQ run (successive)
                        # see "##### Collating the SQs and their answers #####" in prepare_qa_prompt()
                        assert len(sq_messages_successive) % 2 == 0, "there should be an even number of sq_messages (user, asst)"
                        num_turns = len(sq_messages_successive)//2
                        sq_holder = []
                        
                        for nt in range(num_turns):
                            # remove unnecessary system prompt in llama
                            turn = sq_messages_successive[nt*2: (nt*2)+2]
                            turn_prompt  = tokenizer.apply_chat_template(turn, tokenize = False, add_generation_prompt = False)
                            if args.model_name in ['llama','nvidia_llama3']:
                                check_remove = 'Cutting Knowledge Date:'
                                if check_remove in turn_prompt:
                                    turn_prompt = re.search(r"<\|eot_id\|>(.*)", turn_prompt, re.DOTALL).group(1)    
                            
                            elif args.model_name in ['qwen']:    
                                check_remove = 'created by Alibaba Cloud'
                                if check_remove in turn_prompt:                        
                                    turn_prompt = re.search(r"<\|im_end\|>(.*)", turn_prompt, re.DOTALL).group(1).lstrip()    
                            
                            # NOTE: the asst line should not have end of turn, so that the answer can be generated
                            # args.eot_map = {'llama': '<|eot_id|>', 'qwen': '<|im_end|>', 'phi3': '<|end|>\n<|endoftext|>', 'gemma': '<end_of_turn>'}
                            eot = re.escape(args.eot_map[args.model_name])
                            if eot in turn_prompt: # get everything up to the last eot
                                turn_prompt = re.search(rf'(.+)(?:{eot})', turn_prompt, re.DOTALL).group(1)
                            
                            sq_holder.append(turn_prompt) 
                        
                        entry['sq_holder']  = sq_holder

                        if c_dqg_llmgen: 
                            tgt = '[NONE]'
                        if c_dqg_llmgen_qa and c_breakhigh: 
                            tgt = give_hotpot_ans(idx.split('_')[-1], hotpot_holder)

                    else: 
                        messages, tgt, cq = prepare_decomp_qg_prompt(args, line, repro = False, split = split)
                    
                        # a. prepare prompt and run inference
                        prompt  = tokenizer.apply_chat_template(messages, tokenize = False, 
                                                                add_generation_prompt = True)  
                        entry['src'] = tokenizer.encode(prompt)

                    # ensure eos token
                    tgt_enc = tokenizer.encode(tgt)
                    if tokenizer.eos_token_id != tgt_enc[-1]: 
                        tgt_enc.append(tokenizer.eos_token_id)

                    # for llama 3 class, ensure no bos (esp to match COT)
                    if args.model_name in ['llama','nvidia_llama3', 'qwen']:
                        if tokenizer.bos_token_id == tgt_enc[0]:
                            tgt_enc = tgt_enc[1:]

                    entry['tgt']        = tgt_enc
                    entry['cq']         = tokenizer.encode(cq)
                    entry['id']         = idx
                    entry['id_enc']     = tokenizer.encode(idx)
                    
                    dataholder_split.append(entry)
            
                    if i in PICKS or args.do_trial: 
                        print('^'*100)                
                        print(f'\t\t{split} {i} SRC: ', tokenizer.decode(entry['src']))
                        print(f'\t\t{split} {i} TGT: ', tokenizer.decode(entry['tgt']))
                        if entry.get('src_sq', None) is not None:
                            print('\t\t\t', '^'*20)
                            print(f'\t\t{split} {i} SRC (SQ): ', tokenizer.decode(entry['src_sq']))
                            print('\t\t\t', '^'*20)
                        if entry.get('sq_holder', None) is not None:
                            entry['sq_holder']
                            print('\t\t\t', '^'*20)
                            for sqsucc in sq_holder:
                                print(f'\t\t{split} {i} SRC (SQ-SUCCESSIVE): ', sqsucc)
                            print('\t\t\t', '^'*20)
                        print('^'*100)                                        
####
########
############
            c_do_upsamp = args.use_dqg_rtfilt and args.use_dqg_rtfilt.endswith('upsamp')
            if c_do_upsamp and args.use_dqg_rtfilt and len(dataholder_split) < ORIGINAL_SIZE and not args.do_trial:
                print(f'🟧RT filtering leading to {split} split having fewer than {ORIGINAL_SIZE} entries...')
                print('🟧Currently at:', len(dataholder_split))
                short = ORIGINAL_SIZE - len(dataholder_split)
                dataholder_split += random.choices(dataholder_split, k = short)
                print('🟧Now at:', len(dataholder_split))
            
            dataholder[split].extend(dataholder_split)
    
    # 1. filter, add prefixes, encode
    if not args.chat_model:
        dataholder = filter_add_prefixes_encode(args, dataholder, tokenizer)
    else: pass

    print_func = getattr(getattr(args, 'logger', False), 'print', print)

    assert tokenizer.pad_token_id is not None
    pad_token_id = tokenizer.pad_token_id

    collate_obj      = Collate(pad_token_id = pad_token_id, task = args.task,
                               mask_value = getattr(args, 'mask_value', 0), 
                               padding_side = args.padding_side, )
    if args.chat_model:
        # https://github.com/huggingface/transformers/issues/31672
        assert args.padding_side == 'left'
        collate_obj         = Collate(pad_token_id = pad_token_id, task = args.task,
                                      mask_value = getattr(args, 'mask_value', 0), 
                                      padding_side = 'right', )
        collate_obj_eval    = Collate(pad_token_id = pad_token_id, task = args.task,
                                      mask_value = getattr(args, 'mask_value', 0), 
                                      padding_side = args.padding_side, )
    else: 
        collate_obj         = Collate(pad_token_id = pad_token_id, task = args.task,
                                      mask_value = getattr(args, 'mask_value', 0), 
                                      padding_side = args.padding_side, )
        collate_obj_eval    = collate_obj


    train_dataloader, val_dataloader, test_dataloader = \
        give_dataloaders(args, dataholder, print_func, collate_obj, collate_obj_eval)

    return train_dataloader, val_dataloader, test_dataloader, collate_obj, collate_obj_eval

def give_hotpot_holder(split_to_use):
    if split_to_use in ['train', 'test']:        split_str = f'fullwiki_{split_to_use}'
    elif split_to_use == 'validation': split_str = f'distractor_{split_to_use}'
    fp = f'data/01_unified_format/UNIFIED_hotpotqa_{split_str}.jsonl'
    hotpot_holder = {}
    with open(fp, encoding = 'utf-8') as f: 
        for line in f: 
            line = json.loads(line)
            hotpot_holder[line['id']] = line
    return hotpot_holder

def give_hotpot_ans(idx, hotpot_holder):
    answer = hotpot_holder.get(idx, None)['text']['as']
    assert answer is not None
    assert len(answer) >= 1, (idx, answer)
    return answer[-1]

def give_dqg_llmgen(args, dp_llmgen, dataset, split):
    import sys
    sys.path.append('..')
    from evaluation.evaluate_break_suite import remove_special_tokens_lingering_operators

    winner_str = '' 
    if args.use_dqg_llmgen and args.use_dqg_rtfilt:
        
        key = args.use_dqg_llmgen

        scores_holder = {}
        idxes_holder  = {}
        crt_1x = args.use_dqg_rtfilt and args.use_dqg_rtfilt.startswith('rtfilt-1x')
        crt_2x = args.use_dqg_rtfilt and args.use_dqg_rtfilt.startswith('rtfilt-2x')
        if   crt_1x:    winner_strs = ['_winner0'] 
        elif crt_2x :   winner_strs = ['_winner0', '_winner1'] 
        else: raise ValueError

        for winner_str in winner_strs:
            fp_rt_scores = args.dqg_llmgen_roundtrip_map[key][args.task].format(winner_str)
            with open(os.path.join(fp_rt_scores, 'test_scores.json'), encoding='utf-8') as f: __  = json.load(f)
            metric = 'test_tokenf1_allscores'
            scores_holder[winner_str]   = __[metric]
            idxes_holder [winner_str]   = __['idxes']
            assert len(scores_holder[winner_str]) == len(idxes_holder [winner_str]), \
                    (len(scores_holder[winner_str]), len(idxes_holder[winner_str]))
        # ensure orders are the same
        if len(idxes_holder) == 2 and len(idxes_holder['_winner0']) != len(idxes_holder['_winner1']):
            # reorder to idx order of winner0
            scores_holder['_winner1'] = [scores_holder['_winner1'][idxes_holder['_winner1'].index(i)] \
                                         for i in idxes_holder['_winner0']]
            idxes_holder['_winner1']  = idxes_holder['_winner0']
        elif len(idxes_holder) > 2: raise NotImplementedError
        
        keep = {}
        for i, idx in enumerate(idxes_holder['_winner0']):
            w0 = scores_holder['_winner0'][i]
            
            if  crt_1x:
                if w0 >= args.rtfilt_cutoff: keep[idx] = '_winner0'
            elif crt_2x:
                w1 = scores_holder['_winner1'][i]
                if w0 >= args.rtfilt_cutoff or w1 >= args.rtfilt_cutoff:
                    if   w0 >= w1:  keep[idx] = '_winner0'
                    elif w1 > w0:   keep[idx] = '_winner1'

        llmgen_holder = {}
        for winner_str in winner_strs:
            fp_llmgen = args.dqg_llmgen_mapping[key]
            fp_llmgen = os.path.join(dp_llmgen, 
                    fp_llmgen.format(dataset, split, winner_str, args.model_name), 
                    'text_predictions.csv')
            c_CoTnshot0 = '_CoTnshot0' in fp_llmgen
            if args.do_dqg_llmgen_qa:
                __df_llmgen, __1 = remove_special_tokens_lingering_operators(fp_llmgen, cotnshot0 = c_CoTnshot0)
            else: 
                __df_llmgen = pd.read_csv(fp_llmgen)
            if 'question_id' not in __df_llmgen.columns:
                if args.do_dqg_llmgen_qa:
                    __, __1 = remove_special_tokens_lingering_operators(fp_llmgen.replace('predictions', 'labels'),
                                                                        cotnshot0 = c_CoTnshot0)
                else: 
                    __ = pd.read_csv(fp_llmgen.replace('predictions', 'labels'))
                assert len(__df_llmgen) == len(__)
                __df_llmgen['question_id'] = __['question_id'].copy()

            llmgen_holder[winner_str] = {x['question_id']: x for x in __df_llmgen.to_dict('records')}
        
        df_llmgen = {idx: llmgen_holder[winner_str][idx] for idx, winner_str in keep.items()}
        print(f'ORIGINAL SIZE: {len(llmgen_holder[winner_str])}... KEEP SIZE: {len(df_llmgen)}')

    else: 
        if   args.use_dqg_llmgen:   
            key = args.use_dqg_llmgen
            fp_llmgen = args.dqg_llmgen_mapping[key]
        elif args.do_dqg_llmgen_qa is not None: 
            key = args.do_dqg_llmgen_qa
            fp_llmgen = args.dqg_llmgen_qa_validation[key]
    
        if args.roundtrip_filtering_qa and args.qa_task:
            winner_str = re.search(r'_winner\d+', args.test_only).group()
        
        if winner_str == '': winner_str = '_winner0'
        if 'llm_single' in key:                     val_n3 = args.model_name 
        elif 'llm_top1_sft_crossdomain' in key:     val_n3 = args.cross_domain
        else:                                       val_n3 = ''
        fp_llmgen = fp_llmgen.format(dataset, split, winner_str, val_n3)
        fp_llmgen = os.path.join(dp_llmgen, fp_llmgen, 'text_predictions.csv')
        c_CoTnshot0 = '_CoTnshot0' in fp_llmgen
        if args.do_dqg_llmgen_qa:
            df_llmgen, __1 = remove_special_tokens_lingering_operators(fp_llmgen, cotnshot0 = c_CoTnshot0)
        else: 
            df_llmgen = pd.read_csv(fp_llmgen)
        
        if 'question_id' not in df_llmgen.columns:
            if args.do_dqg_llmgen_qa:
                __, __1 = remove_special_tokens_lingering_operators(fp_llmgen.replace('predictions', 'labels'),
                                                                    cotnshot0 = c_CoTnshot0)
            else: 
                __ = pd.read_csv(fp_llmgen.replace('predictions', 'labels'))
            assert len(df_llmgen) == len(__)
            df_llmgen['question_id'] = __['question_id'].copy()

        df_llmgen = {x['question_id']: x for x in df_llmgen.to_dict('records')}

    return df_llmgen

def do_one_dataset_split(original_input_file, dp, dataset, split, other_dataset):
    dataholder_split = []
    if other_dataset is not None and split in ['test', 'validation']:
        dataset = other_dataset 
        print('\t🟥CROSS-DOMAIN Using {} for {} split'.format(other_dataset, split))
    else: 
        dataset = dataset
    fp = original_input_file.format(dp, dataset, split)
    with open(fp, encoding = 'utf-8') as f: 
        for l in f:
            line = json.loads(l)
            if len(re.findall('_', dataset)) > 0: # e.g. 'hotpotqa_fullwiki' 
                line['dataset'], line['split'] = dataset.split('_')[0], split
            else: line['dataset'], line['split'] = dataset, split

            if dataset in ['breakhigh'] and split != 'test':
                try: line['operators'] = eval(line['original_info']['operators'])
                except: 
                    print(line['original_info']['operators'])
                    raise ValueError

            dataholder_split.append(line)
        
    # handle unrealised relations in musique (in sq_originals)
    if dataset == 'musique' and split not in ['test']:
        dataholder_split = realise_musique_predicates(dataholder_split)

    return dataholder_split


def filter_add_prefixes_encode(args, dataholder, tokenizer):
    qpos_tokens = tokenizer.qpos_tokens
    task_prompt = tokenizer.encode(args.src_prompt, add_special_tokens = False)
    
    SPLITS =  ['test', 'validation'] if args.c_dqg_hotpotqa_bypass else ['train', 'validation', 'test']
    for split in SPLITS:
        PICKS = random.sample(list(range(len(dataholder[split]))), 3)
        for i, line in enumerate(dataholder[split]):
            cq_text    = line['text']['qs']
            assert type(cq_text) == list and len(cq_text) == 1
            cq_text = cq_text[0]

            line['src']     = task_prompt + tokenizer.encode(cq_text)
            line['id_enc']  = tokenizer.encode(line['id'])
            
            # for phase3, encode the SQ sequence (with qpos tokens) and task prefix
            sqs = line['decomp_qs']['text']['qs_var']
            if sqs: # in breakhigh, there is no decomposition in test set
                if line.get('dataset', None) == 'breakhigh':
                    operators = [f'[{o.lower()}]' for o in line.get('operators', '')]
                    sq_sequence = ''.join([qpos_tokens[i] + operators[i] + q for i, q in enumerate(sqs)])
                else: 
                    sq_sequence = ''.join([qpos_tokens[i] + q for i,q in enumerate(sqs)])
            else: sq_sequence = ''
            
            line['tgt'] = tokenizer.encode(sq_sequence)
            
            # inspect
            if i in PICKS:                 
                print(f'\t\t{split} {i} SRC: ', tokenizer.decode(line['src']))
                print(f'\t\t{split} {i} TGT: ', tokenizer.decode(line['tgt']))

    return dataholder

def realise_musique_predicates(holder, pattern_forward = re.compile(r'>>')):
    dp = 'data/02_support_data/'
    with open(f'{dp}/03b_musique_relation_mapping.json', encoding = 'utf-8') as f:
        templates = json.load(f)

    for line in holder:
        try: qs_var = line['decomp_qs']['text']['qs_var']
        except: 
            print(line)
            raise ValueError
        for i, q in enumerate(qs_var):
            if re.search(pattern_forward, q):
                e, r = q.split('>>')
                e = e.strip()
                r = r.strip()
                t = random.choice(templates[r])
                line['decomp_qs']['text']['qs_var'][i] = t.replace('#X', e)
    
    return holder


def give_dataloaders(args, dataholder, print_func, collate_obj, collate_obj_eval):
    train_dataloader = None
    if not (args.c_dqg_hotpotqa_bypass or args.test_only):
        random.shuffle(dataholder['train'])
        shuffle = False if args.roundtrip_filtering_qa else True
        train_dataloader = DataLoader(dataholder['train'], batch_size = args.bsz_train, 
                                    shuffle = shuffle, collate_fn = collate_obj.collate_fn, num_workers = 4,
                                    persistent_workers = True)

        print_func('TRAIN BSZ:', args.bsz_train, 'TRAIN SIZE', len(dataholder['train']), 
                'TRAIN #BATCH', len(train_dataloader))
        
    val_dataloader = DataLoader(dataholder['validation'], batch_size = args.bsz_test, 
                                shuffle = False, collate_fn = collate_obj_eval.collate_fn, num_workers = 4, 
                                persistent_workers = True)
    
    # NOTE: no test set for phase3; therefore during training use val set as test set
    if args.test_only == 'phase3_val_as_test':
        test_dataloader = val_dataloader
    elif args.roundtrip_filtering_qa and args.do_dqg_llmgen_qa is None:
        test_dataloader = DataLoader(dataholder['train'], batch_size = args.bsz_test, 
                                    shuffle = False, collate_fn = collate_obj_eval.collate_fn, num_workers = 4,
                                    persistent_workers = True)
    else: test_dataloader = DataLoader(dataholder['test'], batch_size = args.bsz_test, 
                                shuffle = False, collate_fn = collate_obj_eval.collate_fn, 
                                num_workers = 4, persistent_workers = True)
    
    if len(dataholder['validation']):
        print_func('VALID BSZ:', args.bsz_test, 'VALID SIZE', len(dataholder['validation']), 
                'VALID #BATCH',len(val_dataloader))
    if len(dataholder['test']) or args.test_only == 'phase3_val_as_test': 
        print_func('TEST BSZ:', args.bsz_test, 'TEST SIZE', len(dataholder['test']), 
                'TEST #BATCH', len(test_dataloader))
    
    return train_dataloader, val_dataloader, test_dataloader


class Collate:
    '''
    Custom collate function for dataloading in this script.
    '''
    def __init__(self, pad_token_id = None, 
                 mask_value = 0, task = '',
                 padding_side = 'right'):
        assert task, 'task must be provided'
        self.task           = task
        self.pad_token_id   = pad_token_id
        self.padding_side   = padding_side
        assert mask_value in [0,1]
        self.mask_value     = mask_value

    def pad2max(self, batch_elem, make_mask = False, pad_src = False):
        '''
        helper function to pad all of a batch element to the max length within
        '''
        pad_value = self.pad_token_id
        max_len = max([len(s) for s in batch_elem])
        new_batch_elem, batch_elem_mask = [], []
        for s in batch_elem: 
            orig_len, pad_size = len(s), max_len - len(s)
            
            if (pad_src and self.padding_side == 'right' ) or not pad_src: 
                s.extend([pad_value for __ in range(pad_size)])
            else: 
                s = [pad_value for __ in range(pad_size)] + s
                
            new_batch_elem.append(s)

            if make_mask:
                omv = 0 if self.mask_value == 1 else 1
                if (pad_src and self.padding_side == 'right') or not pad_src: 
                    mask = [omv for __ in range(orig_len)] + \
                            [self.mask_value for __ in range(pad_size)]
                else: 
                    mask = [self.mask_value for __ in range(pad_size)] + \
                            [omv for __ in range(orig_len)]
                    
                batch_elem_mask.append(mask)

        return new_batch_elem, batch_elem_mask 

    def collate_fn(self, batch):
        # 1. prepare tgt        
        tgt = [entry['tgt'] for entry in batch]
        tgt, tgt_mask = self.pad2max(tgt, make_mask = True)
        tgt         = torch.LongTensor(tgt)
        tgt_mask    = torch.LongTensor(tgt_mask)
        
        # 2. prepare src 
        src, src_mask = [entry['src'] for entry in batch], []
        # pad to maxlen in batch, also create mask
        src, src_mask = self.pad2max(src, make_mask = True, pad_src = True)
        src              = torch.LongTensor(src)
        src_mask         = torch.LongTensor(src_mask)

        # 3. CQ
        cq = [entry['cq'] for entry in batch]
        cq, __ = self.pad2max([entry['cq'] for entry in batch], make_mask = False)
        cq         = torch.LongTensor(cq)
        
        # 4. idx
        id_enc, __ = self.pad2max([entry['id_enc'] for entry in batch], make_mask = False)
        id_enc = torch.LongTensor(id_enc)

        sq_holder   = None
        src_sq_str  = None
        if self.task.startswith('dqa_') or self.task.startswith('cqadqa_'):
            # leave as strings for easy appending to successive sq_sequence (str)
            src_sq_str = [entry['src_sq_str'] for entry in batch]

            sq_holder = [] # for instances with lesser SQs (turns), pad with None
            max_turns = max([len(entry['sq_holder']) for entry in batch])
            for entry in batch: 
                short = max_turns - len(entry['sq_holder'])
                sq_holder.append(entry['sq_holder'] + [None] * short)
        
        return src, src_mask, tgt, tgt_mask, cq, src_sq_str, sq_holder, id_enc