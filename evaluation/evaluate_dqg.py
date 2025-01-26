import os, re
import pandas as pd


OPERATORS = ['[aggregate]', '[arithmetic]', '[boolean]', '[comparative]', 
 '[comparison]', '[discard]', '[filter]', '[group]', '[intersection]', '[list]',
 '[none]', '[project]', '[select]', '[sort]', '[superlative]', '[union]']

REGEX_OPERATORS = re.compile('|'.join([re.escape(o) for o in OPERATORS]))

def prepare_for_dqg_eval(args, not_chat = False, lower = True):
    '''
    labels.csv format: question_id	question_text	decomposition
    predictions.csv format: decomposition
    '''
    save_path = args.save_path
    if not_chat:
        src_prompt = args.src_prompt
        tgt_prompt = None
        cot        = False
    else: 
        src_prompt = args.user_prompt
        tgt_prompt = args.asst_prompt
        cot        = args.decomp_qg_args.get('cot', False)

    # 1. load predictions 
    with open(os.path.join(save_path, 'test_out.txt'), encoding = 'utf-8') as f:
        eval_results = f.readlines()
        eval_results = [x.strip().split('\t') for x in eval_results]

        l_TEXT = {}
        for l in eval_results:
            if not l: continue
            
            try: idx, cq, tgt, gen = l
            except: 
                print('Error:', l)
                l_TEXT[idx] = {'cq': '', 'tgt': '', 'gen': '', 
                'operators_gen': [], 'operators_tgt': []}
            
            if src_prompt is not None: 
                cq = cq.replace(src_prompt, '').strip()
                
            if not cot and tgt_prompt is not None: 
                tgt = tgt.replace(tgt_prompt, '').strip()
                gen = gen.replace(tgt_prompt, '').strip()
            else:
                # NOTE: for cot, there is a whole prefix of CoT that needs to be removed
                # it will stop just before the first [SQ] token
                try:    gen = re.search(r'\[SQ1\].+', gen).group()
                except: print('Error (CoT gen):', idx, gen)
                try:    tgt = re.search(r'\[SQ1\].+', tgt).group()
                except: print('Error (CoT tgt):', idx, tgt)                    
                        
            gen, ops_gen = get_decomposition(gen)
            tgt, ops_tgt = get_decomposition(tgt)
            
            l_TEXT[idx] = {'cq': cq, 'tgt': tgt, 'gen': gen, 
            'operators_gen': ops_gen, 'operators_tgt': ops_tgt}

    # write to file
    df_holder = {}
    for src_type, coll in [('text', l_TEXT)]:
        if not coll: continue
        
        label_dict = {'question_id': [], 'question_text': [], 'decomposition': [], 
                      'operators_tgt': []}
        pred_dict = {'decomposition': [], 'operators_gen': [], }
        for idx, line in coll.items():
            # a. collect label info
            label_dict['question_id'].append(idx)
            label_dict['question_text'].append(line['cq'].lower()  if lower else line['cq'])
            label_dict['decomposition'].append(line['tgt'].lower() if lower else line['tgt'] )
            label_dict['operators_tgt'].append(line['operators_tgt'])
            
            # b. collect prediction info
            pred_dict['decomposition'].append(line['gen'].lower()  if lower else line['gen'])
            pred_dict['operators_gen'].append(line['operators_gen'])
                
        # i. write labels.csv
        df_label = pd.DataFrame.from_dict(label_dict)
            
        # ii. write predictions.csv
        df_pred = pd.DataFrame.from_dict(pred_dict)                                              
        # NOTE: some models (llama 3.1 70B 4-bit) do not follow instructions fully
        # and generate further CoT
        df_pred['decomposition'] = df_pred['decomposition'].apply(lambda x: \
                    x.split('therefore the sequence of sub-questions should be: ')[-1].strip() \
                    if len(x.split('therefore the sequence of sub-questions should be: ')) > 1 else x)

        df_holder[src_type] = {'label': df_label, 'pred': df_pred}

    return df_holder

def get_decomposition(seq):

    # recover, and remove operators (for BREAKHIGH format)
    ops = re.findall(REGEX_OPERATORS, seq)
    for o in set(ops): seq = seq.replace(o, '')
    
    # 1. strip task tokens 
    if 'task:' in seq:
        seq = seq.split('task:')[1].strip()
    else: seq = seq.strip()

    seq = re.sub(r'\[SQ[0-9]\]', '[SQ]', seq)
    seq = re.sub(r'\[SQ1[0-9]\]', '[SQ]', seq)
    seq_list = seq.split('[SQ]')
    seq_list = [x for x in seq_list if x.strip()]

    return ' ;'.join(seq_list), ops
