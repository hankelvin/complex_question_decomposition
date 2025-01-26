import json, tqdm, re, random, copy
from collections import defaultdict    
import pandas as pd
import sys
sys.path.append('../tools/truecaser')
from transformers import FSMTTokenizer
FSMT_TOK = FSMTTokenizer.from_pretrained('facebook/wmt19-en-ru')
from PredictTruecaser import give_truecase_objs, predict_truecase 
distri_obj      = '../tools/truecaser/distributions.obj'
TRUECASE_OBJS   = give_truecase_objs(distri_obj)

'''
Functions to prepare various datasets by placing them into a single unified data format. 
output is a jsonl file for each dataset where the original splits are maintained.
has the following structure: 

TRAIN 
{'id': ..., 
'text': {'contexts': [.., .., ..], 'qs': [..., ] 'as': [...,] },
'original_data': {}}
'''

def format_hotpotqa():
    
    holder = {}
    splits = ['test', 'train', 'validation']
    for split in splits:
        holder[split] = []
        if split == 'test':         variant, v_str = '_fullwiki', 'v1'
        elif split == 'validation': variant, v_str = '_distractor', 'v1'
        else:                       variant, v_str = '', 'v1.1'
        split_str = 'dev' if split == 'validation' else split
        with open(f'hotpotqa/hotpot_{split_str}{variant}_{v_str}.json', encoding = 'utf-8') as f:
            hotpotqa_split = json.load(f)
        for __, row in enumerate(tqdm.tqdm(hotpotqa_split)):
            idx = row.pop('_id')
            q   = row.pop('question')
            a   = row.pop('answer', None)

            if split != 'test': assert q and a, ('HotpotQA fail:', idx, q,a)

            line = {'id'  : idx,
                    'text': {'contexts': None, 
                             'qs': [q], 'as': [a]},
                    'num_hops': 2,
                    'original_info': row}
            
            holder[split].append(line)
    
    return holder
        
# 2. MuSiQue
def format_musique():    
    holder = {}
    for split in ['train', 'validation', 'test']:
        ss = 'dev' if split == 'validation' else split
        with open(f'musique/musique_ans_v1.0_{ss}.jsonl', 
              encoding = 'utf-8') as f: lines = [json.loads(l) for l in f]
        
        with open('02_support_data/03b_musique_relation_mapping.json', encoding = 'utf-8') as f:
            relations_mapping = json.load(f)

        holder[split] = []
        for __, qrow in enumerate(tqdm.tqdm(lines)):
            idx     = qrow.pop('id')
            q       = qrow.pop('question').strip()
            assert q

            q = make_q_wellformed(q)

            a = None
            if qrow.get('answer', None):
                a = qrow.pop('answer')
            if split != 'test': assert a
            
            decomp_qs_var = decomp_qs_filled = decomp_ans = None
            if split != 'test':
                decomp_qs_var, decomp_qs_filled, decomp_ans = [], [], []
                for sqrow in qrow['question_decomposition']:
                    
                    sq = sqrow['question']
                    if ' >> ' in sq: 
                        sbj, rel = sq.split(' >> ')
                        sq_template = relations_mapping[rel][0]
                        sq          = sq_template.replace('#X', sbj)
                    decomp_qs_var.append(re.sub('\s+', ' ', sq))
                    decomp_ans.append(re.sub('\s+', ' ', sqrow['answer']))

                assert decomp_qs_var and decomp_ans

                decomp_ans_dict = {f'#{i+1}': ans for i, ans in enumerate(decomp_ans)}

                for qqsid, d_qs_v in enumerate(decomp_qs_var):
                    if qqsid==0: 
                        if re.search(r'#[0-9]\b', d_qs_v): print('\t\tMUSIQUE initial fail:', qqsid, d_qs_v, q)
                    else: 
                        variables = re.findall(r'#[0-9]\b', d_qs_v)
                        if not variables: print('MUSIQUE subsequent decomp Q without variables', qqsid, d_qs_v, q)
                        for v in variables: d_qs_v = d_qs_v.replace(v, decomp_ans_dict[v])
                        assert not re.findall(r'#[0-9]', d_qs_v), ('\t\tMUSIQUE final fail:', qqsid, d_qs_v, q)
                    decomp_qs_filled.append(re.sub('\s+', ' ', d_qs_v))

            line = {'id'  : idx,
                'text': {'contexts': None, 'qs': [q], 'as': [a] if a is not None else None},
                'decomp_qs': {
                    'text': {'contexts': None, 'qs': decomp_qs_filled, 'as': decomp_ans,
                             'qs_var': decomp_qs_var},
                },
                'num_hops': int(idx[0]),
                'original_info': qrow}
            
            holder[split].append(line)

    return holder

# 3. BREAK
def format_break(level = 'high'):
    if level != 'high': raise NotImplementedError
    holder = {}
    for split in ['train', 'validation', 'test']:
        ss = 'dev' if split == 'validation' else split
        
        df_src = pd.read_csv(f'Break-dataset/QDMR-high-level/{ss}.csv')

        holder[split] = []
        for __, xx in df_src.iterrows():
            qrow = xx.to_dict()
            idx     = qrow.pop('question_id')
            q       = qrow.pop('question_text').strip()
            assert q
            q = make_q_wellformed(q)

            decomp_qs_var = None
            if split != 'test':
                decomp_qs_var = [re.sub('\s+', ' ', d.strip()) for d in qrow['decomposition'].split(';')]
                assert decomp_qs_var 
            
            # NOTE: BREAK-High does not include answers to the sub-questions
            line = {'id'  : idx,
                'text': {'contexts': None, 'qs': [q], 'as':  None},
                'decomp_qs': {
                    'text': {'contexts': None, 'qs': None, 
                             'qs_var': decomp_qs_var, 'as': None},
                },
                'num_hops': len(decomp_qs_var) if split != 'test' else None,
                'original_info': qrow}
            
            holder[split].append(line)

    return holder

# 4. 2wikimultihop
def format_2wikimultihop():
    from  prepare_data_unifiedformat_2wiki_utils import (wikimultihop_comparison, 
                                                         wikimultihop_bridge_comparison,)

    with open('02_support_data/03b_musique_relation_mapping.json', encoding = 'utf-8') as f:
        relations_mapping = json.load(f)
    with open('02_support_data/03c_data_2wikimultihop_relation_mapping.json', encoding = 'utf-8') as f:
        relations_mapping.update(json.load(f))

    holder = {}
    for split in ['train', 'validation', 'test']:
        holder[split] = []
        ss = 'dev' if split == 'validation' else split
        with open(f'data_2wikimultihop/data_ids_april7/{ss}.json', encoding = 'utf-8') as f:
            data = json.load(f)

        print('WORKING on', split)
        for __, l in enumerate(tqdm.tqdm(data)):
            idx     = l['_id']
            q       = l['question']

            #'supporting_facts': [['Man from Tangier', 0], ['Tarnished Reputations', 0]],
            #  'evidences': [['Man from Tangier', 'director', 'Lance Comfort'],
            #   ['Tarnished Reputations', 'director', 'Herbert Blache'],
            #   ['Tarnished Reputations', 'director', 'Alice Guy-Blaché'],
            #   ['Tarnished Reputations', 'director', 'Léonce Perret']],

            a2var = {t[-1]: i+1 for i,t in enumerate(l['evidences'])}
            if split == 'test': 
                a = None
                decomp_qs_filled = decomp_ans = decomp_qs_var = None
            else: 
                a       = l['answer'] 
                decomp_qs_filled, decomp_ans = [], []
                decomp_qs_var = []
                for sqid, trp in enumerate(l['evidences']):
                    sq_template = relations_mapping[trp[1]][0]
                    sq          = sq_template.replace('#X', trp[0])
                    sq_ans      = trp[-1]
                    # if subject is a variable answer, use the variable
                    sbj         = trp[0]
                    if a2var.get(sbj, 100)-1 < sqid: sbj = f'#{a2var[sbj]}'
                    sq_var      = sq_template.replace('#X', sbj)
                    decomp_qs_filled.append(sq)
                    decomp_ans.append(sq_ans)
                    decomp_qs_var.append(sq_var)
                
                # ensure bridge_comparison and comparison sequence of sub-questions are well-formed
                # add comparison sub-questions not explicit in the evidences
                c_type = l['type']
                if c_type in ['comparison', 'bridge_comparison']: 
                    boolean = a.lower() in ['yes', 'no']
                    wikimultihop_func = wikimultihop_bridge_comparison \
                        if c_type == 'bridge_comparison' \
                        else wikimultihop_comparison
                    add_q, add_a = wikimultihop_func(cq = q, cq_ans = a, 
                                            evidences = l['evidences'], 
                                            boolean = boolean)
                    
                    print('\t\t📕📕 ADDITIONAL:', add_q, add_a, q, a, decomp_qs_var)
                    
                    decomp_ans.extend(add_a)
                    decomp_qs_var.extend(add_q)
                    add_var2ans = {f'#{i+1}': ans for i, ans in enumerate(decomp_ans)}
                    for aq in add_q:
                        if re.search(r'#[0-9]\b', q): 
                            aq = re.sub(r'#[0-9]\b', lambda x: add_var2ans[x.group()], aq)
                        decomp_qs_filled.append(aq)

            for key in ['_id', 'question', 'evidences', 'answer']: l.pop(key)
            
            line = {'id'  : idx,
                'text': {'contexts': None, 'qs': [q], 'as': a},
                'decomp_qs': {
                    'text': {'contexts': None, 'qs': decomp_qs_filled, 'as': decomp_ans,
                                'qs_var': decomp_qs_var},
                },
                'num_hops': len(decomp_qs_var) if split != 'test' else None,
                'original_info': l}
            
            holder[split].append(line)

    return holder

def make_q_wellformed(q):
    '''
    helper to ensure that (i) questions have '?'; and (ii) questions are truecased (if
    their first word is not capitalized or is all capitalized)
    '''
    
    question_words = set(['what', 'who', 'when', 'where', 'which', 'why', 'how'])

    # processing to ensure questions well-formed. help improve AMR parsing quality
    if q[-1] != '?': 
        
        # first, remove odd trailing chars
        if q[-1] in ['>', ',', '/', '`', ':', '\\']: q = q[:-1]
        
        # add '?' if missing and question word present
        if q.split()[0] in question_words and q[-1].isalpha(): q += '?'

        # a number of questions '"' and '?' flipped
        elif q[-2:] == '?"': q = q[:-2] + '"?'

        # ending with '.' instead of '?'
        elif q[-1] == '.': q = q[:-1] + '?'

        # finally, for those still without '?', add '?'
        if q[-1] != '?': q += '?'
    
    if q[0].islower() or q.isupper():
        q = predict_truecase(q, TRUECASE_OBJS)
    
    return FSMT_TOK.moses_detokenize(q.split(), lang='en')

def format_dqg_gpt4o_errors(dataset, num_ranks = 4, num_instances = 5000):
    random.seed(54506)
    if num_ranks > 6: raise ValueError('num_ranks should be less than or equal to 6')
    
    ####################################################
    fp = f'01_unified_format/UNIFIED_{dataset}_train.jsonl'
    with open(fp, encoding = 'utf-8') as f: 
        original_data = {}
        for l in f: 
            line    = json.loads(l)
            oid     = line.pop('id')
            original_data[oid] = line
    print('LOADED ORIGINAL DATA')

    ####################################################
    fp      = f'03_decompqg_errors/outputs/01_CONSOLIDATED_gpt4o_outputs_{dataset}.jsonl'
    data    = defaultdict(dict)
    with open(fp, encoding = 'utf-8') as f:
        for l in f:
            line    = json.loads(l)
            idx     = line.pop('custom_id')
            errid, oid, score = re.search(r'(?:error-)(\d+)(?:_)(.+)(?:-)(\d\.\d)', idx).groups()
            line['id']          = oid
            line['error_id']    = int(errid)
            line['score']       = float(score)

            # remove unused
            line.pop('error')
            
            # extract 
            bline = line.pop('response')['body']
            assert len(bline['choices']) == 1
            line['fingerprint'] = (bline['model'], bline['system_fingerprint'])
            line['outputs']     = bline['choices'][0]['message']['content']
            
            data[oid][line['score']] = line
    print('LOADED DQG GPT40 PRODUCED ERRORS')

    ####################################################
    # reverse the scores (syndqg scores increasing for amount of errors)
    pool    = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0] 
    holder  = {}
    idxes   = []
    while len(idxes) < num_instances:
        idxes += list(data.keys())
    random.shuffle(idxes)
    seen  = defaultdict(list)
    print('SHUFFLED AND READ TO START....')
    for split in ['test']:
        holder[split] = []
        for lid, oid in enumerate(tqdm.tqdm(idxes)):
            try: 
                cq = original_data[oid]['text']['qs']
                assert type(cq) == list and len(cq) == 1, cq
                cq = cq[0]

                run = True
                tries = 0
                while run: 
                    pick_scores = sorted(random.sample(pool, k = num_ranks))
                    if pick_scores not in seen[oid] or tries > 10: run = False
                    tries += 1
                        
                if tries > 10: 
                    print('Warning: Failed to find unique scores for {}, retrying...'.format(oid))
                    continue
                
                for i in range(len(pick_scores)-1):
                    assert pick_scores[i+1] > pick_scores[i]
                seen[oid].append(pick_scores)

                original_lines = []
                sqs_list = []
                oline = copy.deepcopy(original_data[oid]['decomp_qs']['text'])
                for score in pick_scores:
                    cscore = score*-1
                    if cscore == 0.0:
                        line = copy.deepcopy(oline)
                        sqs = line.pop('qs_var')
                    else:
                        line = copy.deepcopy(data[oid][cscore])
                        sqs = line['outputs'].pop('modified_SQs')
                    
                    sqs = [sq for sq in sqs if sq and sq.strip() != '']
                    sqs_list.append(sqs)

                    if dataset == 'breakhigh': 
                        if 'original_info' not in line: line['original_info'] = {}
                        line['operators'] = original_data[oid]['original_info']['operators']
                    
                    original_lines.append(line)

                holder[split].append({'query':      cq, 
                                    'paragraphs':   sqs_list,  # list of modified SQs
                                    'query_id':     f'{oid}|{lid}',
                                    'scores':       pick_scores,
                                    'original_lines': original_lines,})
            
            except Exception as e: print(f'Error, {oid}'.format() + str(e))
            
        holder[split] = holder[split][:num_instances]
    
    return holder

if __name__ == '__main__':
    import os
    dp = '01_unified_format'
    if not os.path.exists(dp): os.makedirs(dp)

    # 1. HotpotQA
    dataset = 'hotpotqa'
    holder = format_hotpotqa()
    for split, rows in holder.items():
        variant = 'fullwiki' if split != 'validation' else 'distractor'
        print(f'WORKING on {dataset} {split}')
        with open(os.path.join(dp, f'UNIFIED_{dataset}_{variant}_{split}.jsonl'),  
                  encoding = 'utf-8', mode = 'w+') as f: 
            for row in rows: f.write(json.dumps(row)+'\n')

    # 2. MuSiQue
    dataset = 'musique'
    holder = format_musique()
    for split, rows in holder.items():
        print(f'WORKING on {dataset} {split}')
        with open(os.path.join(dp, f'UNIFIED_{dataset}_{split}.jsonl'),  
                  encoding = 'utf-8', mode = 'w+') as f: 
            for row in rows: f.write(json.dumps(row)+'\n')
    
    # 3. BREAK (high)
    dataset = 'breakhigh'
    holder = format_break(level='high')
    for split, rows in holder.items():
        print(f'WORKING on {dataset} {split}')
        with open(os.path.join(dp, f'UNIFIED_{dataset}_{split}.jsonl'),  
                  encoding = 'utf-8', mode = 'w+') as f: 
            for row in rows: f.write(json.dumps(row)+'\n')

    # 4. 2wikimultihop
    dataset = '2wikimultihop'
    holder = format_2wikimultihop()
    for split, rows in holder.items():
        print(f'WORKING on {dataset} {split}')
        with open(os.path.join(dp, f'UNIFIED_{dataset}_{split}.jsonl'),  
                  encoding = 'utf-8', mode = 'w+') as f: 
            for row in rows: f.write(json.dumps(row)+'\n')

    # 5. GPT-4 produced errors for DQG
    for nranks in [4, 6]:
        for source_dataset in ['breakhigh', 'musique']:
            dataset = f'syndqggpt4o-{source_dataset}'
            holder = format_dqg_gpt4o_errors(dataset = source_dataset, 
                                             num_ranks = nranks, 
                                             num_instances = 5 if source_dataset in ['breakhigh'] else 5000)
            for split, rows in holder.items():
                print(f'WORKING on {dataset} {split}')
                with open(os.path.join(dp, f'UNIFIED_RANK_{dataset}-{nranks}_{split}.jsonl'),  
                        encoding = 'utf-8', mode = 'w+') as f: 
                    for row in rows: f.write(json.dumps(row)+'\n')