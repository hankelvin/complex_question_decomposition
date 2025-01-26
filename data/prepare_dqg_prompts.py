'''
This script randomly picks 50 instances from each of the 3 datasets
(1) 2wikimultihop
(2) breakhigh
(3) musique

for breakhigh, we keep the QDMR format
'''


import json, random, math
from collections import defaultdict, Counter

random.seed(54506)
holder = defaultdict(list)
num_samples = 50
for dataset in ['2wikimultihop', 'breakhigh', 'musique']:
    with open(f'01_unified_format/UNIFIED_{dataset}_train.jsonl', 'r') as f:
        data = [json.loads(l) for l in f]

    if dataset == 'strategyqa':
        # 1st sampling of 50 instances was done with this script and the following indices were picked
        # manual inspection was  subsequently done (see bottom). to ensure only manually inspected instances 
        # are used (and no need to reinspect again), we filter for these indices only
        with open('02_support_data/strategyqa_inspected_idxes.txt', encoding  = 'utf-8') as f:
            inspected_idxes = set([l.strip() for l in f.readlines()])

        with open(f'01_unified_format/UNIFIED_{dataset}_validation.jsonl', 'r') as f:
            validation_data = [json.loads(l) for l in f]
        val_idxes = set([v['id'] for v in validation_data])

        data = [d for d in data if d['id'] in inspected_idxes and d['id'] not in val_idxes]


    random.shuffle(data)
    sq_sizes = set(len(d['decomp_qs']['text']['qs_var']) for d in data)
    sq_sizes = {s: math.ceil(num_samples/len(sq_sizes)) for s in sq_sizes}
    print(f'{dataset} sq_sizes:', sq_sizes)

    for line in data:
        if dataset != 'strategyqa':
            if sum(sq_sizes.values()) == 0: break

        idx = line['id']
        cq = line['text']['qs']
        assert len(cq) == 1
        cq = cq[0]

        cq_ans = line['text'].get('as', None)
        if cq_ans and type(cq_ans) == list: cq_ans = cq_ans[0]

        sq_list = line['decomp_qs']['text']['qs_var']
        sq_sizes[len(sq_list)] -= 1
        sq_ans_list = line['decomp_qs']['text'].get('as', None)
        original_info = line['original_info']
        newline = {'id': idx, 'cq': cq, 'cq_ans': cq_ans, 
                   'sq_list': sq_list, 'sq_ans_list': sq_ans_list,
                   'original_info': original_info}
        
        newline['dataset'] = dataset
        if dataset == 'breakhigh':
            newline['operators'] = eval(line['original_info']['operators'])
        
        holder[dataset].append(newline)

    random.shuffle(holder[dataset])
    print(f'{dataset} num_samples obtained:', len(holder[dataset]))

if 'strategyqa' in holder:
    # after 1st sampling of 50 instances, a manual inspection was done to check 
    # if (i) the statements in 'facts' and their orders allow them to be used as answers 
    # to the SQs (in original order)
    # NOTE: the following indices are not suitable for CoT as facts are not in order 
    fails = set(['27551dd918fdafe87524', '60c525b944e991fb9821', '0cb73f1ccb217757bddf',
        'b257b34db67a10038f18', '1a8fb1401bdae961beea', '77f814c0e9766c9cdb4d',
        '1111510e448112ed3c85', '1ab9f6651469645c0573', '3655a4efabd3358429d0',
        '48cac3b98391f6da285f', '979b4b0fa0a8606bfcae', '8116aa0a9157b809ac9b',
        '2cb0bc060c5fb708a43f', '74936a1e1f16a8e97d68', '465c5d8486aa87851072',
        '53ddbc5daaa0bb43606e'])
        
    holder['strategyqa'] = [d for d in holder['strategyqa'] if d['id'] not in fails]

    # a number of instances have fact orders that can be easily reversed to match the SQs
    # we do so here to make them suitable for CoT
    reverses = set(['46fc399a48a40e78dc60', '25cb5d3136c997326121', 'e175b012fc9b5db8da3f',
           '0a32d7cfde6cec332fd6', '040f15ccc61888c73b48','47e4f407f7186ba6b86f'])
    
    for i, ent in enumerate(holder['strategyqa']):
        idx = ent['id']
        sq_list = ent['sq_list']
        sq_ans_list = ent['sq_ans_list']
        if idx in reverses: sq_ans_list.reverse()
        # add None to sq_ans_list if needed
        sq_ans_list += [None for i in range(len(sq_list)-len(sq_ans_list))] 

        holder['strategyqa'][i]['sq_ans_list'] = sq_ans_list

    print('strategyqa num_samples obtained:', len(holder['strategyqa']))


with open(f'02_support_data/2wiki_breakhigh_musique_strategyqa_prompt_cands.json', 
          encoding = 'utf-8', mode = 'w') as f:
    json.dump(holder, f)