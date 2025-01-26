import json, random, copy, os, re
'''
This script loads the training split of the following datasets:

- musique
- breakhigh 
- 2wikimultihop

and samples N instances from each dataset

Each of these instances will be sent to GPT-4o to produce a rewrite
so that the text in the decomposed questions is different from the original, 
and at the same time introduce one or more faults into the decomposed questions.

The objective is to obtain multiple sets of synthetic DQG outputs (mimicking 
system outputs) which we can gather to use for ranking their quality.
'''

########################################################################
MODEL = "gpt-4o-2024-08-06"
TEMPLATE = '''You are presented with a complex question (CQ) and an ordered sequence of subquestions (SQs). The SQs have been verified as being simpler decomposed questions of the CQ, i.e. when the SQs are correctly answered, the answer obtained for the final one will be the same as that for the CQ. The references to the answer of an earlier SQ is marked with a # followed by a number that is 1-indexed, e.g. in the SQ "SQ3 >> When was the company in #1 founded?", "#1" refers to the answer to the first SQ in the sequence; it is very important that you make sure to maintain this format when modifying the SQs. Your task is to paraphrase the SQs, and then introduce one or more faults into them as instructed. 

Complex question (CQ): {}
{}
Subquestions (SQs): {}
{}

Do the following steps: 
Step 1: Please paraphrase at least half of the SQs. When paraphrasing an SQ, try to give it a different syntactic structure. It is very important to keep the meaning of the original SQ intact. Do not add or remove any information from any of the SQs at this step.

Step 2: Please introduce the following faults into the SQs. It is very important: (i) To follow each of the instructions carefully. (ii) That you must not add or remove any SQs unless one of the instructions specifically asks you to do so. {}
{}

Step 3: Return the results in json_schema format with the following keys: (i) "modified_SQs" - is a list holding the SQs after the modifications, (ii) "index_modsSQs" - is a list holding the indices of the SQs that were modified, (iii) "mod_desc" - a list that corresponds with "index_modsSQs" with a short description (of 7 words or less) of what was modified for that SQ. Only return the modifed SQs, i.e. you MUST NOT include the prefix (e.g. "SQ1 >> ") from the input. Only return this json line, do not say anything else before or after it.'''

BATCH_LINE = {"custom_id": None, "method": "POST", "url": "/v1/chat/completions", 
            "body": {"model": MODEL, "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": None}], "max_tokens": 500}}

ERRORS_MAP = \
{
################################################################
1: '''Introduce one or two syntactic and grammatical errors to SQ{0}; make sure that it causes the SQ to be less fluent but does not change its meaning, i.e. the errors should not cause the SQ to be unanswerable or change its answer. (0.5 points)''',
2: '''Introduce one or two typographical errors to one of the entity/entities mentioned in SQ{0}, but make sure that the errors do not cause the SQ to be unanswerable or change its answer. (0.5 points)''',
3: '''Randomly pick one of the words in SQ{0} and repeat it. (0.5 points)''',
4: '''Randomly pick a word in SQ{0} that is (i) more than five characters long, and (ii) part of a named entity; split this word into two parts and replace one part with the span "<unk>" (e.g. the named entity "Montréal" becomes "Mont<unk>")". (0.5 points)''',
################################################################
5: '''Add a new SQ after SQ{0}; this new SQ should have some relation to the answer of any of the preceding SQs; make sure that answering this additional SQ is totally unnecessary for obtaining the final answer, but answering it will not affect whether the final answer is obtainable. Because there is now one more SQ, you must make sure to update the answer variable number of any subsequent SQs that referred to the answers of SQs coming before the new SQ you added. (1 point)''', 
6: '''Modify every SQ so that every one of them has one or two grammatical errors in them that are typical of non-native English speakers. Make sure that they are still barely understandable, and that the errors do not cause any of the SQs to be unanswerable or change their answers. (1 point)''',
7: '''Rewrite SQ{0} so that its answer is leaked in it; example: given the SQ "SQ0 >> Who is the president of France?" whose answer is "Emmanuel Macron", a possible modification to the SQ could be: "What is the name of France's President Macron?" (1 point)''', 
################################################################
8: '''Replace SQ{0} and SQ{1} with a new SQ that merges the both of them; example: given these SQs "... SQ2 >> Which street can the tallest building in #1 be found on?; SQ3 >> Which famous actress lives on #2?; SQ4 >> What is the birthplace of #3? ", a possible merger between SQ2 and SQ3 could be: "Which famous actress lives on the street where the tallest building in #1 is located on?". Because there is now one less SQ, you must make sure to update the answer variable number of any subsequent SQs that referred to the original SQ{0} or SQ{1} (e.g. SQ4 in the example should be updated to "What is the birthplace of #2?") (2 points) ''',
9: '''Remove some information from SQ{0} so that it is now ambiguously worded, i.e. there is insufficient information to answer it easily. (2 points)''',
10: '''Make semantic errors that change the meaning of SQ{0} so that its answer is no longer correct (2 points)''',
################################################################
11: '''Change SQ{0} to ask for something different, i.e. after this point in the modified sequence of SQs the reasoning chain is disconnected. DO NOT add new SQs. It is important that the answer to the modified SQ is not the same as the final answers to the CQs or SQs, but make sure that it is still somewhat related to one of the entities mentioned in the CQ. (3 points)''',
12: '''Add a final boolean SQ that makes some comparison between the answers to SQ{0} and SQ{1}. (3 points)''', 
#NOTE: Append Boolean step from BPB,
13: '''Change the CQ to ask for an answer that is entirely different. Rewrite all of the SQs as decompositions of this new CQ. Make sure that all the SQs remain as simple questions asking for one single fact. (3 points)''',
14: '''Completely change all the SQs so that they are now all unrelated to the CQ and all of the orignal SQs. Make sure that none of the answers to the new SQs are the same as the original answers. (3 points)''',
################################################################
}

CONSEQ2_ID = {8}
CONSEQ2_ID_ALT = [9, 10]
ANY2_ID    = {12}
MID_ID     = {11}
NOTLAST_ID = {5, 10}
DISALLOW05 = {1: [6]}

POINTS_MAP = {0.5: [1,2,3,4], 1.0: [5,6,7], 2.0: [8,9,10], 3.0: [11,12,13,14]}

def load_sample_data(dataset, num_samples = 2000):
    random.seed(54506)
    print('WORKING on dataset', dataset)
    fp = f'01_unified_format/UNIFIED_{dataset}_train.jsonl'
    with open(fp, encoding='utf-8') as f: lines = [json.loads(line) for line in f]

    if dataset == 'breakhigh':
        lines = [line for line in lines if 'HOTPOT_' in line['id']]
        for l in lines: l['operators'] = l['original_info'].pop('operators')

    pick_lines = []
    seen_idx   = set()
    short = num_samples - len(pick_lines)
    while short:
        __ = random.sample(lines, short)
        __ = [line for line in __ if len(line['decomp_qs']['text']['qs_var']) > 2 and line['id'] not in seen_idx]
        seen_idx.update(set(line['id'] for line in __))
        pick_lines.extend(__)
        short = num_samples - len(pick_lines)

    HOLDER = []
    if dataset == 'breakhigh':
        breakhigh_string = '(iii) When you make the modifications, it is very important that you MUST NOT change the first word of any of the SQs.'
    else: breakhigh_string = '' 
    for i,line in enumerate(pick_lines):
        idx = line['id']
        
        complex_question    = line['text']['qs'][0]
        answer              = line['text']['as'][0] if line['text']['as'] is not None else None
        answer_string       = f'Answer to CQ: {answer}' if answer else ''
        subquestions        = line['decomp_qs']['text']['qs_var']
        sq_answers          = line['decomp_qs']['text']['as']
        sq_answers_string   = f'Answers to SQs: {sq_answers}' if sq_answers else ''

        num_sq              = len(subquestions)

        errors_holder = produce_errors(num_sq, subquestions)

        for points, errors in errors_holder.items():
            line = make_one_line(i, idx, complex_question, answer_string, subquestions, 
                                sq_answers_string, breakhigh_string, errors, points)
            HOLDER.append(line)

    return HOLDER

def produce_errors(num_sq, subquestions):

    errors_holder = {}

    #### A: 1 x 0.5 ###############################################
    points = 0.5
    if random.random() >= 0.5:  errors_id = [1, ]
    else:                       errors_id = [2, ]
    errors_holder[points] = fill_errors_line(errors_id, num_sq, subquestions)
    ###############################################################

    #### B: 1 x 1.0 - pick 1 or 2 errors (1, or 0.5 + 0.5) ########
    points = 1.0
    if random.random() >= 0.5:   
        errors_id = random.sample(POINTS_MAP[1.0], 1)
    else:                       
        errors_id = random.sample(POINTS_MAP[0.5], 2)
    errors_holder[points] = fill_errors_line(errors_id, num_sq, subquestions)
    ###############################################################

    #### C: 1 x 1.5 - pick 2 errors (1 + 0.5) #####################
    points = 1.5
    errors_id = []
    errors_id.append(random.choice(POINTS_MAP[0.5])) 
    # if error #1 (grammar) already picked, ensure #7 is not added
    watch = set(errors_id).intersection(DISALLOW05)
    pool = POINTS_MAP[1.0].copy()
    if watch:
        for eid in watch: pool = list(set(pool) - set(DISALLOW05[eid]))
    errors_id.append(random.choice(pool)) 
    errors_holder[points] = fill_errors_line(errors_id, num_sq, subquestions)
    ###############################################################
    
    #### D: 1 x 2.0 - pick 1 or 2 errors (2, or 1 + 1) ############
    points = 2.0
    if random.random() >= 0.5 or num_sq <= 3:  
        errors_id = random.sample(POINTS_MAP[2.0], 1)
        while MID_ID.intersection(set(errors_id)) and num_sq == 2: 
            # needs to have enough SQs so 1st and last not modified
            errors_id = random.sample(POINTS_MAP[2.0], 1)
    else:                       
        errors_id = random.sample(POINTS_MAP[1.0], 2)
    errors_holder[points] = fill_errors_line(errors_id, num_sq, subquestions)
    ###############################################################

    #### E: 1 x 2.5 - pick 2 or 3 errors: (2 + 0.5, or 1 + 1 + 0.5)
    points = 2.5
    if random.random() >= 0.5 or num_sq <= 3:   
        errors_id = random.sample(POINTS_MAP[2.0], 1)
        while MID_ID.intersection(set(errors_id)) and num_sq == 2: 
            errors_id = random.sample(POINTS_MAP[2.0], 1)
        errors_id.extend(random.sample(POINTS_MAP[0.5], 1))
    else:                       
        errors_id.extend(random.sample(POINTS_MAP[0.5], 1))
        # if error #1 (grammar) already picked, ensure #7 is not added
        watch = set(errors_id).intersection(DISALLOW05)
        pool = POINTS_MAP[1.0].copy()
        if watch:
            for eid in watch: pool = list(set(pool) - set(DISALLOW05[eid]))
        errors_id = random.sample(POINTS_MAP[1.0], 2)
    errors_holder[points] = fill_errors_line(errors_id, num_sq, subquestions)
    ###############################################################

    #### F: 1 x 3.0 - pick 1 or 2 errors: (3, or 2 + 1, or 1 + 1 + 1)
    points = 3.0
    if random.random() >= 0.33 :   
        errors_id = random.sample(POINTS_MAP[3.0], 1)
    elif random.random() >= 0.66 and num_sq > 3:   
        errors_id = random.sample(POINTS_MAP[1.0], 3)
    else:                       
        errors_id = random.sample(POINTS_MAP[2.0], 1)
        while MID_ID.intersection(set(errors_id)) and num_sq == 2: 
            errors_id = random.sample(POINTS_MAP[2.0], 1)
        errors_id.extend(random.sample(POINTS_MAP[1.0], 1))
    errors_holder[points] = fill_errors_line(errors_id, num_sq, subquestions)
    ###############################################################

    return errors_holder

def make_one_line(i, idx, complex_question, answer_string, subquestions, 
                  sq_answers_string, breakhigh_string, errors, points):
    random.shuffle(errors)
    line = copy.deepcopy(BATCH_LINE)
    
    line['custom_id'] = 'error-{}-{}'.format(f'{i}_{idx}', points)
    subquestions_string = '; '.join([f'SQ{i} >> "{sq}"' for i, sq in enumerate(subquestions)])
    
    prompt = TEMPLATE.format(complex_question, answer_string, 
                             subquestions_string, sq_answers_string, 
                             breakhigh_string, '\n'.join(errors))
    
    line['body']['messages'][1]['content'] = prompt

    return line

def fill_errors_line(errors_id, num_sq, subquestions):
    random.shuffle(errors_id)
    pool = list(range(num_sq))
    num_req = len(errors_id)
    check = len((CONSEQ2_ID | ANY2_ID).intersection(set(errors_id)))
    if check: num_sq += check
    
    # ensure CONSEQ2_ID is front of the list, so that consecutive sqs can be found
    if CONSEQ2_ID.intersection(set(errors_id)): 
        for eid in CONSEQ2_ID: 
            errors_id.remove(eid)
            errors_id.insert(0, eid)
    # ensure MID_ID is near front of the list (1st pos, unless there is CONSEQ2_ID)
    if MID_ID.intersection(set(errors_id)): 
        pos = 1 if CONSEQ2_ID.intersection(set(errors_id)) else 0
        for eid in MID_ID: 
            errors_id.remove(eid)
            errors_id.insert(pos, eid)

    assert len(pool) >= num_req, f'Not enough questions in the pool {pool}, {num_req}'

    error_lines = []
    for eid in errors_id:
        eline = ERRORS_MAP.get(eid)
        
        if eid in CONSEQ2_ID: 
            # pick 2 sqs that are consecutive
            picks = []
            go = True
            while go:
                __ = random.choice(pool[:-1]) # ensure not last
                if __+1 in pool: go = False
            
            c1 = len(subquestions) > __+1+1
            if c1 and len(re.findall(r'\#', subquestions[__+1+1])) >= 2: 
                eline = ERRORS_MAP[random.choice(CONSEQ2_ID_ALT)]

            pool.remove(__)
            pool.remove(__+1)
            picks.append(__)
            picks.append(__+1)
            error_lines.append(eline.format(*picks))

        elif eid in ANY2_ID: 
            # pick any 2 sqs
            picks = random.sample(pool, 2)
            for __ in range(2):
                pool.remove(__)
            error_lines.append(eline.format(*picks))

        elif eid in NOTLAST_ID:
            # pick a sq that is not the last one
            pick = random.choice([i for i in pool if i!= num_sq-1])
            pool.remove(pick)
            error_lines.append(eline.format(pick))

        else:
            if eid in MID_ID:
                # make sure that SQ is not the first SQ
                __pool = [i for i in pool if i!= 0]
            else: __pool = pool
            
            if not __pool: 
                print(f'🚨No eid {eid} found in pool {__pool}')
                __pool = list(range(num_sq))
                # pick 1 sq 
                pick = random.choice(__pool)
            else: 
                pick = random.choice(__pool)
                pool.remove(pick)
            error_lines.append(eline.format(pick))

    return error_lines

for dataset in ['musique', 'breakhigh', '2wikimultihop']:
# for dataset in ['breakhigh']:
    holder = load_sample_data(dataset, num_samples = 500)
    SAVEDIR = '03_decompqg_errors'
    if not os.path.exists(SAVEDIR): 
        os.makedirs(SAVEDIR)
    fp = os.path.join(SAVEDIR, f'01_{dataset}_error_prompts.jsonl')
    with open(fp, 'w+', encoding = 'utf8') as f:
        for line in holder:
            f.write(json.dumps(line) + '\n')