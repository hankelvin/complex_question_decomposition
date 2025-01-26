import re, random, datetime
from collections import Counter

DATE_FORMATS = ['%d %B %Y', 
                '%B %d, %Y', 
                '%d %B, %Y', 
                '%B %d %Y',
                '%Y',
                '%B %Y',
                '%Y-%m-%d',
                '%Y-%m',
                '%Y-%m-%d']

##########################################
C_PREDICATES = {1: re.compile('the same producer|the same director|the same occupation'),
                2: re.compile('the same nationality|the same country|born in the same place'),
                3: re.compile('opened first|formed earlier|founded first|established first'),
                4: re.compile('released more recently|was released earlier|released first|published first|came out earlier|came out first'),
                5: re.compile('has more producers|has more directors'),
                6: re.compile('more scope of profession|wider scope of profession'),
                7: re.compile('died first|died earlier|died later'),
                8: re.compile('born first|born earlier|born later'),
                9: re.compile('is older|is younger'),
                10: re.compile('lived longer')}

BC_PREDICATES = {1: re.compile('born first|born earlier|born later'),
                 2: re.compile('died first|died earlier|died later'),
                 3: re.compile('the same nationality|the same country'),
                 4: re.compile('is older|older than the other|is younger'),}


# COMPARISON 
def wikimultihop_comparison(cq, cq_ans, evidences, boolean, c_preds = C_PREDICATES):
    add_questions, add_answers = [], []
    
    # the same producer|the same director|the same occupation
    if re.search(c_preds[1], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'director' in t[1] or 'producer' in t[1] or 'occupation' in t[1]]
        # an evidence could be like (note multiple occupations for a given subject): 
        # ['Asli Hassan Abade', 'occupation', 'pilot'],
        # ['Asli Hassan Abade', 'occupation', 'military figure'],
        # ['Asli Hassan Abade', 'occupation', 'civil activist'],
        # ['Albrecht Alt', 'occupation', 'theologian'],
        # ['Albrecht Alt', 'occupation', 'lecturer'],
        # ['Albrecht Alt', 'occupation', 'professor']],
        qst = 'Are there any repeitions in the following: '
        for i, av in enumerate(ans_vars): 
            if i == 0: qst += f'#{av}'
            else: qst += f' and #{av}'
        qst += '?'
        add_questions.append(qst)
        assert boolean, (cq, evidences, ans_vars)
        add_answers.append(cq_ans)

    # the same nationality|the same country|born in the same place
    elif re.search(c_preds[2], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'country' in t[1] or 'birth' in t[1]]
        # assert len(ans_vars) == 2, (cq, evidences, ans_vars)
        qst = 'Are '
        for i, av in enumerate(ans_vars): 
            if i == 0: qst += f'#{av}'
            else: qst += f' and #{av}'
        qst += ' the same?'
        add_questions.append(qst)
        assert boolean, (cq, evidences, ans_vars)
        add_answers.append(cq_ans)

    # opened first|formed earlier|founded first|established first
    elif re.search(c_preds[3], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'inception' in t[1]]
        ans_cands = [get_dt_obj(evidences[i-1][-1]) for i in ans_vars]
         
        assert len(ans_vars) == 2, (cq, evidences, ans_vars)
        qst = 'Which comes first: '
        for i, av in enumerate(ans_vars):
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        add_answers.append(str(min(ans_cands)))

        # e.g. 'Which magazine' or 'Which high school'
        affix = re.search(r'.+ was (opened|formed|founded|established)', cq)[0]
        add_questions.append(f'{affix} on #{len(evidences)+len(add_questions)}?')
        add_answers.append(cq_ans)

    # released more recently|was released earlier|released first|published first|came out earlier|came out first
    elif re.search(c_preds[4], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'publication' in t[1] or 'inception' in t[1]]
        assert ans_vars, (cq, evidences, ans_vars)
        ans_cands = [get_dt_obj(evidences[i-1][-1]) for i in ans_vars]
        
        qst = 'Which comes first: '
        for i, av in enumerate(ans_vars):
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        add_answers.append(str(min(ans_cands)))

        # e.g. 'Which film' or 'Which song'
        affix = ' '.join(cq.split()[:2])
        add_questions.append(f'{affix} was released/published on #{len(evidences)+len(add_questions)}?')
        add_answers.append(cq_ans)

    # has more producers|has more directors
    elif re.search(c_preds[5], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'producer' in t[1] or 'director' in t[1]]
        # get the films (i.e. subjects)
        ans_cands = [evidences[i-1][0] for i in ans_vars]
        ctr = Counter(ans_cands)
        qst = 'Which of these numbers is larger:'
        for i, av in enumerate(ctr.values()): 
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        ans = max(ctr.values())
        add_answers.append(ans)

        if 'producer' in cq: affix = 'producer'
        elif 'director' in cq: affix = 'director'
        qst = f'Which film has #{len(evidences) + len(add_questions)} {affix}(s)?'
        add_questions.append(cq_ans)

    # more scope of profession|wider scope of profession
    elif re.search(c_preds[6], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'occupation' in t[1]]

        qst = 'Which of these occupations has a wider scope: '
        for i, av in enumerate(ans_vars): 
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        ans_cands = [evidences[i-1][-1] for i in ans_vars \
                     if evidences[i-1][0].strip().lower() == cq_ans.strip().lower() \
                        and 'occupation' in evidences[i-1][1]]
        # we don't know the answer (some evidences have multiple occupations for 1 subject),
        # we will randomly pick one that is held by the cq_ans
        try: ans = random.choice(ans_cands)
        except: 
            print(cq, cq_ans, evidences, ans_vars, ans_cands)
            raise ValueError
        add_answers.append(ans) 

        add_questions.append(f'Who has this profession #{len(evidences) + len(add_questions)}?')
        add_answers.append(cq_ans)

    # died first|died earlier|died later
    elif re.search(c_preds[7], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'death' in t[1]]
        assert len(ans_vars) == 2, (cq, evidences, ans_vars)

        ans_dates = [evidences[i-1][-1] for i in ans_vars]
        dt_objs = [get_dt_obj(ac) for ac in ans_dates]
        
        affix = re.search(r'(first|earlier|later)', cq)[0]
        qst = f'Which is {affix}: '
        for i, av in enumerate(ans_vars): 
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        if affix == 'later': func = max
        else: func = min
        pick_idx = dt_objs.index(func(dt_objs))
        ans = ans_dates[pick_idx]
        add_answers.append(str(ans))

        add_questions.append(f'Who died on/in #{len(evidences) + len(add_questions)}?')
        add_answers.append(cq_ans)

    # born first|born earlier|born later
    elif re.search(c_preds[8], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'birth' in t[1]]
        assert len(ans_vars) == 2, (cq, evidences, ans_vars)
        
        ans_dates = [evidences[i-1][-1] for i in ans_vars]
        dt_objs = [get_dt_obj(ac) for ac in ans_dates]
        
        affix = re.search(r'(first|earlier|later)', cq)[0]
        qst = f'Which is {affix}: '
        for i, av in enumerate(ans_vars): 
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        if affix == 'later': func = max
        else: func = min
        pick_idx = dt_objs.index(func(dt_objs))
        ans = ans_dates[pick_idx]
        add_answers.append(str(ans))

        add_questions.append(f'Who was born on #{len(evidences) + len(add_questions)}?')
        add_answers.append(cq_ans)

    # is older|is younger
    elif re.search(c_preds[9], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'birth' in t[1]]
        assert len(ans_vars) == 2, (cq, evidences, ans_vars)
        
        ans_dates = [evidences[i-1][-1] for i in ans_vars]
        dt_objs = [get_dt_obj(ac) for ac in ans_dates]
        
        if 'older' in cq: 
            func = max 
            affix = 'more'
        else: 
            func = min
            affix = 'less'
        
        qst = f'Which is {affix}: '
        for i, av in enumerate(ans_vars):
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        pick_idx = dt_objs.index(func(dt_objs))
        ans = ans_dates[pick_idx]
        add_answers.append(str(ans))
        
        add_questions.append(f'Who was born on #{len(evidences) + len(add_questions)}?')
        add_answers.append(cq_ans)

    # lived longer
    elif re.search(c_preds[10], cq):        
        ans_vars_birth = [i+1 for i,t in enumerate(evidences) if 'birth' in t[1]]
        ans_vars_death = [i+1 for i,t in enumerate(evidences) if 'death' in t[1]]
        assert len(ans_vars_birth) == len(ans_vars_death), (cq, evidences, ans_vars_birth, ans_vars_death)
        for i, (b, d) in enumerate(zip(ans_vars_birth, ans_vars_death)):
            # ensure that the birth and death dates are for the same person
            assert evidences[b-1][0] == evidences[d-1][0], (cq, evidences, ans_vars_birth, ans_vars_death)
        
        ans_dates_birth = [evidences[i-1][-1] for i in ans_vars_birth]
        ans_dates_death = [evidences[i-1][-1] for i in ans_vars_death]
        dt_objs_birth = [get_dt_obj(ac) for ac in ans_dates_birth]
        dt_objs_death = [get_dt_obj(ac) for ac in ans_dates_death]
        dt_objs = [b-d for b,d in zip(dt_objs_birth, dt_objs_death)]

        qst = f'Which is longer: '
        for i, av in enumerate(ans_vars_birth):
            if i == 0: qst += f'#{av} - #{ans_vars_death[i]}'
            else: qst += f' or #{av} - #{ans_vars_death[i]}'
        qst += '?'
        add_questions.append(qst)
        pick_idx = dt_objs.index(max(dt_objs))
        ans = ans_dates_birth[pick_idx] + ' - ' + ans_dates_death[pick_idx]
        add_answers.append(ans)
        
        add_questions.append(f'Who lived from #{len(evidences) + len(add_questions)}?')
        add_answers.append(cq_ans)

    else: raise ValueError

    return add_questions, add_answers

# BRIDGE COMPARISON
def wikimultihop_bridge_comparison(cq, cq_ans, evidences, boolean, bc_preds = BC_PREDICATES):
    add_questions, add_answers = [], []
    # born first/earlier/later
    if re.search(bc_preds[1], cq): 
        add_questions, add_answers = \
            one_entry_bridge_born_died(cq, cq_ans, boolean, evidences, born = True)
    
    # died first/earlier/later
    elif re.search(bc_preds[2], cq):
        add_questions, add_answers = \
            one_entry_bridge_born_died(cq, cq_ans, boolean, evidences, born = False)

    elif re.search(bc_preds[3], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'country of' in t[1]]
    
        qst = 'Are '
        for i, av in enumerate(ans_vars): 
            if i == 0: qst += f'#{av}'
            else: qst += f' and #{av}'
        qst += ' the same?'

        add_questions.append(qst)
        assert boolean, (cq, evidences, ans_vars)
        add_answers.append(cq_ans)

    # is older/younger
    elif re.search(bc_preds[4], cq):
        ans_vars = [i+1 for i,t in enumerate(evidences) if 'birth' in t[1]]
        assert not boolean, (cq, evidences, ans_vars)
        assert len(ans_vars) == 2, (cq, evidences, ans_vars)
        
        ans_cands = [evidences[i-1][0] for i in ans_vars]
        ans_dates = [evidences[i-1][-1] for i in ans_vars]
        dt_objs = [get_dt_obj(ac) for ac in ans_dates]
        
        if 'older' in cq: 
            func = max 
            affix = 'more'
        else: 
            func = min
            affix = 'less'

        qst = f'Which is {affix}: '
        for i, av in enumerate(ans_vars):
            if i == 0: qst += f'#{av}'
            else: qst += f' or #{av}'
        qst += '?'
        add_questions.append(qst)
        pick_idx = dt_objs.index(func(dt_objs))
        ans = ans_dates[pick_idx]        
        add_answers.append(str(ans))
        
        if 'director' in cq: 
            affix = 'direct'
            qst = f'Which director was born on #{len(evidences)+len(add_questions)}?'
        elif 'producer' in cq: 
            affix = 'produce'
            qst = f'Which producer was born on #{len(evidences)+len(add_questions)}?'

        add_questions.append(qst)
        ans = ans_cands[pick_idx]
        add_answers.append(ans)

        add_questions.append(f'What film did #{len(evidences)+len(add_questions)} {affix}?')
        add_answers.append(cq_ans)

    else: raise ValueError


    return add_questions, add_answers
    

def one_entry_bridge_born_died(cq, cq_ans, boolean, evidences, born = True):
    # born first/earlier/later
    # died first/earlier/later
    add_questions = []
    add_answers   = []

    if born:
        c_string1  = 'birth'
        c_string2  = 'born'
    else: 
        c_string1  = 'death'
        c_string2  = 'died'
    
    # get the positions of the evidence (the answer) where the birth/death is mentioned
    # in the predicate (i.e. what is being compared)
    ans_vars = [i+1 for i,t in enumerate(evidences) if c_string1 in t[1]] # e.g. 'birth'
    assert not boolean, (cq, evidences, ans_vars)
    # see paper https://aclanthology.org/2020.coling-main.580.pdf (appendix A)
    assert cq.lower().startswith('which film'), cq 
    assert len(ans_vars) == 2, (cq, evidences, ans_vars)
    
    ans_dates = [evidences[i-1][-1] for i in ans_vars]
    dt_objs = [get_dt_obj(ac) for ac in ans_dates]
    
    ans_cands = [evidences[i-1][0] for i in ans_vars]

    if re.search(f'{c_string2} later', cq): # e.g. 'born'
        func = max 
        affix = 'later'
    else: 
        func = min
        affix = 'first/earlier'
    
    # get min/max datetime obj, get its index pos in ans_cands, get the answer
    pick_idx = dt_objs.index(func(dt_objs))
    ans = ans_dates[pick_idx]
    add_answers.append(ans)
    add_questions.append(f'Which is {affix} #{ans_vars[0]} or #{ans_vars[1]}?')

    
    if 'director' in cq: affix = 'director'
    elif 'producer' in cq: affix = 'producer'
    if c_string2 == 'died': copula = ''
    else: copula = 'was '

    # get the person who was born/dead 
    qst = f'Which {affix} {copula}{c_string2} on #{len(evidences)+len(add_questions)}?'
    add_questions.append(qst)
    ans = ans_cands[pick_idx]
    add_answers.append(ans)

    if 'director' in cq: affix = 'direct'
    elif 'producer' in cq: affix = 'produce'

    # len(evidences) + len(add_questions) to reference just added qst above
    add_questions.append(f'What film did #{len(evidences)+len(add_questions)} {affix}?')
    add_answers.append(cq_ans)

    return add_questions, add_answers

def get_dt_obj(date_str, date_formats = DATE_FORMATS):
    check = date_str.strip().split()
    if len(check) == 3: 
        if check[-1].isnumeric() and len(check[-1]) == 3: 
            # e.g. '13 August 900'
            date_str = ' '.join(check[:-1]) + ' ' + check[-1].zfill(4)
    elif len(check) == 1:
        if check[0].isnumeric() and len(check[0]) == 3: 
            # e.g. '164'
            date_str = check[0].zfill(4)

    dt = None
    for df in date_formats:
        try: 
            dt = datetime.datetime.strptime(date_str.strip(), df).date()
            return dt
        except: pass
    
    if dt is None: raise ValueError(f'Could not parse date: [{date_str.strip()}]')
    