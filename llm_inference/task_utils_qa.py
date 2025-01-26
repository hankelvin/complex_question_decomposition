import re
### ALLOW REUSE ELSEWHERE ###
import os, sys
filedir = os.path.dirname(os.path.abspath(__file__))
cwd     = os.getcwd()
os.chdir(filedir)
#############################
import sys
sys.path.append('..')
from llm_inference.task_utils_decomp_qg import give_n_shot_examples
### BACK #################
os.chdir(cwd)
##########################

def prepare_qa_prompt(args, line, is_sequence = True, repro = True, split = None):
    '''
    is_sequence (bool): whether the prompt should be set up for sequential answering of 
    the decompositions to a complex question. 
    '''
    context_prompt = 'Answer using these supporting information:'
    dataset, split = args.input_file
    c1 = split == 'test'
    c2 = args.test_only == 'phase3_val_as_test' and split == 'validation'
    if args.cross_domain and (c1 or c2):
        dataset = args.cross_domain
    n_shot = args.qa_args['n_shot']
    cot    = args.qa_args['cot']
    am_start = args.qa_args['ans_markers']['start']
    am_end   = args.qa_args['ans_markers']['end']

    cq_messages = give_prefix_qa(args.model_name, prompt_version = args.prompt_version, 
                              qa_task = 'cq', ans_markers = args.qa_args['ans_markers'],
                              cot = cot)
    sq_messages = give_prefix_qa(args.model_name, prompt_version = args.prompt_version, 
                              qa_task = 'sq', ans_markers = args.qa_args['ans_markers'],
                              cot = cot)

    if n_shot > 0:
        examples = give_n_shot_examples(dataset, n_shot, qa_task = True, legacy = not args.do_dqg_llmgen_qa)  
        for i, exline in enumerate(examples):
            cq_content = f'\n{args.user_prompt["cq"]}{exline["cq"]}'

            # add knowledge (e.g. retrieve passages)
            if getattr(args, 'do_dqg_llmgen_qa_usecontext', False):
                context = gather_knowledge_context(args, exline, split)
                cq_content += f'\n{context_prompt}\n{context}\n'
            cq_messages.extend([{'role': 'user', 'content': cq_content}])
            
            sqs             = exline['sq_list']
            sqs_ans         = exline['sq_ans_list']
            sqs_ans_dict    = {i+1: a for i, a in enumerate(sqs_ans)} if sqs_ans else None
            
            cq_cot_one_turn = ''
            for sq_pos, sq in enumerate(sqs):
                ans = sqs_ans_dict.get(sq_pos+1, '[NONE]')
                
                # replace answer vars in the sub-questions
                answer_var = re.findall(r'(?:#)(\d+)', sq)
                answer_var = [int(v) for v in answer_var if int(v) < sq_pos+1]
                for av in answer_var: 
                    sq = sq.replace(f'#{av}', sqs_ans_dict[av])
                
                sq_msg  = f'{args.user_prompt["sq"]}{sq}'
                # ensure answer markers added 
                ans_msg = f'{args.asst_prompt["sq"]}{f"{am_start}{ans}{am_end}"}'

                cq_cot_one_turn += f'{sq_msg}\n{ans_msg}\n'
                
                msgs = [{'role': 'user',        'content': f"{cq_content}\n{sq_msg}" if sq_pos == 0 else sq_msg},
                        {'role': 'assistant',   'content': ans_msg}]

                sq_messages.extend(msgs)
            
            cq_messages.extend([{'role': 'assistant', 'content': cq_cot_one_turn}])

    #####################
    cq = line['text']['qs']
    assert len(cq) == 1, f'🚨\t\tExpected 1 complex question, found {len(cq)}'
    cq = cq[0]
    cq_content = f'{args.user_prompt["cq"]}{cq}'
    if getattr(args, 'do_dqg_llmgen_qa_usecontext', False):
        context = gather_knowledge_context(args, line, split) #  NOTE: ensure 'line'
        cq_content += f'\n{context_prompt}\n{context}\n'
    eo_prompt = [{'role': 'user', 'content': cq_content},]
    cq_messages.extend(eo_prompt)
    sq_messages.extend(eo_prompt)

    ##### Collating the SQs and their answers #####
    sq_messages_successive = []
    sqs = line['decomp_qs']['text']['qs_var']
    # NOTE: the assistant message does not have the answer, the model is being prompted for that
    for sq_pos, sq in enumerate(sqs):
        msgs = [{'role': 'user',        'content': f'{args.user_prompt["sq"]}{sq}'},
                {'role': 'assistant',   'content': f'{args.asst_prompt["sq"]}'}]
        
        sq_messages_successive.extend(msgs)
    ###############################################

    tgt = line['text']['as']
    if tgt is not None: 
        assert len(tgt) == 1
        tgt = tgt[0]

    return cq_messages, sq_messages, sq_messages_successive, tgt, cq

def gather_knowledge_context(args, line):
    c1 = split == 'test'
    c2 = args.test_only == 'phase3_val_as_test' and split == 'validation'
    dataset = args.input_file[0]
    if args.cross_domain and (c1 or c2):
        # ensure properly set up for cross-domain setting
        dataset = args.cross_domain

    context = ''
    if dataset in ['breakhigh', 'hotpotqa', 'hotpotqa_fullwiki', 'hotpotqa_distractor']:
        key = 'context'
        
        supporting_facts = [t[0].strip().lower() for t in line['original_info']['supporting_facts']]
        titles = [i[0] for i in line['original_info'][key]]
        paragraphs = [' '.join(i[1]) for i in line['original_info'][key]]

        ctr     = 0 
        for i, (t, p) in enumerate(zip(titles, paragraphs)):
            if t.strip().lower() in supporting_facts:  
                ctr += 1
                context += f'{ctr}. Title: {t} \t {p}\n'
    
    elif dataset in ['musique']:
        key     = 'paragraphs'
        ctr     = 0
        for c in line['original_info'][key]:
            if c['is_supporting']: 
                ctr += 1
                context += f'{ctr}. Title: {c["title"]} \t {c["paragraph_text"]}\n'

    else: raise NotImplementedError

    return context


def give_prefix_qa(model_name, prompt_version = 2, qa_task = 'cq', 
                   ans_markers = {'start': '[ANS_S]', 'end': '[ANS_E]'},
                   cot = True):

    if model_name in ['mistral', 'gemma']:
        messages = []
    else: 
        messages = [{'role': 'system',
                    'content': '''You are an intelligent assistant that can get to the answer of a complex question by answering a sequence of simpler sub-questions that are decompositions of the complex question.''',},]
    if prompt_version == 1:
        raise NotImplementedError
        
    
    elif prompt_version == 2:
        if not cot: raise NotImplementedError

        if qa_task == 'cq':
            
            messages += [
            {'role': 'user',
            'content': f'''I will provide you with a complex question. Give me its answer by breaking it down and reasoning through it before giving me the final answer. Give me the intermediate and final answers surrounded by these markers: "{ans_markers["start"]}" and "{ans_markers["end"]}".'''},
            {'role': 'assistant',
            'content': f'''I understand the instructions and I will answer the complex question.'''},]
        
        elif qa_task == 'sq':

            messages += [
            {'role': 'user',
            'content': f'''I will provide you with a complex question. We will obtain the answer to it by answering a sequence of its decomposed sub-questions. At each turn, give me the intermediate answer to that sub-question. It is very important to always attempt every sub-question and give a clear answer to it; if you are not sure, just give an answer that is to the best of your knowledge. Give only the answer and make sure to surround it with these markers: "{ans_markers["start"]}" and "{ans_markers["end"]}". Immediately start your reply with the answer and then stop immediately -- do not say anything else, do not explain, do not hedge and do not ask for more information.'''},
            {'role': 'assistant',
            'content': f'''I understand the instructions and I will answer the complex question by answering the sequence of decomposed sub-questions.'''},]


    return messages