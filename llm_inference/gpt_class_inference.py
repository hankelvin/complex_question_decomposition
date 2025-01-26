from openai import OpenAI
import time, json, os, random, copy
import pandas as pd
from task_utils_decomp_qg import prepare_decomp_qg_prompt
import sys
sys.path.append('..')
from evaluation.evaluate_dqg import get_decomposition


########################################################################
MODEL = "gpt-4o-2024-08-06"
BATCH_LINE = {"custom_id": None, "method": "POST", "url": "/v1/chat/completions", 
            "body": {"model": MODEL, "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": None}], "max_tokens": 500}}
################################################################


def load_sample_data(args, keep_idxes = None):
    
    random.seed(54506)
    print('WORKING on dataset', args.dataset)
    fp = f'data/01_unified_format/UNIFIED_{args.dataset}_{args.split}.jsonl'
    with open(fp, encoding='utf-8') as f: lines = [json.loads(line) for line in f]

    if keep_idxes is not None:
        print(f'BEFORE: keep only {len(keep_idxes)} of {len(lines)} lines')
        lines = [line for line in lines if line['id'] in keep_idxes]
        print('AFTER:  left with:', len(lines))

    elif args.dataset == 'breakhigh':
        for l in lines: l['operators'] = l['original_info'].pop('operators')

    HOLDER = []
    for i, line in enumerate(lines):
        idx = line['id']
        
        gpt_line = copy.deepcopy(BATCH_LINE)
        gpt_line['custom_id'] = idx
        
        args.input_file = [args.dataset, args.split]
        messages, tgt, cq = prepare_decomp_qg_prompt(args, line, repro = False)

        prompt = ''
        msg_num = 0
        for m in messages: 
            if m['role'] in ['user', 'assistant']: 
                if msg_num == 1: 
                    prompt = prompt + '''Return the results in json_schema format with the following key: "decomposed_questions", whose value is a single containing the decomposed questions separated by a ' ;' between them.\n---\n'''
                if m['role'] == 'assistant': 
                    if m['content'].startswith('I understand the instructions'):
                        continue
                    prompt += f"{m['content']}"
                    prompt += f"\n\n---\n"
                else: 
                    prompt += f"{m['content']}\n"
                msg_num += 1
        
        gpt_line['body']['messages'][1]['content'] = prompt
        if i == 0: print('PROMPT', prompt)
        HOLDER.append(gpt_line)

    return HOLDER, lines


def do_one_batch(batch, client, args):
    tmp_fp = f'{args.save_dir}/tmp_gpt.jsonl'
    with open(tmp_fp, encoding='utf-8', mode = 'w+') as f:
        for l in batch: f.write(json.dumps(l)+'\n')
    
    batch_input_file = client.files.create(file = open(tmp_fp, 'rb'), purpose = 'batch')
    batch_in = client.batches.create(input_file_id = batch_input_file.id,
                                    endpoint = '/v1/chat/completions',
                                    completion_window = '24h',
                                    metadata = {'description': 'errors introduction'})
    
    output_file = None
    while output_file is None:
        output_file = client.batches.retrieve(batch_in.id).output_file_id
        if output_file: break
        time.sleep(15)
    file_response = client.files.content(output_file)
    
    return file_response

def consolidate_dqg(args, save_paths, lines_label, lines_prediction):
    failed, seen = set(), set()
    for __, fp in enumerate(tqdm.tqdm(save_paths)):
        with open(fp, encoding='utf-8') as f2:
            for l in f2: 
                line = json.loads(l.strip())
                for ci, c in enumerate(line['response']['body']['choices']):
                    idx = line['custom_id']
                    try: 
                        ccc = json.loads(line['response']['body']['choices'][ci]['message']['content'])
                    
                        decomposition = ccc['decomposed_questions']
                        decomposition = [f"{args.qpos_tokens[i]} {q.strip()} " for i, q in \
                                         enumerate(decomposition.split(';')) if q.strip()]
                        decomposition = ' '.join(decomposition)
                        decomposition, ops_gen = get_decomposition(decomposition)
                        if args.dqg_lower: decomposition = decomposition.lower()
                        
                        pos = lines_prediction['question_id'].index(idx)
                        lines_prediction['decomposition'][pos] = decomposition.strip()
                        lines_prediction['operators_gen'][pos] = ops_gen
                        seen.add(idx)
                    except: 
                        failed.add(idx)
    
    return failed, seen, lines_label, lines_prediction

if __name__ == '__main__':
    from openai import OpenAI
    import time, json, os, argparse, tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'breakhigh', 
                        help = 'Dataset to process (breakhigh, musique)'  )
    parser.add_argument('--split', type = str, default = 'validation', 
                        help = 'Split to process (train, dev, test)'  )
    parser.add_argument('--task', type = str, default = 'decomp_qg', 
                        help = 'task to run (decomp_qg, decomp_qa, cq_qa)'  )
    parser.add_argument('--bsz', type = int, default = 500, 
                        help = 'size of each batch'  )
    parser.add_argument('--dqg_lower', type = bool, default = True, 
                        help = 'Whether to lowercase gold and preds'  )
    parser.add_argument('--save_dir', type = str, default = 'results/decomp_qg/gpt4o_{}_{}_{}', 
                        help = 'Directory to save outputs'  )
    args = parser.parse_args()

    args.model_name = MODEL
    args.prompt_version = 2
    args.decomp_qg_args = {'n_shot': 0, 'cot': False}
    args.qpos_tokens = [f'[SQ{i+1}]' for i in range(50)]
    if args.task == 'decomp_qg':
        args.decomp_qg_args = {'n_shot': 5, 'cot': True} 
    args.user_prompt = "Complex question: "
    args.asst_prompt = "Decomposed sub-questions: " if not args.decomp_qg_args['cot'] else ''

    with open('token_openai_gpt4o.txt', encoding = 'utf-8') as f: 
        api_key = f.read()
    client = OpenAI(api_key = api_key)

    args.save_dir = args.save_dir.format(args.task, args.dataset, args.split)
    if args.decomp_qg_args:
        if args.decomp_qg_args['cot'] == True: 
            if args.decomp_qg_args['n_shot'] != 5: 
                args.save_dir += f'_CoTnshot{args.decomp_qg_args["n_shot"]}'
        else: args.save_dir += f'_noCoT'
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)  
    
    HOLDER, lines = load_sample_data(args, keep_idxes = None)
    
    num_batches = len(HOLDER)//args.bsz 
    save_paths  = []
    batches     = list(range(num_batches))
    
    ################################################################
    for __, bn in enumerate(tqdm.tqdm(batches)):
        batch = HOLDER[bn*args.bsz : (bn+1)*args.bsz ]
        
        file_response = do_one_batch(batch, client, args)
        save_path = f'{args.save_dir}/gpt4o_outputs_bn{bn}.txt'
        
        with open(save_path, encoding='utf-8', mode = 'w+') as f: 
            f.write(file_response.text)
        save_paths.append(save_path)
    ################################################################
    
    if args.task == 'decomp_qg':
        all_idxes = [line['id'] for line in lines]
        lines_label = \
        {'question_id': [], 
        'question_text': [], 
        'decomposition': [], 
        'operators_tgt': []}
        for i, line in enumerate(lines):
            idx = line['id']
            lines_label['question_id'].append(idx)
            
            question_text = line['text']['qs'][0].strip()
            decomposition = ' ;'.join(line['decomp_qs']['text']['qs_var'])
            if args.dqg_lower: 
                question_text = question_text.lower()
                decomposition = decomposition.lower()
            
            lines_label['question_text'].append(question_text)
            lines_label['decomposition'].append(decomposition)
            if args.dataset == 'breakhigh':
                lines_label['operators_tgt'].append(line['operators'])
            else: 
                lines_label['operators_tgt'].append(None)

        assert len(set(lines_label['question_id'])) == len(lines_label['question_id'])
        
        lines_prediction = \
        {'question_id': lines_label['question_id'].copy(), 
        'decomposition': [None for _ in lines_label['question_id']], 
        'operators_gen': [None for _ in lines_label['question_id']]}
        
        failed, seen, lines_label, lines_prediction = consolidate_dqg(args, save_paths, 
                                                        lines_label, lines_prediction)
        
        failed.update(set(all_idxes).difference(seen))
        tries = 0
        redo_save_paths  = {}
        while failed and tries < 10: 
            redo_save_paths[tries] = []
            HOLDER, lines = load_sample_data(args, keep_idxes = failed)

            num_batches = len(HOLDER)//args.bsz 
            if len(HOLDER) > 0 and num_batches == 0: 
                num_batches = 1
            
            batches     = list(range(num_batches))
            
            for __, bn in enumerate(tqdm.tqdm(batches)):
                batch = HOLDER[bn*args.bsz : (bn+1)*args.bsz ]
                
                file_response = do_one_batch(batch, client, args)
                save_path = f'{args.save_dir}/gpt4o_outputs_redo{tries}-bn{bn}.txt'
                
                with open(save_path, encoding='utf-8', mode = 'w+') as f: 
                    f.write(file_response.text)
                redo_save_paths[tries].append(save_path)
            
            failed, redo_seen, lines_label, lines_prediction = consolidate_dqg(
                                                            args, redo_save_paths[tries], 
                                                            lines_label, lines_prediction)
            seen.update(redo_seen)
            failed.update(set(all_idxes).difference(seen))
            tries += 1

        print(f'Failed to process lines:', failed)
        df_gold = pd.DataFrame(lines_label)
        df_pred = pd.DataFrame(lines_prediction)
        if not args.dqg_lower:
            dqg_lower_str = '/lowercase' 
            if not os.path.exists(f'{args.save_dir}{dqg_lower_str}'): 
                os.makedirs(f'{args.save_dir}{dqg_lower_str}')
        else: 
            dqg_lower_str = ''
        df_gold.to_csv(f'{args.save_dir}{dqg_lower_str}/text_labels.csv', index = False)
        df_pred.to_csv(f'{args.save_dir}{dqg_lower_str}/text_predictions.csv', index = False)

    else: 
        save_fp      = f'{args.save_dir}/01_CONSOLIDATED_gpt4o_outputs.jsonl'
        with open(save_fp, encoding = 'utf-8', mode = 'w+') as f1:
            for fp in save_paths:
                with open(fp, encoding='utf-8') as f2:
                    for l in f2: 
                        passed = False
                        line = json.loads(l.strip())
                        for ci, c in enumerate(line['response']['body']['choices']):
                            try: 
                                ccc = json.loads(line['response']['body']['choices'][ci]['message']['content'])
                                passed = True
                            except: 
                                continue
                            line['response']['body']['choices'][ci]['message']['content'] = ccc
                        if passed: f1.write(json.dumps(line)+'\n')
