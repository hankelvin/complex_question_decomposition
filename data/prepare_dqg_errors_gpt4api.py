from openai import OpenAI
import time, json, os

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

if __name__ == '__main__':
    from openai import OpenAI
    import time, json, os, argparse, tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'breakhigh', 
                        help = 'Dataset to process'  )
    parser.add_argument('--bsz', type = int, default = 75, 
                        help = 'size of each batch'  )
    parser.add_argument('--save_dir', type = str, default = '03_decompqg_errors/outputs', 
                        help = 'Directory to save outputs'  )
    args = parser.parse_args()

    with open('../llm_inference/token_openai_gpt4o.txt', encoding = 'utf-8') as f: 
        api_key = f.read()
    client = OpenAI(api_key = api_key)

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    DS_MAP = {'musique':        '03_decompqg_errors/01_musique_error_prompts.jsonl',
              'breakhigh':      '03_decompqg_errors/01_breakhigh_error_prompts.jsonl',
              '2wikimultihop':  '03_decompqg_errors/01_2wikimultihop_error_prompts.jsonl', }

    input_fp = DS_MAP[args.dataset]
    with open(input_fp, encoding='utf-8') as f:
        lines = [json.loads(l) for l in f]

    num_batches = len(lines)//args.bsz 
    save_paths  = []
    batches     = list(range(num_batches))
    for __, bn in enumerate(tqdm.tqdm(batches)):
        batch = lines[bn*args.bsz : (bn+1)*args.bsz ]
        
        file_response = do_one_batch(batch, client, args)
        save_path = args.save_dir + f'/gpt4o_outputs_{args.dataset}_bn{bn}.txt'
        
        with open(save_path, encoding='utf-8', mode = 'w+') as f: 
            f.write(file_response.text)
        save_paths.append(save_path)

        break

    save_fp      = f'03_decompqg_errors/outputs/01_CONSOLIDATED_gpt4o_outputs_{args.dataset}.jsonl'
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
