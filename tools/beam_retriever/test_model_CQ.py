import json, os
from retrieval.datasets import make_2wiki_musique_like
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from qa.reader_model import Reader
from retrieval.retriever_model import Retriever
### CHANGE START ###
from utils.utils_test_model import (give_qa_pred_filename, give_retr_pred_filename, normalize_answer, 
                                    load_saved, calculate_em_f1, update_sp, compute_exact, compute_f1)
import time
### CHANGE END ###
from tqdm import tqdm
import argparse

# SYNALP = ''
SYNALP = 'synalp_me/'

### CHANGE START ###
CHECKPOINT_HOLDER = {
    'reader':{
    # 1: '04-17-2024/musique_reader_deberta_large-seed42-bsz4-fp16True-lr5e-06-decay0.0-warm0.1-valbsz32',
    1: '10-02-2024/musique_reader_deberta_large-seed42-bsz6-fp16True-lr5e-06-decay0.0-warm0.1-valbsz32-singlehop-sq_all_pararagraphs',
    2: '09-07-2024/2wiki_reader_deberta_large-seed42-bsz6-fp16True-lr5e-06-decay0.0-warm0.1-valbsz32-singlehop',
    3: '09-06-2024/hotpot_reader_deberta_large-seed42-bsz8-fp16True-lr5e-06-decay0.0-warm0.1-valbsz32',
    },
    'retriever':{
    # 1: '04-17-2024/deberta_use_two_classier_musique_beam_size2-seed42-bsz8-fp16True-lr2e-05-decay0.0-warm0.1-valbsz1',
    1: '09-21-2024/deberta_use_two_classier_musique_beam_size2-seed42-bsz32-fp16True-lr2e-05-decay0.0-warm0.1-valbsz1',
    2: '09-22-2024/deberta_use_two_classier_2wiki_beam_size2-seed42-bsz32-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1',
    3: '09-06-2024/deberta_use_two_classier_hotpot_beam_size2-seed42-bsz32-fp16True-lr2e-05-decay0.0-warm0.1-valbsz1',
    }}
CHECKPOINT_DP   = f'/home/khan/{SYNALP}fathom/tools/beam_retriever/output/'
DATASET_TO_CODE =  {'musique': 1, '2wiki': 2, 'hotpot': 3}
### CHANGE END ###

### CHANGE START ###
def main(args):
    if not args.cross_domain: 
        assert f'{args.model_type}' in args.test_ckpt_path, args.test_ckpt_path
    else: 
        assert f'{args.dataset_type}'    not in args.test_ckpt_path, args.test_ckpt_path
    
    test_raw_data = load_data(args) # add tokenizer below

    args.ds_type = args.dataset_type
    if args.use_gold_passages:
        retr_json = give_gold_passages_idxes(args, test_raw_data)
    else: 
        retr_json = get_retr_output(args, test_raw_data)
    get_reader_qa_output(args, retr_json, test_raw_data)

def load_data(args, hotpot_dev=False):
    DATA_DP = '/home/khan/fathom/data'

    if   args.dataset_type == 'musique':
        args.test_file_path = f"{DATA_DP}/musique/musique_ans_v1.0_{'dev' if args.is_dev else 'test'}.jsonl"
    elif args.dataset_type == '2wiki':
        args.test_file_path = f"{DATA_DP}/01_unified_format/UNIFIED_2wikimultihop_{'validation' if args.is_dev else 'test'}.jsonl"
    elif args.dataset_type == 'hotpot':
        hotpot_str = 'dev_distractor' if args.is_dev else 'test_fullwiki'
        if hotpot_dev: hotpot_str = 'dev_distractor'
        args.test_file_path = f"{DATA_DP}/hotpotqa/hotpot_{hotpot_str}_v1.json"
    else: raise NotImplementedError

    if args.dataset_type == 'hotpot':
        with open(args.test_file_path, 'r') as f:
            test_raw_data = json.load(f)
    ### CHANGE START ###
    elif args.dataset_type == '2wiki':    
        with open(args.test_file_path, 'r') as f:
            test_raw_data = [json.loads(l) for l in f]
        # put CQ for 2wiki into the same format as musique (so we can reuse musique's code)
        test_raw_data = make_2wiki_musique_like(test_raw_data, is_test = not args.is_dev)
    ### CHANGE END ###
    elif args.dataset_type == 'musique': 
        with open(args.test_file_path, encoding = 'utf-8') as f:
            test_data = f.readlines()
        test_raw_data = [json.loads(item) for item in test_data]

    return test_raw_data

def get_retr_output(args, test_raw_data):
    ds_type = args.ds_type
    is_dev  = args.is_dev
    beam_size = args.beam_size
    ### CHANGE END ###

    retr_dic = {}
    ### CHANGE START ###
    re_tokenizer_path = 'microsoft/deberta-v3-base'
    re_model_path = re_tokenizer_path
    # follow setting in beam retriever paper and use checkpoint for num epochs trained till
    # §4.2 in https://arxiv.org/pdf/2308.08973
    re_checkpoint = CHECKPOINT_DP + CHECKPOINT_HOLDER['retriever'][args.model_key]  + '/checkpoint_last.pt'# + '/checkpoint_best.pt'
    pred_filename = give_retr_pred_filename(args, is_dev)
    ### CHANGE END ###

    re_max_seq_len = args.re_max_seq_len
    mean_passage_len = 250 if ds_type == 'hotpot' else 120 
    ### CHANGE START ###
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ### CHANGE END ###
    tokenizer = AutoTokenizer.from_pretrained(re_tokenizer_path)
    config = AutoConfig.from_pretrained(re_tokenizer_path)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    re_model = Retriever(config, re_model_path, encoder_class = AutoModel, 
                         mean_passage_len = mean_passage_len, beam_size = beam_size, 
                         gradient_checkpointing = True)
    re_model = load_saved(re_model, re_checkpoint, exact = False if args.cross_domain else True)
    re_model = re_model.to(device)
    re_model.eval()
    if is_dev:
        em_tot, f1_tot = [], []
        time_tot_cq = []
        retr_scores_cq = {}
    # get tensors
    for __, sample in enumerate(tqdm(test_raw_data, desc = "RE Predicting:")):
        if args.do_trial and __ >= 5: break
        start_time = time.time()
        try: question = sample['question']
        except: 
            print('\t🚨 MISSING QUESTION', __)
            print('\t🚨 ', sample.keys())
            raise KeyError

        if question.endswith("?"):
            question = question[:-1]
        idnum = sample['id'] if ds_type in ['musique', '2wiki'] else sample['_id']
        q_codes = tokenizer.encode(question, add_special_tokens = False, return_tensors = "pt", 
                                   truncation = True, max_length = re_max_seq_len).squeeze(0)
        c_codes = []
        if is_dev:
            sf_idx = []
            sp_title_set = set()
        if ds_type in ['hotpot']:
            for idx, (title, sentences) in enumerate(sample['context']):
                if is_dev:
                    for sup in sample['supporting_facts']:
                        sp_title_set.add(sup[0])
                    if title in sp_title_set:
                        sf_idx.append(idx)
                l = title + "".join(sentences)
                encoding = tokenizer.encode(l, add_special_tokens = False, return_tensors = "pt", truncation = True, 
                                             max_length = re_max_seq_len-q_codes.shape[-1]).squeeze(0)
                encoding = encoding.to(device)
                c_codes.append(encoding)
        elif ds_type in ['musique', '2wiki']:
            nhop = 0 # used for 2wiki
            for i, para in enumerate(sample['paragraphs']):
                if is_dev:
                    if para['is_supporting']:
                        sf_idx.append(i)
                        nhop += 1
                l = para['title'] + '.' + para['paragraph_text']
                encoding = tokenizer.encode(l, add_special_tokens = False, return_tensors = "pt", truncation = True, 
                                            max_length = re_max_seq_len-q_codes.shape[-1]).squeeze(0)
                encoding = encoding.to(device)
                c_codes.append(encoding)
        q_codes = q_codes.to(device)
        q_codes_input = [q_codes]
        c_codes_input = [c_codes]
        
        if   ds_type in ['musique']:hop = int(idnum[0]) 
        elif ds_type in ['2wiki']:  hop = nhop
        elif ds_type in ['hotpot']: hop = 2
        
        # if ds_type == '2wiki' and sample['type'] == 'bridge_comparison':
        #     hop = 4
        with torch.no_grad():
            current_preds = re_model(q_codes_input, c_codes_input, 
                                     [] if not is_dev else sf_idx, hop = hop)['current_preds']
        retr_dic[idnum] = current_preds[0]
        if is_dev:
            f1, em = calculate_em_f1(current_preds[0], sf_idx)
            em_tot.append(em)
            f1_tot.append(f1)
            time_tot_cq.append(round(time.time() - start_time, 4))
            retr_scores_cq[idnum] = {'em': em, 'f1': f1, 'time': time_tot_cq[-1]}
    if is_dev:
        print(f"em:{sum(em_tot) / len(em_tot)}, f1:{sum(f1_tot) / len(f1_tot)}")
    with open(pred_filename, "w", encoding = "utf-8") as f:
        json.dump(retr_dic, f, ensure_ascii = False, indent = 4)

    if is_dev: 
        pf = pred_filename.replace('pred_', 'scores_CQ_') # e.g. 'pred_.....jsonl'
        print('\t\t CHECK (RETR) ... save filename:', pred_filename)
        with open(pf, "w", encoding = "utf-8") as f:
            json.dump(retr_scores_cq, f, ensure_ascii = False, indent = 4)
    print(f"retr evaluation finished!")
    torch.cuda.empty_cache()
    return retr_dic

def give_gold_passages_idxes(args, test_raw_data):
    print('Using gold passages')
    ds_type = args.ds_type
    is_dev  = args.is_dev
    if not is_dev: raise ValueError("Gold passages are only available for development set.")
    pred_filename = give_retr_pred_filename(args, is_dev)

    retr_dic_gold = {}
    for __, sample in enumerate(tqdm(test_raw_data, desc = "RE Collecting Gold:")):
        idnum = sample['id'] if ds_type in ['musique', '2wiki'] else sample['_id']
        if is_dev:
            sf_idx = []
            sp_title_set = set()
        if ds_type in ['hotpot']:
            for idx, (title, sentences) in enumerate(sample['context']):
                if is_dev:
                    for sup in sample['supporting_facts']:
                        sp_title_set.add(sup[0])
                    if title in sp_title_set:
                        sf_idx.append(idx)
        elif ds_type in ['musique', '2wiki']:
            # musique
            for i, para in enumerate(sample['paragraphs']):
                if is_dev:
                    if para['is_supporting']:
                        sf_idx.append(i)
        retr_dic_gold[idnum] = sf_idx
    
    with open(pred_filename, "w", encoding = "utf-8") as f:
        json.dump(retr_dic_gold, f, ensure_ascii = False, indent = 4)

    return retr_dic_gold

def merge_find_ans(start_logits, end_logits, ids, punc_token_list, topk = 5, max_ans_len = 20):
    def is_too_long(span_id, punc_token_list):
        for punc_token_id in punc_token_list:
            if punc_token_id in span_id:
                return True
        return False
    start_candidate_val, start_candidate_idx = start_logits.topk(topk, dim = -1)
    end_candidate_val, end_candidate_idx = end_logits.topk(topk, dim = -1)
    pointer_s, pointer_e = 0, 0
    start = start_candidate_idx[pointer_s].item()
    end = end_candidate_idx[pointer_e].item()
    span_id = ids[start: end + 1]
    while start > end or (end - start) > max_ans_len or is_too_long(span_id, punc_token_list):
        if start_candidate_val[pointer_s] > end_candidate_val[pointer_e]:
            pointer_e += 1
        else:
            pointer_s += 1
        if pointer_s >= topk or pointer_e >= topk:
            break
        start = start_candidate_idx[pointer_s].item()
        end = end_candidate_idx[pointer_e].item()
        span_id = ids[start: end + 1]
    return span_id

### CHANGE START ###
def get_reader_qa_output(args, retr_pred_dic, test_raw_data):
    is_dev = args.is_dev 
    ds_type = args.dataset_type
    answer_merge = args.answer_merge
    topk = args.topk 
    qa_tokenizer_path = "microsoft/deberta-v3-large"
    ### CHANGE END ###
    qa_model_path = qa_tokenizer_path
    if   args.model_type  == 'musique':    key = 1
    elif args.model_type  == '2wiki':      key = 2
    elif args.model_type  == 'hotpot':     key = 3
    qa_checkpoint = CHECKPOINT_DP + CHECKPOINT_HOLDER['reader'][key] + '/checkpoint_last.pt'

    pred_filename = give_qa_pred_filename(args, ds_type, is_dev)
    ### CHANGE END ### 

    qa_max_seq_len = args.qa_max_seq_len
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = AutoConfig.from_pretrained(qa_model_path)
    config.max_position_embeddings = qa_max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(qa_tokenizer_path)

    SEP = "</e>"
    DOC = "</d>"
    if args.model_type == 'hotpot': # else mismatch in embeddings
        tokenizer.add_tokens([SEP, DOC])
    if ds_type == 'hotpot':
        sp_pred  = {}
        ans_pred = {}
        SEP_id = tokenizer.convert_tokens_to_ids(SEP)
        DOC_id = tokenizer.convert_tokens_to_ids(DOC)
    qa_model = Reader(config, qa_model_path, task_type = args.dataset_type,
                      len_tokenizer = len(tokenizer) if args.model_type == 'hotpot' else 0, )
    qa_model = load_saved(qa_model, qa_checkpoint, exact = False if args.cross_domain else True)
    qa_model = qa_model.to(device)
    qa_model.eval()
    
    pred_list = []
    ### CHANGE START ###
    from collections import defaultdict
    scores_dict = defaultdict(list)
    ### CHANGE END ###
    if is_dev:
        em_tot, f1_tot = [], []
        if ds_type == 'hotpot':
            sp_em_tot, sp_f1_tot = [], []
    # get tensors
    for __, sample in enumerate(tqdm(test_raw_data, desc="QA Predicting:")):
        if args.do_trial and __ >= 5: break
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        idnum = sample['id'] if ds_type in ['musique', '2wiki'] else sample['_id']
        q_codes = tokenizer.encode(question, add_special_tokens = False, 
                                   truncation = True, max_length = qa_max_seq_len)
        sp_list = retr_pred_dic[idnum]
        idx2title = {}
        c_codes = []
        if ds_type in ['hotpot']:
            # hotpot format
            sts2title = {}
            sts2idx = {}
            sts_idx = 0
            sentence_label = []
            if is_dev:
                sp_title_set = {}
                for sup in sample['supporting_facts']:
                    if sup[0] not in sp_title_set:
                        sp_title_set[sup[0]] = []
                    sp_title_set[sup[0]].append(sup[1])
            for idx, (title, sentences) in enumerate(sample['context']):
                if idx in sp_list:
                    idx2title[idx] = title
                    l = DOC + " " + title
                    for idx2, c in enumerate(sentences):
                        l += (SEP + " " + c)
                    encoding = tokenizer.encode(l, add_special_tokens = False, truncation = True, 
                                                max_length = qa_max_seq_len-len(q_codes))
                    c_codes.append(encoding)
        elif ds_type in ['musique', '2wiki']:
            # musique
            for i, para in enumerate(sample['paragraphs']):
                if i in sp_list:
                    l = para['title'] + '.' + para['paragraph_text']
                    encoding = tokenizer.encode(l, add_special_tokens = False, truncation = True, 
                                                max_length = qa_max_seq_len-len(q_codes))
                    c_codes.append(encoding)
        total_len = len(q_codes) + sum([len(item) for item in c_codes])
        context_ids = [tokenizer.cls_token_id] + q_codes
        avg_len = (qa_max_seq_len - 2 - len(q_codes)) // len(c_codes)
        
        if ds_type == 'hotpot':
            sp_list.sort() # only hotpot format, for sp prediction, align sentence order and passages order
        for idx, item in enumerate(c_codes):
            if total_len > qa_max_seq_len - 2:
                # 可能把答案截断
                item = item[:avg_len]
            if ds_type == 'hotpot':
                sts_idx_local = 0
                for i in range(len(item)):
                    if item[i] == SEP_id:
                        sts2title[sts_idx] = idx2title[sp_list[idx]]
                        sts2idx[sts_idx] = sts_idx_local
                        sts_idx += 1
                        sts_idx_local += 1
            context_ids.extend(item)
        context_ids = context_ids[:qa_max_seq_len - 1] + [tokenizer.sep_token_id]
        pred_answer = None
        input_ids = torch.tensor(context_ids, dtype = torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones([1, len(context_ids)], dtype = torch.long, device=device)
        if ds_type == 'hotpot':
            SEP_index = []
            for i in range(len(context_ids)):
                if context_ids[i] == SEP_id:
                    SEP_index.append(i)
            SEP_index = torch.LongTensor([SEP_index]).to(device)

            ### MISSING... added from:
            # MHReader.__getitem__
            # reader_mhop_collate in dataset.py
            answer_type = 2
            answer = sample['answer']
            if answer == "no": answer_type = 0
            elif answer == "yes": answer_type = 1
            answer_type = torch.tensor([answer_type], dtype = torch.long)
            ##############################

            with torch.no_grad():
                outputs = qa_model(input_ids, attention_mask, sentence_index = SEP_index[0],
                                   answer_type = answer_type)
            sentence_select = torch.argmax(outputs['sentence_select'], dim=-1)
            assert sentence_select.shape[-1] == len(sts2idx)
            output_answer_type = outputs['output_answer_type']
            ans_type = torch.argmax(output_answer_type).item()
            if ans_type == 0:
                pred_answer = 'no'
            elif ans_type == 1:
                pred_answer = 'yes'
            sp = []
            sts_idx = 0
            for s in range(len(sentence_select)):
                if sentence_select[s] == 1:
                    sp.append([sts2title[s], sts2idx[s]])
            sp_pred[idnum] = sp

        else:
            with torch.no_grad():
                outputs = qa_model(input_ids, attention_mask)
        start_logits = outputs['start_qa_logits'][0]
        end_logits = outputs['end_qa_logits'][0]
        input_ids = input_ids[0]

        if pred_answer is None:
            if answer_merge:
                punc_token_list = tokenizer.convert_tokens_to_ids(['[CLS]', '?'])
                if ds_type == 'hotpot':
                    punc_token_list.extend([SEP_id, DOC_id])
                span_id = merge_find_ans(start_logits, end_logits, 
                                         input_ids.tolist(), punc_token_list, topk = topk)
                pred_answer = tokenizer.decode(span_id)
            else:
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

                answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
                pred_answer = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(answer_tokens)
                )

        ### CHANGE START ###
        if ds_type == 'hotpot':
            ans_pred[idnum] = pred_answer

        pred_list.append({'id': idnum, 'cq': sample['question'], 
                            'predicted_answer': pred_answer, 
                            'gold_answer': sample.get('answer', None),
                            'subquestions': None, 
                            'pred_sq_as': None, 
                            'sp_list_sq': None,
                            'predicted_support_idxs': sp_list, 
                            'predicted_answerable': True})
        
        pred_answer = normalize_answer(pred_answer)
        ### CHANGE END ###
        if is_dev:
            ground_truth_answer = sample['answer']
            ground_truth_answer = normalize_answer(ground_truth_answer)

            em = compute_exact(ground_truth_answer, pred_answer)
            f1 = compute_f1(ground_truth_answer, pred_answer)
            em_tot.append(em)
            f1_tot.append(f1)
            ### CHANGE START ###
            scores_dict['qa_em'].append(em)
            scores_dict['qa_f1'].append(f1)
            ### CHANGE END ###

            if ds_type == 'hotpot':
                sp_em, sp_f1 = update_sp(sp, sample['supporting_facts'])
                sp_em_tot.append(sp_em)
                sp_f1_tot.append(sp_f1)
                ### CHANGE START 
                # print(f"sp em:{sum(sp_em_tot) / len(sp_em_tot)}, sp f1:{sum(sp_f1_tot) / len(sp_f1_tot)}")
                scores_dict['sp_em'].append(sp_em)
                scores_dict['sp_f1'].append(sp_f1)
                ### CHANGE END ###
    if is_dev:
        print(f"em:{sum(em_tot) / len(em_tot)}, f1:{sum(f1_tot) / len(f1_tot)}")
        if ds_type == 'hotpot':
            print(f"sp em:{sum(sp_em_tot) / len(sp_em_tot)}, sp f1:{sum(sp_f1_tot) / len(sp_f1_tot)}")
        
        ### CHANGE START ###
        scores_file_name = pred_filename.replace('sorted_pred', 'sorted_scores').replace('jsonl', 'json')
        with open(scores_file_name, 'w+', encoding='utf-8') as f:
            json.dump(scores_dict, f)
        print(f"📘📘Scores saved to {scores_file_name}")
        ### CHANGE END ###

    ### CHANGE START ###
    # save for all datasets (not just for musique)
    if ds_type in ['musique', '2wiki', 'hotpot']:
        with open(pred_filename, "w", encoding = 'utf-8') as f:
            ### CHANGE END ###
            for data in pred_list:
                f.write(json.dumps(data)+'\n')
    if ds_type in ['hotpot']:
        with open(pred_filename.replace('.json', '_spfact.json'), "w", encoding="utf-8") as f:
            json.dump({"answer": ans_pred, "sp": sp_pred}, f)
    print(f"evaluation finished!")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    ### CHANGE START ###
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_dev',         type = str,    default = 'True')
    parser.add_argument('--model_type',     type = str,     default = 'musique')
    parser.add_argument('--dataset_type',   type = str,     default = 'musique', 
                        help = 'the dataset to test on (i.e. may not be the one model trained on).')
    parser.add_argument('--use_gold_passages', type = str,     default = 'False')
    parser.add_argument('--beam_size',      type = int,     default = 2)
    parser.add_argument('--answer_merge',   type = str,    default = 'True')
    parser.add_argument('--topk',           type = int,     default = 10)
    parser.add_argument('--do_trial',       type = str,    default = 'False')
    parser.add_argument('--filter_hotpot',  type = str,    default = 'True')
    parser.add_argument('--test_ckpt_path', type = str,     default = '../../checkpoints/{}')

    args = parser.parse_args()

    for key in ['is_dev', 'use_gold_passages', 'answer_merge', 'do_trial', 'filter_hotpot']:
        setattr(args, key, eval(getattr(args, key)))

    args.do_single_hop = False
    args.qa_max_seq_len = 1024
    args.re_max_seq_len = 512
    if args.dataset_type == 'hotpot':
        assert args.is_dev == True
        args.filter_hotpot = True

    args.cross_domain = False 
    if args.dataset_type != args.model_type: args.cross_domain = True

    args.model_key          = DATASET_TO_CODE[args.model_type]
    args.test_ckpt_path     = os.path.join(CHECKPOINT_DP, CHECKPOINT_HOLDER['retriever'][args.model_key])
    cross_str               = '_crossdomain' if args.cross_domain else ''
    args.checkpoint_name    = os.path.basename(args.test_ckpt_path) + cross_str
    args.cqsq_pfx = 'CQ_'

    print('🟥🟥USING DATASET:', args.dataset_type, 'CROSS_DOMAIN:', args.cross_domain, 'CKPT:', args.checkpoint_name)
    
    main(args)