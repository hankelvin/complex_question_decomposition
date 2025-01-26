'''
modifying beam retriever test script to take decomposed subquestions of CQs
answering will be done successively for each subquestion
final SQ answer will be compared against the original CQ answer
'''

import json, re, tempfile
import torch
from transformers import AutoConfig, AutoTokenizer

from qa.reader_model import Reader
### CHANGE START ###
from utils.utils_test_model import (give_qa_pred_filename, normalize_answer, 
                                    load_saved, update_sp, compute_exact, compute_f1)
from test_model_CQ import (CHECKPOINT_HOLDER, CHECKPOINT_DP, DATASET_TO_CODE, load_data, merge_find_ans,
                           get_retr_output, give_gold_passages_idxes)
### CHANGE END ###
from tqdm import tqdm
import argparse

### CHANGE START ###
sup_dp          = f'results/dqg_dqa_ouputs'
llm_single_dp   = f'results/decomp_qg'
llm_top1_sft_dp = f'results/dqg_dqa_ouputs'
DECOMP_RESULT_PATH = {
    'supervised': {
        'musique':      f'{sup_dp}/dqg_musique/dqg_musique_flan-t5-large_fp32',},
    'gpt4o': {
        'musique':      f'{llm_single_dp}/gpt4o_decomp_qg_musique_{{0}}',},
    'llm_single_zeroshot': {
        'musique':      f'{llm_single_dp}/decomp_qg_n_shot-0_CoT_musique_{{0}}_{{1}}_modelline-2_prmptv2'},
    'llm_single': {
        'musique':      f'{llm_single_dp}/decomp_qg_n_shot-5_CoT_musique_{{0}}_{{1}}_modelline-2_prmptv2'},
    'llm_top1': {
        'musique':      f'{llm_single_dp}/decomp_qg_musique_{{0}}_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ge-ll-ph-qw_modelline-2_prmptv2/original_run0'},
    'llm_top1_sft': {
        'musique':      f'{llm_top1_sft_dp}/dqg_musique/dqg_musique_{{0}}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test'},
    'llm_top1_sft_rtfilt-1x': {             
        'musique':      f'{llm_top1_sft_dp}/dqg_musique/dqg_musique_{{0}}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-1x-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test'},
    'llm_stv_sft_rtfilt-2x': {
        'musique':      f'{llm_top1_sft_dp}/dqg_musique/dqg_musique_{{0}}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-2x-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test'},
    'llm_stv_sft_rtfilt-2x-upsamp': {
        'musique':      f'{llm_top1_sft_dp}/dqg_musique/dqg_musique_{{0}}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-2x-upsamp-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test'},
    }
### CHANGE END ###

### CHANGE START ###
def give_decomp_csv_path(args):
    if args.decomp_origin[0] == 'supervised':
        dqg_inference_model, dqg_inference_data = args.decomp_origin[1:]
        if dqg_inference_data != 'breakhigh': 
            assert args.dataset_type == dqg_inference_data
        else: 
            assert args.dataset_type == 'hotpot'
        
        decomp_csv_path = DECOMP_RESULT_PATH[args.decomp_origin[0]][dqg_inference_model]
        # swop to shortened 2wiki for supervised DQG handling
        if dqg_inference_model == '2wiki': dqg_inference_model = '2wikimultihop'
        if dqg_inference_data  == '2wiki': dqg_inference_data  = '2wikimultihop'

        c1 = dqg_inference_model != dqg_inference_data
        c2 = args.is_dev
        if c1 or c2:
            decomp_csv_path += '_test_only'
            if c2: decomp_csv_path += '-phase3_val_as_test'
            else:  decomp_csv_path += '-test'
        if c1: 
            decomp_csv_path += f'_cross_domain-{dqg_inference_data}'

    elif args.decomp_origin[0] == 'gpt4o': 
        dqg_inference_model, dqg_inference_data = args.decomp_origin[1:]
        decomp_csv_path = DECOMP_RESULT_PATH[args.decomp_origin[0]][dqg_inference_data]
        split_str = 'validation' if args.is_dev else 'test'
        decomp_csv_path = decomp_csv_path.replace('{0}', split_str)

    elif args.decomp_origin[0] in ['llm_single', 'llm_single_zeroshot']: 
        dqg_inference_model, dqg_inference_data = args.decomp_origin[1:]
        decomp_csv_path = DECOMP_RESULT_PATH[args.decomp_origin[0]][dqg_inference_data]
        split_str = 'validation' if args.is_dev else 'test'
        decomp_csv_path = decomp_csv_path.replace('{0}', split_str)
        decomp_csv_path = decomp_csv_path.replace('{1}', dqg_inference_model)

    elif args.decomp_origin[0] in ['llm_top1', 'llm_top1_sft', 
                                   'llm_top1_sft_rtfilt-1x', 'llm_stv_sft_rtfilt-2x',
                                   'llm_top1_sft_rtfilt-1x-upsamp', 'llm_stv_sft_rtfilt-2x-upsamp',]: 

        dqg_inference_model, dqg_inference_data = args.decomp_origin[1:]
        decomp_csv_path = DECOMP_RESULT_PATH[args.decomp_origin[0]][dqg_inference_data]
        split_str = 'validation' if args.is_dev else 'test'
        if args.decomp_origin[0] in ['llm_top1']:
            decomp_csv_path = decomp_csv_path.replace('{0}', split_str)
        else: 
            decomp_csv_path = decomp_csv_path.replace('{0}', dqg_inference_model)
        
    else: raise NotImplementedError

    return decomp_csv_path, dqg_inference_data

########## CLEAN UP: pad tokens and lingering operators (break) ##########
# same at evaluate_break_suite.py
def remove_special_tokens_lingering_operators(csv_filepath, pad_char = '<|pad|>', do_tmp = True):
    
    df = pd.read_csv(csv_filepath)
    
    if 'question_text' in df.columns:
        df['question_text'] = df['question_text'].apply(lambda x: x.replace(pad_char, ''))
    df['decomposition'] = df['decomposition'].apply(lambda x: x.replace(pad_char, ''))
    # ensure no remaining operators in sub-qs. issues encountered by LLMs (from GPT4o to cllms)
    df['decomposition'] = df['decomposition'].apply(lambda x: clean_lingering_operators_cot(x))

    tmp_csv_filepath = None
    if do_tmp:
        tmp_csv_filepath = tempfile.NamedTemporaryFile(delete=False).name
        df.to_csv(tmp_csv_filepath, index = False)

    return df, tmp_csv_filepath
    
def clean_lingering_operators_cot(decompositions_str, 
    cot_pattern = r'.+(the sequence of sub-questions should be: )(.+)'):

    cot_check = re.search(cot_pattern, decompositions_str, re.IGNORECASE)
    if cot_check and len(cot_check.groups()) > 1 and len(cot_check.group(2).strip()) > 0:
        decompositions_str = cot_check.group(2)

    ds = decompositions_str.split(';')
    ds = [detect_remove_lingering_operators(d) for d in ds if d.strip()]
    return " ;".join(ds)

def detect_remove_lingering_operators(decomp):
    for rep in re.findall(r'(\[.*?\])', decomp): 
        decomp = decomp.replace(rep, '')
    decomp = re.sub('\s{2,}', ' ', decomp).strip()
    return decomp
###########################################################################

def main(args):
    args.ds_type = args.dataset_type
    if not args.cross_domain: 
        assert f'{args.model_type}' in args.test_ckpt_path, args.test_ckpt_path
    else:
        assert f'{args.dataset_type}' not in args.test_ckpt_path, args.test_ckpt_path

    hotpot_dev = True if args.dataset_type == 'hotpot' else None
    test_raw_data = load_data(args, hotpot_dev) # add tokenizer below
    
    decomp_csv_path, dqg_inference_data = give_decomp_csv_path(args)
    
    pred_fp     = f"{decomp_csv_path}/text_predictions.csv"
    print('PREDICTIONS LOADED FROM:', pred_fp)
    pred, __    = remove_special_tokens_lingering_operators(pred_fp, do_tmp = False, 
                                                            cotnshot0 = '_CoTnshot0' in pred_fp)
    pred.columns = [col if col.startswith('question') else f'pred_{col}' for col in  pred.columns]
    
    gold_fp     = f"{decomp_csv_path}/text_labels.csv"
    print('GOLDS LOADED FROM:', pred_fp)
    gold, __    = remove_special_tokens_lingering_operators(gold_fp, do_tmp = False,
                                                            cotnshot0 = '_CoTnshot0' in gold_fp)
    gold.columns = [col if col.startswith('question') else f'gold_{col}' for col in  gold.columns ]
    
    assert len(pred) == len(gold)
    if 'question_id' in pred.columns and 'question_id' in gold.columns:
        assert pred.question_id.tolist() == gold.question_id.tolist()
        pred.drop(columns=['question_id'], inplace = True)
    
    args.sq_pred_data = pd.concat([gold, pred], axis = 1)
    if dqg_inference_data == 'breakhigh':
        # keep only the hotpot instances
        args.sq_pred_data = args.sq_pred_data[args.sq_pred_data['question_id'].str.contains('HOTPOT_')]
        args.sq_pred_data['question_id'] = args.sq_pred_data['question_id'].apply(lambda x: x.replace('HOTPOT_dev_', ''))
    
    ### CHANGE START ###
    if   args.dataset_type in ['musique', '2wiki']: id_key = 'id' 
    elif args.dataset_type == 'hotpot':             id_key = '_id' 
    drop_positions = []
    for i, sample in enumerate(test_raw_data):
        # 1. get id num
        cq_id = sample[id_key]
        # for breakhigh, drop hotpot instances which are not in the breakhigh decompositions
        if dqg_inference_data == 'breakhigh' and cq_id not in args.sq_pred_data['question_id'].tolist():
            drop_positions.append(i)
            continue

        # 2. get SQs
        subquestions = args.sq_pred_data[args.sq_pred_data['question_id'] == cq_id]['pred_decomposition'].tolist()
        assert len(subquestions) == 1, (len(subquestions), subquestions, cq_id, args.sq_pred_data['question_id'].tolist())
        subquestions = clean_lingering_operators_cot(subquestions[0])
        subquestions = [sq for sq in subquestions.split(' ;') if sq.strip() != '']
        sample['subquestions'] = subquestions
    if drop_positions: 
        print(f'HOTPOT dropping {len(drop_positions)} of {len(test_raw_data)}')
        test_raw_data = [sample for i, sample in enumerate(test_raw_data) if i not in drop_positions]
    ### CHANGE END ###
    print('\t📚📚 Starting retrieval and QA predictions')

    if args.use_gold_passages:
        retr_json = give_gold_passages_idxes(args, test_raw_data)
    else: 
        retr_json = get_retr_output(args, test_raw_data)
    get_reader_qa_output(args, retr_json, test_raw_data)


### CHANGE START ###
def get_reader_qa_output(args, retr_pred_dic, test_raw_data):
    is_dev  = args.is_dev 
    qa_ds_type = args.dataset_type

    ### CHANGE START ###
    pred_filename = give_qa_pred_filename(args, qa_ds_type, is_dev)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ### CHANGE END ### 
    qa_max_seq_len = args.qa_max_seq_len
    
    ### CHANGE START ###
    qa_model, qa_tokenizer = load_qa_model(args, qa_max_seq_len, device)
    
    pred_list, idx_list = [], []
    sp_pred, ans_pred = {}, {}
    if is_dev:
        em_tot, f1_tot = [], []
        if qa_ds_type == 'hotpot':
            sp_em_tot, sp_f1_tot = [], []
    
    # get tensors
    ### CHANGE START ###
    from collections import defaultdict
    scores_dict = defaultdict(list)
    ### CHANGE END ###
    for __, sample in enumerate(tqdm(test_raw_data, desc = "QA Predicting:")):
        if args.do_trial and __ >= 5: continue
        ### CHANGE START ###
        id_key = '_id' if qa_ds_type == 'hotpot' else 'id'
        cq_id = sample[id_key]

        pred_sq_as = []
        sp_list_sq = []
        subquestions = sample['subquestions']

        sp_list = retr_pred_dic[cq_id] 

        for sq_id, question in enumerate(subquestions):
            idnum =  f'{cq_id}_sq_{sq_id}'
            idx_list.append(idnum)
            # 1. replace prev answer placeholders
            question = question_replace_var(question, pred_sq_as, sq_id)
            
            # store subquestions with ref var replaced
            subquestions[sq_id] = question
            
            pred_answer, sp_pred, sp = \
                qa_one_step(qa_model, qa_tokenizer, idnum, question, sample, sp_list, sp_pred,
                            qa_max_seq_len, is_dev, qa_ds_type, device, answer_merge = args.answer_merge, 
                            topk = args.topk)
            
            # 2. collect answers and support paras for subquestions
            pred_sq_as.append(pred_answer)
            sp_list_sq.append(sp_list)

        ### CHANGE START ###
        pred_answer = pred_sq_as[-1] if pred_sq_as else ''
        if qa_ds_type == 'hotpot':
            ans_pred[cq_id] = pred_answer
            ### CHANGE END ###

        ### CHANGE START ###
        pred_list.append({'id': cq_id, 'cq': sample['question'],
                            'predicted_answer': pred_answer, 
                            'gold_answer': sample.get('answer', None),
                            'subquestions': subquestions, 
                            'pred_sq_as': pred_sq_as, 
                            'sp_list_sq': sp_list_sq,
                            # 'predicted_support_idxs': sp_list, 
                            'predicted_answerable': True})
        ### CHANGE END ###
        pred_answer = normalize_answer(pred_answer)
        if is_dev:
            ground_truth_answer = sample['answer']
            ground_truth_answer = normalize_answer(ground_truth_answer)

            em = compute_exact(ground_truth_answer, pred_answer)
            f1 = compute_f1(ground_truth_answer, pred_answer)
            em_tot.append(em)
            f1_tot.append(f1)
            ### CHANGE START ###
            scores_dict['sq_idx'].append(idnum)
            scores_dict['qa_em'].append(em)
            scores_dict['qa_f1'].append(f1)
            ### CHANGE END ###

            if qa_ds_type == 'hotpot':
                sp_em, sp_f1 = update_sp(sp, sample['supporting_facts'])
                sp_em_tot.append(sp_em)
                sp_f1_tot.append(sp_f1)
                ### CHANGE START ###
                scores_dict['sp_em'].append(sp_em)
                scores_dict['sp_f1'].append(sp_f1)
                ### CHANGE END ###
    
    if is_dev:
        print(f"em:{sum(em_tot) / len(em_tot)}, f1:{sum(f1_tot) / len(f1_tot)}")
        if qa_ds_type == 'hotpot':
            print(f"sp em:{sum(sp_em_tot) / len(sp_em_tot)}, sp f1:{sum(sp_f1_tot) / len(sp_f1_tot)}")
        
        ### CHANGE START ###
        scores_file_name = pred_filename.replace('sorted_pred', 'sorted_scores').replace('jsonl', 'json')
        with open(scores_file_name, 'w+', encoding='utf-8') as f:
            json.dump(scores_dict, f)
        print(f"📘📘Scores saved to {scores_file_name}")
        ### CHANGE END ###
    
    if qa_ds_type in ['musique', '2wiki', 'hotpot']:
        ### CHANGE START ###
        with open(pred_filename, "w", encoding = 'utf-8') as f:
            ### CHANGE END ###
            for data in pred_list:
                f.write(json.dumps(data)+'\n')

        with open('utils/ircot_musique_validation_idxes.txt', encoding='utf-8') as f:
            ircot_idxes = [l.strip() for l in f.readlines() if l.strip() != '']
        
        assert len(pred_list) == len(scores_dict['qa_em']) == len(scores_dict['qa_f1']), \
            f"\t🚨🚨 Length mismatch: {len(pred_list)} vs {len(scores_dict['qa_em'])} vs {len(scores_dict['qa_f1'])}"
        filt_pred_pos = [i for i, pred in enumerate(pred_list) if pred['id'] in ircot_idxes]
        print(f"\t📘📘Filtered {len(filt_pred_pos)} out of {len(pred_list)}")
        filt_em = [ems for i, ems in enumerate(scores_dict['qa_em']) if i in filt_pred_pos]
        filt_f1 = [f1s for i, f1s in enumerate(scores_dict['qa_f1']) if i in filt_pred_pos]
        if filt_em and filt_f1:
            print(f"\t📘📘 Filtered EM: {sum(filt_em) / len(filt_em)}, F1: {sum(filt_f1) / len(filt_f1)}")
        else: print("\t🚨🚨 No filtered scores")
        
    if qa_ds_type in ['hotpot']:
        with open(pred_filename.replace('.json', '_spfact.json'), "w", encoding="utf-8") as f:
            json.dump({"answer": ans_pred, "sp": sp_pred}, f)
    print(f"evaluation finished!")
    torch.cuda.empty_cache()


### CHANGE START ###
def question_replace_var(question, pred_sq_as, sq_id):
    # check if the pred_sq has a referring variable
    vars = re.findall(r'#[0-9]+', question)
    for v in vars:
        v_num = int(v[1:]) -1  # NOTE: 1-indexed for vars, 0-indexed for pred_sq_as
        try: ref_a = pred_sq_as[v_num]
        except: 
            print('\t🚨 MISSING REFERRING ANS', sq_id, f'VNUM: {v_num} of [{vars}]', question, pred_sq_as)
            ref_a = None
        if ref_a is not None: 
            question = question.replace(v, ref_a)
        else: print('\t🚨 EMPTY REFERRING ANS', sq_id, question, pred_sq_as)
    return question


def load_qa_model(args, qa_max_seq_len, device):
    ### CHANGE START ###
    qa_tokenizer_path = 'microsoft/deberta-v3-large'
    ds_type = args.dataset_type
    ### CHANGE END ###
    
    qa_model_path = qa_tokenizer_path
    qa_checkpoint = CHECKPOINT_DP + CHECKPOINT_HOLDER['reader'][args.model_key]  + '/checkpoint_last.pt'
    
    config = AutoConfig.from_pretrained(qa_model_path)
    config.max_position_embeddings = qa_max_seq_len
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_tokenizer_path)
    ds_type = args.dataset_type
    
    qa_tokenizer.SEP = "</e>"
    qa_tokenizer.DOC = "</d>"
    if args.model_type == 'hotpot': # else mismatch in embeddings
        qa_tokenizer.add_tokens([qa_tokenizer.SEP, qa_tokenizer.DOC])
    if ds_type == 'hotpot':
        qa_tokenizer.SEP_id = qa_tokenizer.convert_tokens_to_ids(qa_tokenizer.SEP)
        qa_tokenizer.DOC_id = qa_tokenizer.convert_tokens_to_ids(qa_tokenizer.DOC)

    qa_model = Reader(config, qa_model_path, task_type = args.dataset_type,
                      len_tokenizer = len(qa_tokenizer) if args.model_type == 'hotpot' else 0, )
    qa_model = load_saved(qa_model, qa_checkpoint, exact = False if args.cross_domain else True)
    qa_model = qa_model.to(device)
    qa_model.eval()

    return qa_model, qa_tokenizer

def qa_one_step(qa_model, qa_tokenizer, idnum, question, sample, sp_list, sp_pred,
                qa_max_seq_len, is_dev, ds_type, device, answer_merge = True, topk = 5):
    ### CHANGE START ###
    while question.endswith("?"): question = question[:-1]
    ### CHANGE END ###

    q_codes = qa_tokenizer.encode(question, add_special_tokens = False, 
                                truncation = True, max_length = qa_max_seq_len)
    
    idx2title = {}
    c_codes = []
    if ds_type == 'hotpot':
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
                l = qa_tokenizer.DOC + " " + title
                for idx2, c in enumerate(sentences):
                    l += (qa_tokenizer.SEP + " " + c)
                encoding = qa_tokenizer.encode(l, add_special_tokens = False, 
                                            truncation = True, max_length = qa_max_seq_len-len(q_codes))
                c_codes.append(encoding)
    elif ds_type in ['musique', '2wiki']:
        # musique
        for i, para in enumerate(sample['paragraphs']):
            if i in sp_list:
                l = para['title'] + '.' + para['paragraph_text']
                encoding = qa_tokenizer.encode(l, add_special_tokens = False, 
                                            truncation = True, max_length = qa_max_seq_len-len(q_codes))
                c_codes.append(encoding)
    total_len = len(q_codes) + sum([len(item) for item in c_codes])
    context_ids = [qa_tokenizer.cls_token_id] + q_codes
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
                if item[i] == qa_tokenizer.SEP_id:
                    sts2title[sts_idx] = idx2title[sp_list[idx]]
                    sts2idx[sts_idx] = sts_idx_local
                    sts_idx += 1
                    sts_idx_local += 1
        context_ids.extend(item)
    context_ids = context_ids[:qa_max_seq_len - 1] + [qa_tokenizer.sep_token_id]
    pred_answer = None
    input_ids = torch.tensor(context_ids, dtype = torch.long, device = device).unsqueeze(0)
    attention_mask = torch.ones([1, len(context_ids)], dtype = torch.long, device=device)
    sp = []
    if ds_type == 'hotpot':
        SEP_index = []
        for i in range(len(context_ids)):
            if context_ids[i] == qa_tokenizer.SEP_id:
                SEP_index.append(i)
        SEP_index = torch.LongTensor([SEP_index]).to(device)
        
        ### MISSING... added from:
        # MHReader.__getitem__
        # reader_mhop_collate in dataset.py
        answer_type = 2
        answer = sample['answer']
        if answer == "no": answer_type = 0
        elif answer == "yes": answer_type = 1
        answer_type = torch.tensor([answer_type], dtype=torch.long)
        ##############################
        
        with torch.no_grad():
            outputs = qa_model(input_ids, attention_mask, sentence_index = SEP_index[0],
                               answer_type = answer_type)
        
        output_answer_type = outputs['output_answer_type']
        ans_type = torch.argmax(output_answer_type).item()
        if ans_type == 0:
            pred_answer = 'no'
        elif ans_type == 1:
            pred_answer = 'yes'
        
        sentence_select = torch.argmax(outputs['sentence_select'], dim=-1)
        try:
            # cross-domain might not have sp task set up
            assert sentence_select.shape[-1] == len(sts2idx)
            sts_idx = 0
            for s in range(len(sentence_select)):
                if sentence_select[s] == 1:
                    sp.append((sts2title[s], sts2idx[s]))
            sp_pred[idnum]  = sp
        except AssertionError: pass
    else:
        with torch.no_grad():
            outputs = qa_model(input_ids, attention_mask)
    start_logits = outputs['start_qa_logits'][0]
    end_logits = outputs['end_qa_logits'][0]
    input_ids = input_ids[0]

    if pred_answer is None:
        if answer_merge:
            punc_token_list = qa_tokenizer.convert_tokens_to_ids(['[CLS]', '?'])
            if ds_type == 'hotpot':
                punc_token_list.extend([qa_tokenizer.SEP_id, qa_tokenizer.DOC_id])
            span_id = merge_find_ans(start_logits, end_logits, input_ids.tolist(), punc_token_list, topk=topk)
            pred_answer = qa_tokenizer.decode(span_id)
        else:
            all_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids.tolist())

            answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
            pred_answer = qa_tokenizer.decode(
                qa_tokenizer.convert_tokens_to_ids(answer_tokens)
            )
    return pred_answer, sp_pred, sp


### CHANGE END ###

if __name__ == '__main__':
    ### CHANGE START ###
    import argparse, os, pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_dev',             type = str,    default = 'True')
    parser.add_argument('--model_type',         type = str,     default = 'musique')
    parser.add_argument('--dataset_type',       type = str,     default = 'musique', 
                        help = 'the dataset to test on (i.e. may not be the one model trained on).')
    parser.add_argument('--output_variants',    type = str, default = [], nargs = '*') # for e.g. rerank_dqg (calcontrol)
    parser.add_argument('--use_gold_passages', type = str,     default = 'False')
    parser.add_argument('--decomp_origin',      type = str,     default = ['supervised', 'musique', 'musique'], 
                        nargs = '*')
    parser.add_argument('--beam_size',          type = int,     default = 2)
    parser.add_argument('--answer_merge',       type = str,    default = 'True')
    parser.add_argument('--topk',               type = int,     default = 10)
    parser.add_argument('--test_ckpt_path',     type = str,     default = '../../checkpoints/{}')
    parser.add_argument('--do_trial',           type = str,    default = 'False')
    parser.add_argument('--filter_hotpot',      type = str,    default = 'True')
    args = parser.parse_args()

    for key in ['is_dev', 'use_gold_passages', 'answer_merge', 'do_trial', 'filter_hotpot', ]:
        setattr(args, key, eval(getattr(args, key)))
    for i, val in enumerate(args.decomp_origin):
        if val == 'None': args.decomp_origin[i] = ''

    args.do_single_hop = True
    args.qa_max_seq_len = 1024
    args.re_max_seq_len = 512
    if args.dataset_type == 'hotpot':
        # NOTE: different from CQ script (allow both val and test of breakhigh to load)
        # assert args.is_dev == True
        args.filter_hotpot = True

    args.cross_domain = False
    if args.dataset_type != args.model_type: args.cross_domain = True

    args.model_key          = DATASET_TO_CODE[args.model_type]
    args.test_ckpt_path     = os.path.join(CHECKPOINT_DP, CHECKPOINT_HOLDER['retriever'][args.model_key])
    cross_str               = '_crossdomain' if args.cross_domain else ''
    args.checkpoint_name    = os.path.basename(args.test_ckpt_path) + cross_str
    args.cqsq_pfx = 'SQ_'

    print('🟥🟥USING DATASET:', args.dataset_type, 'CROSS_DOMAIN:', args.cross_domain, 'CKPT:', args.checkpoint_name)

    main(args)