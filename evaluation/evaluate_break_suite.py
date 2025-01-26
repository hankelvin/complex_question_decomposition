import glob, os, argparse, json, re, tempfile
import pandas as pd

def main(args): 
    fps = {}
    ###### SUPERVISED #####
    coll = []
    sup_dp = 'results/dqg_dqa_outputs'
    for dataset in ['musique', 'breakhigh']:
        directories = sorted(glob.glob(sup_dp+f'/dqg_{dataset}/dqg_{dataset}_flan-t5-large_fp32_*'))
        directories = [d for d in directories if 'phase3_val_as_test' in d]        
        coll.extend(directories)
    fps[''] = coll
    #######################


    ###### SFT #####
    sup_dp = 'results/dqg_dqa_outputs'
    mapping = {
                'llm_top1_sft': 
                'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
                
                'llm_top1_sft_rtfilt-1x': 
                'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-1x-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
                
                'llm_stv_sft_rtfilt-2x': 
                'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-2x-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
                
                'llm_stv_sft_rtfilt-2x-upsamp': 
                'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-2x-upsamp-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test'
                }
    
    for variant, dp_template in mapping.items():
        coll = []
        for dataset in ['breakhigh', 'musique']:
            for model in ['llama', 'qwen']: 
                line = f'{sup_dp}/{dp_template.format(dataset, model)}'
                coll.append(line)
                lline = line.replace('picked-ge-ll-ph-qw_', 'picked-ge-ll-ph-qw_CoTnshot0_')
                coll.append(lline)
        fps[variant] = coll
    #######################


    ###### VARIOUS LLMs and STV #####
    fps['results/decomp_qg'] = \
            [
            'gpt4o_decomp_qg_breakhigh_validation', 
            'gpt4o_decomp_qg_breakhigh_validation_CoTnshot0', 
            'gpt4o_decomp_qg_musique_validation', 
            'gpt4o_decomp_qg_musique_validation_CoTnshot0', 

            'decomp_qg_n_shot-5_CoT_breakhigh_validation_llama_modelline-2_prmptv2_large',
            'decomp_qg_n_shot-5_CoT_musique_validation_llama_modelline-2_prmptv2_large',

            'decomp_qg_breakhigh_validation_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ge-ll-ph-qw_modelline-2_prmptv2/original_run0',
            'decomp_qg_breakhigh_validation_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ay-mi-nv-ol_modelline-2_prmptv2/original_run0',
            'decomp_qg_breakhigh_validation_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ay-ge-ll-mi-nv-ol-ph-qw_modelline-2_prmptv2/original_run0',

            'decomp_qg_musique_validation_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ge-ll-ph-qw_modelline-2_prmptv2/original_run0',
            'decomp_qg_musique_validation_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ay-mi-nv-ol_modelline-2_prmptv2/original_run0',
            'decomp_qg_musique_validation_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ay-ge-ll-mi-nv-ol-ph-qw_modelline-2_prmptv2/original_run0',
            ]
    if 'results/decomp_qg' not in fps: fps['results/decomp_qg'] = []
    for n_shot in [0, 5]:
        for dataset in ['breakhigh','musique']:
            for model in ['gemma', 'llama', 'phi3' ,'qwen']:
                fps['results/decomp_qg'].append(f'decomp_qg_n_shot-{n_shot}_CoT_{dataset}_validation_{model}_modelline-2_prmptv2')
    #######################

    results = {}
    results_fp = 'MASTER_breakeval_results.csv'
    results_df = pd.DataFrame()
    key_order = []
    for dp, directories in fps.items():
        for directory in directories:
            print('Working on this DP:', dp, directory)

            if "_run" not in directory: key = os.path.basename(directory) 
            else:                   key = '/'.join(directory.split('/')[-2:])

            if dp in ['llm_top1_sft', 'llm_top1_sft_rtfilt-1x', 'llm_stv_sft_rtfilt-2x',
                      'llm_top1_sft_rtfilt-1x-upsamp', 'llm_stv_sft_rtfilt-2x-upsamp',]:
                model       = re.search('llama|qwen', directory).group()
                dataset     = re.search('breakhigh|musique', directory).group(0)
                cotnshot0_str   = re.search('_CoTnshot0', directory)
                if cotnshot0_str: cotnshot0_str = f'-{cotnshot0_str.group()}'
                else: cotnshot0_str = ''
                assert model and dataset
                key = f'{dp}-{model}-{dataset}{cotnshot0_str}'
                dp_to_use = ''
            else: dp_to_use = dp

            hotpot_filters = [False, True] if 'breakhigh' in directory else [False]
            for hp_filter in hotpot_filters:
                if hp_filter:   key_to_use = key + '_hponly'
                else:           key_to_use = key
                print('KEY!', key_to_use)
                
                if os.path.exists(results_fp):
                    results_df = pd.read_csv(results_fp, index_col = 0)
                    results = results_df.T.to_dict()
                
                if key_to_use in results_df.index:
                    print('Already processed this DP:', dp_to_use, directory)
                    key_order.extend(set(results.keys()).difference(key_order))
                    continue
                else: 
                    try:
                        df, res = do_one_file(os.path.join(dp_to_use, directory), hp_filter)
                        results[key_to_use] = {k: round(float(v), 4) if v is not None else v for k, v in res.items()}
                        key_order.append(key_to_use)
                    except Exception as e: 
                        print(f'🟥🟥Error processing {directory}', e)
                        continue
                    
                    # save along the way
                    results_df = pd.DataFrame(results).T[['EM',	'SARI', 'GED',	'GLEU',	'CHRF',	]]
                    results_df.loc[key_order].to_csv(results_fp, index = True)

    return results

########## CLEAN UP: pad tokens and lingering operators (break) ##########
# same at test_model_SQ.py
def remove_special_tokens_lingering_operators(csv_filepath, pad_char = '<|pad|>', do_tmp = True,
                                              cotnshot0 = False):
    
    df = pd.read_csv(csv_filepath)
    
    if 'question_text' in df.columns:
        df['question_text'] = df['question_text'].apply(lambda x: x.replace(pad_char, ''))
    df['decomposition'] = df['decomposition'].apply(lambda x: x.replace(pad_char, ''))
    # ensure no remaining operators in sub-qs. issues encountered by LLMs (from GPT4o to cllms)
    df['decomposition'] = df['decomposition'].apply(lambda x: clean_lingering_operators_cot(x, cotnshot0))

    tmp_csv_filepath = None
    if do_tmp:
        tmp_csv_filepath = tempfile.NamedTemporaryFile(delete=False).name
        df.to_csv(tmp_csv_filepath, index = False)

    return df, tmp_csv_filepath
    
def clean_lingering_operators_cot(decompositions_str, cotnshot0 = False):

    cot_patterns = [r'.+(the sequence of sub-questions should be: )(.+)']
    if cotnshot0: 
        cot_patterns.append(r'.+(here are the sub-questions:)(.+)')

    for cot_pattern in cot_patterns:  # prioritize the first pattern (sequence of sub-questions)
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

def hotpot_filter(dataset_file, preds_file, do_tmp = True):

    df_data = pd.read_csv(dataset_file)
    df_pred = pd.read_csv(preds_file)

    keep_indices = df_data[df_data.question_id.str.startswith('HOTPOT_')].index
    if len(keep_indices) == 0: 
        with open('evaluation/hotpot_idxes.json', encoding = 'utf-8') as f: 
            __ = json.load(f)
        # split = re.search('train|validation|test', dataset_file).group(0)
        # NOTE: not possible to extract split (some are phase3_val_as_test)
        hotpot_idxes = []
        for split in ['train', 'validation', 'test']: hotpot_idxes.extend(__[split])
        keep_indices = set(df_data.question_id.tolist()).intersection(set(hotpot_idxes))
        print('\t\tHERE 1000', keep_indices)
        
    print('BEFORE', df_data.shape, df_pred.shape)
    df_data = df_data.loc[keep_indices]
    df_pred = df_pred.loc[keep_indices]
    print('AFTER', df_data.shape, df_pred.shape)
    empty = df_pred.shape[0] == 0
    

    tmp_csv_filepath_data = None
    tmp_csv_filepath_pred = None
    if do_tmp:
        tmp_csv_filepath_data = tempfile.NamedTemporaryFile(delete=False).name
        tmp_csv_filepath_pred = tempfile.NamedTemporaryFile(delete=False).name
        df_data.to_csv(tmp_csv_filepath_data, index = False)
        df_pred.to_csv(tmp_csv_filepath_pred, index = False)

    return df_data, df_pred, tmp_csv_filepath_data, tmp_csv_filepath_pred, empty

###########################################################################

def do_one_file(dp, hp_filter = False):
    import sys
    sys.path.append('tools/break_evaluator/scripts')
    from evaluate_predictions import validate_args
    from evaluate_predictions import main as evaluate_break_main   
    class Args: pass 
    eval_args                   = Args()
    eval_args.dataset_file_orig = os.path.join(dp, 'text_labels.csv')
    c_CoTnshot0 = '_CoTnshot0' in dp
    __, eval_args.dataset_file  = remove_special_tokens_lingering_operators(eval_args.dataset_file_orig, 
                                                                            cotnshot0 = c_CoTnshot0)
    eval_args.preds_file_orig   = os.path.join(dp, 'text_predictions.csv')
    __, eval_args.preds_file    = remove_special_tokens_lingering_operators(eval_args.preds_file_orig, 
                                                                            cotnshot0 = c_CoTnshot0)
    print('TMP FILES HERE:', eval_args.dataset_file, eval_args.preds_file)
    if hp_filter:
        __, __, eval_args.dataset_file, eval_args.preds_file, empty = \
            hotpot_filter(eval_args.dataset_file, eval_args.preds_file)
        print('TMP FILES HERE (FILTERED HOTPOT):', eval_args.dataset_file, eval_args.preds_file)
        if empty: return None, {'EM': None, 'SARI': None,	'GED': None, 'GLEU': None, 'CHRF': None}

    eval_args.random_n          = False
    eval_args.no_cache          = True
    eval_args.output_file_base  = f'{dp}/text_results'
    eval_args.metrics           = ['exact_match', 'sari', 'ged', 'gleu', 'chrf']

    validate_args(eval_args)
    for strip_qmark_whitespace in [True]:
        eval_args.strip_qmark_whitespace = strip_qmark_whitespace
        res, fine_grained_scores = evaluate_break_main(eval_args)

        map = {'bleu': 'BLEU', 'normalized_bleu': 'norm_BLEU',
                'gleu': 'GLEU', 'normalized_gleu': 'norm_GLEU',
                'bleurt': 'BLEURT', 'normalized_bleurt': 'norm_BLEURT',
                'chrf': 'CHRF', 'normalized_chrf': 'norm_CHRF',
                'exact_match': 'EM', 'normalized_exact_match': 'norm_EM', 
                'sari': 'SARI', 'ged': 'GED', 'ged_plus': 'GED+'}
        res = {map.get(k, k): v for k,v in res.items()}
        print(res)
        savepath = os.path.join(dp, 'breakeval_scores')
        if not os.path.exists(savepath): os.makedirs(savepath)
        str_strip_qmark_whitespace = '_strip_qmark_whitespace' if eval_args.strip_qmark_whitespace else ''
        with open(f'{savepath}/text_scores{str_strip_qmark_whitespace}.json', 'w+') as json_file:
            json.dump(res, json_file)

        for agg_field, df in fine_grained_scores.items():
            if df is not None:
                df.to_csv(f'{savepath}/text_score_{agg_field}{str_strip_qmark_whitespace}.csv', index = False)

    return df, res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='results/decomp_qg')
    parser.add_argument('--datasets', type=str, nargs='+', default=['musique', 'breakhigh'])
    main_args = parser.parse_args()

    results = main(main_args)
    





