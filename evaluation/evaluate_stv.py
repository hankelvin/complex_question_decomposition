import random, json, os
from collections import defaultdict
import numpy as np
from evaluate_stv_utils import give_original_data, Dataset



TASK_HOLDER  = {2: 'rerank_dqg',
                3: 'rerank_synqdggpt4o'}
DATA_MAP     = {1: 'musique',
                2: 'breakhigh',}
# NOTE: keys here are from TASK_HOLDER
DATA_DIR_MAP = {'rerank_outputs':
                {2: 'results/rerank_dqg_{0}_listwise/rerank_dqg_{0}_listwise_{0}_{1}_{2}_modelline-2_prmptv2_rankllm-nshot-2-GE-LL-PH-QW_CBS-nb1_sc-sch', 
                 3: 'results/calib/rerank_syndqggpt4o-{0}_listwise_full-unprc-scs_nrk-4',},
                'original_data':
                {2: 'data/01_unified_format',
                 3: 'data/01_unified_format',},
                'model_outputs':
                {2: 'results/decomp_qg/decomp_qg_n_shot-5_CoT_{0}_{1}_{2}', # NOTE: model_lineup, prompt_version added in evaluate_stv_utils
                 3: None,},
                 }
DSNAME_TO_CODE  = {'musique': 490, 'breakhigh': 470}
SYSNAME_TO_CODE = {'llama': 1, 'gemma': 2, 'phi3': 3, 'gritlm_gen': 4,}

def main(args):
    c1 = args.task == 1 and args.dataset in [5]
    c2 = args.task == 3 and args.dataset in [8,9,10]
    c3 = args.task == 4 and args.dataset in [6,7]
    c4 = args.task == 5 and args.dataset in [1,2,3]
    args.has_ground_truth = c1 or c2 or c3 or c4
    
    random.seed(54506)
    args.task_name  = TASK_HOLDER[args.task]
    args.ds_name    = DATA_MAP[args.dataset]
    args.ds_code    = DSNAME_TO_CODE.get(args.ds_name, None)
    if args.ds_name == 'hotpotqa_distractor': assert args.split == 'validation'
    if args.split == 'test': args.ds_code -= 10
    args.c_dqg = args.task_name in ['rerank_dqg', 'rerank_dqg_calib'] and \
                 args.ds_name   in ['musique', 'breakhigh']

    ##### 1. load original data #################################################
    dataset_obj = Dataset(args              = args,
                          instances         = give_original_data(args, DATA_DIR_MAP),
                          data_dir_map      = DATA_DIR_MAP,
                          sysname_to_code   = SYSNAME_TO_CODE)
    #############################################################################
    
    
    ##### 2. (OPTIONAL) load model DQG predictions ##############################
    if args.c_dqg:
        dataset_obj.load_model_dqg_predictions()
    #############################################################################
    
    ##### 3. load LLM ranking outputs ###########################################
    if args.task_name in ['rerank_syndqggpt4o']:
        dp_line = 'insts-5000_smps-100000_sc-sch/model_outputs.jsonl'
    elif args.c_dqg:
        if args.task_name in ['rerank_dqg']:
            if args.model_lineup > 1 or args.prompt_version > 1:
                dp_line = '/model_outputs.jsonl'
            else: 
                assert args.split != 'test'
                assert args.output_variants
                calcontrol = args.output_variants[0]
                dp_line = f'insts-999_smps-999_sc-sch_calcontrol-{calcontrol}/model_outputs.jsonl'
        else: 
            assert args.split != 'test'
            dp_line = 'insts-50_smps-10000_sc-sch/model_outputs.jsonl'
    elif args.task_name in ['rerank']:
        dp_line = 'model_outputs.jsonl'

    # TODO: set up dp_line for joint llm pred and calibration directly from llm_inference
    
    c1 = args.ds_name.startswith('syndqg')
    if args.split == 'test' and not c1:
        assert args.ds_name in ['musique', 'breakhigh']
        # TODO: 
        raise NotImplementedError
    else: 
        dataset_obj.load_rankllm_predictions(dp_line)
    #############################################################################


    ##### 4. STV voting ################################################################
    # NOTE: use same ref_perm_order_to_use across original and calibrated ranks (for comparability)
    ref_perm_order_holder = dataset_obj.create_ref_perm_order_to_use(num_runs = args.num_runs)
    run_results     = {'all_results_topk': defaultdict(dict), 
                       'all_results_ndcg': defaultdict(dict)}
    c_rerank_dqg    = args.task_name in ['rerank_dqg']      and args.ds_name in ['musique', 'breakhigh']
    save_run_results= False
    ranks_to_use = ['original']
    for run_num in range(args.num_runs):
        for rank_to_use in ranks_to_use: 
            if c_rerank_dqg and rank_to_use == 'optimal': continue
            if rank_to_use == 'optimal' and run_num > 0: continue  # only need to run optimal once
            print(f'\t{"🟥"*(run_num+1)} working on run number: ', run_num, '\trank_to_use: ', rank_to_use)
            ref_perm_order_to_use = {k: v[run_num] for k,v in ref_perm_order_holder.items()}
            dataset_obj.reset_for_new_run()
            
            # a. collect the votes
            dataset_obj.collate_votes(rank_to_use   = rank_to_use, 
                                      ref_perm_order_to_use = ref_perm_order_to_use)
            print(f'\t{"🟥"*(run_num+1)} votes collated.')
            # b. compute the STV voting
            dataset_obj.compute_stv()
            print(f'\t{"🟥"*(run_num+1)} STV computed.')
            # (OPTIONAL): compute for rerank_dqg task
            if args.task_name == 'rerank_dqg':
                dataset_obj.compile_stv_dataset_rerank_dqg(rank_to_use = rank_to_use, 
                                                           run_num = run_num)
                print(f'\t{"🟥"*(run_num+1)} rerank_dqg STV dataset compiled.')
            #############################################################################

            ##### 6. identifying the best and worst predictions #########################
            if args.has_ground_truth:
                save_run_results = True
                all_results_topk = dataset_obj.pick_top_ranked()
                run_results['all_results_topk'][rank_to_use][run_num] = all_results_topk
            #############################################################################

        ##### 7. give average over runs ###############################
        if args.has_ground_truth:
            print(f'{rank_to_use.upper()}')
            for hname, holder in run_results.items():
                sysnames = list(holder[0].keys())
                metrics  = list(holder[0][sysnames[0]].keys())
                print(f'{hname.upper()} mean over runs:')
                for mn in metrics:
                    if not (mn in ['em_mean', 'f1_mean'] or mn.startswith('ndcg')): continue
                    for sn in sysnames: 
                        vals = [h2[sn][mn] for rn, h2 in holder.items()]
                        val  = round(np.mean(vals), 4) 
                        print(f"{mn} for {sn}: {' '*(15-len(sn))} \t {val}")

    if save_run_results: 
        # save run results to a file
        savepath = 'results/stv_run_results/'
        if not os.path.exists(savepath): os.makedirs(savepath)  # create directory if not exists
        savepath += f'{args.task_name}_{args.ds_name}_{args.split}_{args.nranks}_{args.single_perm}'
        if args.output_variants: savepath += f'_variants-' + '-'.join(args.output_variants)
        savepath    += '_dqg-'     + '-'.join([sys[:2] for sys in args.dqg_systems])
        if args.dqg_roundtrip: savepath += f'_roundtrip-top{args.dqg_roundtrip}'
        savepath    += '_rankllm-' + '-'.join([sys[:2] for sys in args.rankllm_systems])
        if args.model_lineup > 1: savepath += f'_modelline-{args.model_lineup}'
        if args.prompt_version > 1: savepath += f'_prmptv{args.prompt_version}'
        with open(f'{savepath}.json', encoding = 'utf-8', mode = 'w+') as f:
            json.dump(run_results, f)
        print(f'\t🟩 run results saved to:', savepath)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',               type = int, default = 2)
    parser.add_argument('--dataset',            type = int, default = 2)
    parser.add_argument('--split',              type = str, default = 'validation')
    parser.add_argument('--nranks',             type = int, default = 4)
    parser.add_argument('--dqg_roundtrip',      type = int, default = 2) # if doing roundtrip, return top-n
    parser.add_argument('--num_runs',           type = int, default = 20)
    parser.add_argument('--single_perm',        type = bool, default = True) # i.e. no perm order sampling
                                                # e.g. rerank_dqg (calcontrol) , or _rankllm-nshot- 
    parser.add_argument('--output_variants',    type = str, default = [], nargs = '*') 
    parser.add_argument('--model_lineup',       type = int, default = 2)
    parser.add_argument('--prompt_version',     type = int, default = 2)
    parser.add_argument('--rankllm_systems',    type = str, nargs = '*', 
                        default = ['gemma','llama', 'phi3', 'qwen'])
    # gemma gritlm_gen phi3 llama 
    parser.add_argument('--dqg_systems',        type = str, nargs = '*', 
                        default = ['gemma', 'llama', 'phi3', 'qwen'])
    args = parser.parse_args()

    if args.task not in [2]: args.dqg_roundtrip = False
    args.rankllm_systems = sorted(args.rankllm_systems)
    args.dqg_systems     = sorted(args.dqg_systems)
    if args.single_perm: args.num_runs = 1

    main(args)



