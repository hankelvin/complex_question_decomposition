import torch, os, json, shutil, glob, re, sys
torch.manual_seed(54506)

'''
NOTE:
padding_side for llama: 
right side at train, left side at generate
sources: 
- https://github.com/meta-llama/llama3/issues/42
- https://github.com/huggingface/transformers/issues/31672
adding pad_token if None 
- https://github.com/turboderp/exllamav2/issues/415
'''

def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    from utils import load_base_model, make_DQG_model, Logger
    from lightning_accelerate_utils import accelerate_forward, lightning_forward
    from dataloaders import prep_data
    from huggingface_hub import login
    login(args.hf_token)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    args = create_savepath(args)
    logger = args.logger = Logger(args.savepath, main_process = True) 
    
    tokenizer, base_model = load_base_model(args)
    train_dataloader, val_dataloader, test_dataloader, \
        collate_obj, collate_obj_eval = prep_data(args, tokenizer)
    DQG_model, trainer = make_DQG_model(args, tokenizer, base_model,
                                    train_dataloader, val_dataloader, test_dataloader,
                                    bypass_trainer = True if args.use_accelerate else False)
    DQG_model.collate_obj = collate_obj
    DQG_model.collate_obj_eval = collate_obj_eval

    ##### save config ###################
    holder = {}
    for key in ['logger', 'device', 'qa_args', 'model_args']:
        holder[key] = getattr(args, key)
        delattr(args, key)
    with open(os.path.join(args.savepath, 'exp_configs.json'), 'w+') as f: 
        json.dump(vars(args), f)
    logger.print('Exp config saved!!')
    for key, val in holder.items(): setattr(args, key, val)
    #####################################

    ###### training and validation ######
    if args.use_accelerate:    DQG_model = accelerate_forward(args, DQG_model, 
                                                  monitor = 'eval_em' if DQG_model.eval_em else 'eval_bleu',
                                                  save_model = False if args.do_lora else True)
    else:                      lightning_forward(args, DQG_model, trainer = trainer)
    #####################################

    ###### model saving #################
    if args.test_only == False:
        if args.use_accelerate: DQG_model = DQG_model.accelerator.unwrap_model(DQG_model)
        else:                   DQG_model = DQG_model
        
        for fp in glob.glob(f'{args.savepath}/*.ckpt'): 
            shutil.rmtree(fp)
        for fp in glob.glob(f'{args.savepath}/*.bin') + glob.glob(f'{args.savepath}/*.pkl'):  
            os.remove(fp)
        
        if args.do_lora:    ckpt_savepath = f'{args.savepath}/test_model_lora.ckpt'
        else:               ckpt_savepath = f'{args.savepath}/test_model.ckpt'
        DQG_model.model.save_pretrained(ckpt_savepath)
    #####################################

def create_savepath(args):
    save_dir = f'results/dqg_dqa/{args.task}'
    args.savepath = f'{save_dir}/{args.task}_{args.model_name}'
    if args.prompt_version > 1: args.savepath += f'_prmptv{args.prompt_version}'
    if args.use_dqg_llmgen:     args.savepath += f'_dqg_llmgen{args.use_dqg_llmgen}'
    if args.use_dqg_rtfilt:     
                                args.savepath += f'_{args.use_dqg_rtfilt}'
                                args.savepath += f'-cut{args.rtfilt_cutoff}'
    if args.exp_str:            args.savepath += args.exp_str
    if args.fp32:               args.savepath += '_fp32'
    if args.load_in_nbits:      args.savepath += f'_nbits-{args.load_in_nbits}'
    if args.do_dqg_llmgen_qa : 
         args.savepath += f'_dqg_llmgen-QA{args.do_dqg_llmgen_qa}'
         if args.do_dqg_llmgen_qa_usecontext:  args.savepath += '_usecontext'
    if not args.qa_task:
        print("HERE 10", args.decomp_qg_args)
        if args.decomp_qg_args:
            if args.decomp_qg_args['cot'] == True: 
                if args.decomp_qg_args['n_shot'] != 3: 
                     args.savepath += f'_CoTnshot{args.decomp_qg_args["n_shot"]}'
            else: args.savepath += f'_noCoT'
        if args.do_lora:            
                                    args.savepath += '_LORA'
                                    args.savepath += f'_peft'
                                    for k,v in args.lora_settings.items():
                                        if k == 'peft_lora_keys':
                                            plk = "-".join([''.join([vvv[0] for vvv in vv.split('_')]) for vv in v])
                                            args.savepath += f'_{"".join([kk[0] for kk in k.split("_")])}-{plk}'
                                        else:
                                            args.savepath += f'-{"".join([kk[0] for kk in k.split("_")])}-{v}'
        if args.use_accelerate:     args.savepath += '_accelerate'
    if args.test_only != False: args.savepath += f'_test_only-{args.test_only}'
    if args.cross_domain:       args.savepath += f'_cross_domain-{args.cross_domain}'
    if args.do_trial:           args.savepath += '_TRIAL'

    if not os.path.exists(args.savepath): 
        os.makedirs(args.savepath)
        print('SAVEPATH CREATED AT:', args.savepath)
    else: print('SAVEPATH EXISTS AT:', args.savepath)
    args.save_dir = args.savepath
    return args

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arugments for running the script to train DQE-like QA and QG models.')
    parser.add_argument('--task', help = 'Which (MAIN) task to run  (i)     dqg_breakhigh; \
                                                                    (ii)    dqg_musique; \
                                                                    (iii)   dqg_2wikimultihop; \
                                                                    (iv)    dqg_hotpotqa',
                        default = 'dqg_breakhigh', type = str)
    parser.add_argument('--model_name', help = 'name of the model to load for finetuning.',
                        default = 'llama', type = str)
    parser.add_argument('--prompt_version',         type = int, default = 2)
    parser.add_argument('--max_epochs', help = '# of epochs to run training for', 
                        default = 20, type=int)
    parser.add_argument('--lr', help = 'learning rate', 
                        default = 3e-5, type=float)
    parser.add_argument('--bsz_train', help = 'training batch size', 
                        default = 8, type=int)
    parser.add_argument('--bsz_test', help = 'validation and testing batch size', 
                        default = 32, type=int)
    parser.add_argument('--num_accumulation_steps', help = 'number of steps to take before optim.step()', 
                        default = 1, type=int)
    parser.add_argument('--save_top_k_models', help = 'number of model checkpoints to save.', 
                        default = 1, type=int)
    parser.add_argument('--gen_args', help = 'arguments for .generate()', 
                        default = {'max_length': 200, 'num_beams': 5, 'use_cache': True}, type = dict)
    parser.add_argument('--label_smoothing', help = 'label smoothing rate', 
                        default = 0.0, type = float)
    parser.add_argument('--test_only', help = 'Whether to run test split on model only (has to load ckpt)', 
                        default = 'False', type = str) # phase3_val_as_test, roundtrip_filtering
    parser.add_argument('--load_ckpt_path', help = 'The path of the ckpt to load the model weights from.', 
                        default = None, type = str)
    parser.add_argument('--resume_ckpt_path', help = 'The path of the ckpt to resume training from.', 
                        default = 'None', type = str)
    parser.add_argument('--hf_token',               type = str, default = '')
    parser.add_argument('--num_gpu', help = '# number of GPUs.',
                        default = 1, type = int)
    parser.add_argument('--num_node', help = '# of nodes.',
                        default = 1, type = int)
    parser.add_argument('--exp_str', help = 'identifier string to add to savepath.',
                        default = '', type = str)
    parser.add_argument('--no_weight_decay', help = 'Whether ensure no weight decay done.',
                        default = False, type = bool)
    parser.add_argument('--fp32', help = 'Whether to load and train/tune the model in fp32 (else bf 16 for t5-class).',
                        default = True, type = bool)
    parser.add_argument('--cross_domain', help = 'Whether to test on a different dataset. If so give applies \
                        to e.g. (i) dqg_musique-dqg_2wikimultihop; (ii) dqg_2wikimultihop-dqg_musique',
                        default = 'False', type = str)
    parser.add_argument('--load_in_nbits',  type = str, default = 'False', 
                        help = 'the number of bits to load the model in (False, 4, 8)')
    parser.add_argument('--do_lora',        type = str, default = 'True',
                        help = 'whether to use LoRA adapters or not')
    parser.add_argument('--lora_settings',  type = str, default = '''['128', 'none', '0.5', '0.05']''', nargs = '*', 
                        help = 'LoRA settings: ["lora_r", "lora_bias", "lora_alpha_ratio", "lora_dropout"]')
    parser.add_argument("--use_accelerate", type = str, default = 'True',
                        help = "whether to use the accelerate package for training")
    parser.add_argument('--decomp_qg_args',         type = str, default = ['3', 'True'], nargs = '*')
    parser.add_argument('--do_trial',       type = str, default = 'False')
    parser.add_argument('--use_dqg_llmgen', type = str, default = 'None')
    parser.add_argument('--use_dqg_rtfilt', type = str, default = 'None')   # 'rtfilt-1x'/'rtfilt-2x' where to use roundtrip filtered instances
    parser.add_argument('--do_dqg_llmgen_qa', type= str, default = 'None')  # see args.dqg_llmgen_qa_validation
    parser.add_argument('--do_dqg_llmgen_qa_usecontext', type= str, default = 'False')  # whether to add context prior to the CQ
    args = parser.parse_args()

    args.save_last      = True
    args.trial_num      = 10
    args.rtfilt_cutoff  = 0.7
    args.mask_value     = 0
    args.patience       = 5
    args.do_validation  = False
    args.do_test        = False
    
    for key in ['test_only', 'use_dqg_llmgen', 'do_dqg_llmgen_qa', 'do_dqg_llmgen_qa_usecontext', 'use_dqg_rtfilt']:
        val = getattr(args, key)
        if val in ['True', 'False', 'None']: setattr(args, key, eval(val))
        val = getattr(args, key)
    for key in ['load_in_nbits', 'do_lora', 'lora_settings', 'use_accelerate', 'do_trial']:
        setattr(args, key, eval(getattr(args, key)))

    args.qa_task = args.task.startswith('cqa_') or args.task.startswith('dqa_') or args.task.startswith('cqadqa_')
    if not args.qa_task: args.do_dqg_llmgen_qa = None

    if args.do_dqg_llmgen_qa is not None: 
         assert args.use_dqg_rtfilt is None, 'Cannot use both dqg_llmgen_qa and dqg_rtfilt (roundtrip has to stay zeroshot)'
         args.use_dqg_llmgen = None
         args.do_dqg_llmgen_qa_usecontext = True
         args.do_lora = False

    if 'zeroshotCoT' in args.task: 
         args.do_lora = False

    args.model_mapping = {
        'gemma':            'google/gemma-2-9b-it',
        'llama':            'meta-llama/Llama-3.1-8B-Instruct',
        'llama-large':      'meta-llama/Llama-3.1-70B-Instruct',
        'phi3':             'microsoft/Phi-3.5-mini-instruct',
        'qwen':             'Qwen/Qwen2.5-7B-Instruct',
        'qwen-large':       'Qwen/Qwen2.5-72B-Instruct',
        'flan-t5-large':    'google/flan-t5-large',
    }
    CHECKPOINT_MAPPING = {
        'flan-t5-large': {  'dqg_breakhigh':     'dqg_breakhigh_flan-t5-large_fp32/epoch=10-step=24068.ckpt',
                            'dqg_musique':       'dqg_musique_flan-t5-large_fp32/epoch=4-step=12465.ckpt',
                            'dqg_2wikimultihop': 'dqg_2wikimultihop_flan-t5-large_fp32/epoch=17-step=376776.ckpt',},
                                                  
        'llama':         {  'dqg_breakhigh':     'dqg_breakhigh_llama_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw{}_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate/test_model_lora.ckpt',
                            'dqg_zeroshotCoT_breakhigh': None,
                            'dqg_musique':       'dqg_musique_llama_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw{}_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate/test_model_lora.ckpt',
                            'dqg_zeroshotCoT_musique': None,},
        'qwen':          {  'dqg_breakhigh':     'dqg_breakhigh_qwen_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw{}_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate/test_model_lora.ckpt',
                            'dqg_zeroshotCoT_breakhigh': None,
                            'dqg_musique':       'dqg_musique_qwen_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw{}_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate/test_model_lora.ckpt',
                            'dqg_zeroshotCoT_musique': None,},
        'gemma':         {  'dqg_zeroshotCoT_breakhigh': None,
                            'dqg_zeroshotCoT_musique': None,},
        'phi3':         {  'dqg_zeroshotCoT_breakhigh': None,
                            'dqg_zeroshotCoT_musique': None,},                                                        
                           }
    args.eot_map    = {'llama': '<|eot_id|>', 'llama-large': '<|eot_id|>', 'qwen': '<|im_end|>', 
                        'phi3': '<|end|>\n<|endoftext|>', 'gemma': '<end_of_turn>'}
    args.chat_model = True if re.search(rf"{'|'.join(['gemma', 'llama', 'phi3', 'qwen'])}", args.model_name) else False

    args.dqg_llmgen_mapping = {
        # no predictions on train split for llama-large and gpt4o
        #   'llama-large':            'decomp_qg_n_shot-5_CoT_{0}_{1}_llama_modelline-2_prmptv2_large',
        #   'gpt4o':                  'gpt4o_decomp_qg_{0}_{1}',
        'stv-picked-ge-ll-ph-qw':               'decomp_qg_{0}_{1}_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ge-ll-ph-qw_modelline-2_prmptv2/original_run0{2}',    
        'stv-picked-ay-mi-nv-ol':               'decomp_qg_{0}_{1}_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ay-mi-nv-ol_modelline-2_prmptv2/original_run0{2}',   
        'stv-picked-ay-ge-ll-mi-nv-ol-ph-qw':   'decomp_qg_{0}_{1}_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ay-ge-ll-mi-nv-ol-ph-qw_modelline-2_prmptv2/original_run0{2}',
        'llama-large':                          'decomp_qg_n_shot-5_CoT_{0}_{1}_llama_modelline-2_prmptv2_large',
        'gpt4o':                                'gpt4o_decomp_qg_{0}_{1}',
    }
    ##### FOR DOWNSTREAM QA #####
    args.dqg_llmgen_qa_validation_dp = {
        'llama-large':              f'results/decomp_qg',
        'gpt4o':                    f'results/decomp_qg',
        'gpt4o_CoTnshot0':          f'results/decomp_qg',
        'sft_ft5':                  f'results/dqg_dqa', 
        'llm_single':               f'results/decomp_qg',
        'llm_single_CoTnshot0':     f'results/dqg_dqa',
        'llm_top1':                 f'results/decomp_qg',
        'llm_top1_sft':             f'results/dqg_dqa',
        'llm_top1_sft_CoTnshot0':   f'results/dqg_dqa',
        'llm_top1_sft_roundtrip':   f'results/dqg_dqa',
        'llm_stv_sft_roundtrip':    f'results/dqg_dqa'
    }
    args.dqg_llmgen_qa_validation = {
        'llama-large':              'decomp_qg_n_shot-5_CoT_{0}_{1}_llama_modelline-2_prmptv2_large',  
        'gpt4o':                    'gpt4o_decomp_qg_{0}_{1}', 
        'gpt4o_CoTnshot0':          'gpt4o_decomp_qg_{0}_{1}_CoTnshot0', 
        'sft_ft5':                  'dqg_{0}/dqg_{0}_flan-t5-large_fp32_test_only-phase3_val_as_test',
        # 0 is dqg_llmgen_qa_validation_dp, 1 is dataset, 2 is split
        'llm_single':               'decomp_qg_n_shot-5_CoT_{0}_{1}_{3}_modelline-2_prmptv2',
        'llm_single_CoTnshot0':     'dqg_zeroshotCoT_{0}/dqg_zeroshotCoT_{0}_{1}_prmptv2_CoTnshot0_accelerate_test_only-phase3_val_as_test',
        'llm_top1':                 'decomp_qg_{0}_{1}_STV_listwise_dqg-ge-ll-ph-qw_rankllm-ge-ll-ph-qw_modelline-2_prmptv2/original_run0',
        'llm_top1_sft':             'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
        'llm_top1_sft_CoTnshot0':   'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_CoTnshot0_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
        'llm_top1_sft_roundtrip':   'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-1x-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
        'llm_stv_sft_roundtrip':    'dqg_{0}/dqg_{0}_{1}_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_rtfilt-2x-cut0.7_LORA_peft-lr-64-lb-none-lar-0.5-ld-0.05-la-16.0_plk-qp-kp-vp-op-dp-gp-up_accelerate_test_only-phase3_val_as_test',
    }
    for m in ['gemma', 'llama', 'phi3', 'qwen', 'aya', 'mistral', 'nvidia_llama3', 'olmo']:   
        for key in ['llm_single', 'llm_top1_sft', 'llm_top1_sft_roundtrip', 'llm_stv_sft_roundtrip']:
            for zs_str in ['', '_CoTnshot0']:
                args.dqg_llmgen_qa_validation_dp[f'{key}_{m}{zs_str}'] = args.dqg_llmgen_qa_validation_dp[key]
                if key == 'llm_single': 
                    args.dqg_llmgen_qa_validation[f'{key}_{m}{zs_str}'] = \
                        args.dqg_llmgen_qa_validation[key].replace('{3}', m)
                else: 
                    args.dqg_llmgen_qa_validation[f'{key}_{m}{zs_str}'] = \
                        args.dqg_llmgen_qa_validation[key].replace('{1}', m)

    ##### FOR TRAINING WITH ROUNDTRIP FILTERING #####
    args.dqg_llmgen_roundtrip_map = {
         'stv-picked-ge-ll-ph-qw': 
            {'dqg_breakhigh': 'results/dqg_dqa/cqadqa_breakhigh/cqadqa_breakhigh_llama_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_test_only-roundtrip_filtering{}',
             'dqg_musique':   'results/dqg_dqa/cqadqa_musique/cqadqa_musique_llama_prmptv2_dqg_llmgenstv-picked-ge-ll-ph-qw_test_only-roundtrip_filtering{}',}
    }

    args.roundtrip_filtering_qa = type(args.test_only) == str and args.test_only.startswith('roundtrip_filtering')
    if '-large' in args.model_name and args.chat_model: args.load_in_nbits = 4

    ##### LoRA settings ###########################
    lora_keys = ['lora_r', 'lora_bias', 'lora_alpha_ratio', 'lora_dropout']
    __ = {}
    for i, val in enumerate(args.lora_settings):
        key = lora_keys[i]
        if key == 'lora_bias': __[key] = val
        else: __[key] = eval(val)
    args.lora_settings = __
    if re.search(rf"{'^('+'|'.join(['gemma', 'llama', 'phi3', 'qwen']) + ')'}", args.model_name): 
        args.lora_settings['lora_r'] = 32
        if '-large' in args.model_name: args.lora_settings['lora_r'] = 16
    args.lora_settings['lora_alpha'] = args.lora_settings['lora_r'] * args.lora_settings['lora_alpha_ratio']
    if args.model_name.startswith('flan-t5'):
            args.lora_settings['peft_lora_keys'] = ['q', 'v', 'k', 'o', 'wi_0', 'wi_1', 'wo']#, 'lm_head', 'shared']
    elif re.search(rf"{'^('+'|'.join(['gemma', 'llama', 'qwen']) + ')'}", args.model_name): 
        args.lora_settings['peft_lora_keys'] = target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                               'down_proj', 'gate_proj', 'up_proj']
    elif re.search(rf"{'^('+'|'.join(['phi3']) + ')'}", args.model_name):
        args.lora_settings['peft_lora_keys'] = target_modules = ['qkv_proj', 'down_proj', 'gate_up_proj']
    if not args.do_lora: args.lora_settings = {k: None for k in args.lora_settings.keys()}
    ###############################################
    
    # NOTE / TODO: hardcoded here. check.
    args.max_epochs = 0 if not getattr(args, 'test_only', False) else args.max_epochs
    if args.test_only in ['False', 'True']: 
        args.test_only = eval(args.test_only)
    if args.test_only and args.load_ckpt_path is not None and '_lora' in args.load_ckpt_path: args.do_lora = True
    if args.roundtrip_filtering_qa: 
         args.use_dqg_llmgen = 'stv-picked-ge-ll-ph-qw'
    ##### Decoder-only DQG settings ###############
    args.qa_args = {}
    args.decomp_qg_args = {} if not args.task.startswith('dqg_') else args.decomp_qg_args
    if args.model_name.startswith('flan-t5'):
        args.padding_side   = 'right'
        args.prompt_version = 1
        args.load_in_nbits  = False
        args.do_lora        = False
        args.lora_settings  = []
        args.use_accelerate = False

    else: 
        args.padding_side   = 'left'
        args.max_epochs     = 2 
        args.bsz_train      = min(1, args.bsz_train)
        args.fp32           = False   
        args.num_accumulation_steps = 4
        args.lr             = 1e-5      
        args.lora_settings['lora_r'] = 64
        
        if args.task.startswith('dqg_'):
            args.decomp_qg_args = {key: eval(args.decomp_qg_args[i]) for i, key in enumerate(['n_shot', 'cot'])}
            if args.chat_model: args.bypass_predgen = True
            args.gen_args = {'max_new_tokens': 200, 'num_beams': 1, 'do_sample': False, 
                            'top_p': None, 'top_k': None, 'temperature': 0.0}
            
            if re.search(rf"{'^('+'|'.join(['llama', 'nvidia_llama', 'qwen']) + ')'}", args.model_name):
                args.gen_args['bos_token_id'] = None
            
            if args.decomp_qg_args['cot']: args.gen_args['max_new_tokens'] = 500
            args.user_prompt = "Complex question: " # used in prepare_decomp_qg_prompt()
            args.asst_prompt = "Decomposed sub-questions: " if not args.decomp_qg_args['cot'] else ''
        elif args.qa_task:
            args.qa_args    = {'n_shot': 3, 'cot': True, 'ans_markers': {'start': '[ANS_S]', 'end': '[ANS_E]',},
                               'max_new_tokens_cq': 200, 'max_new_tokens_sq': 50}
            am_start        = re.escape(args.qa_args['ans_markers']['start'])
            am_end          = re.escape(args.qa_args['ans_markers']['end'])
            regex_pattern   = re.compile(rf"(?:{am_start})(.+)(?:{am_end})")
            args.qa_args['ans_markers']['regex_pattern'] = regex_pattern
            args.gen_args   = {'max_new_tokens': 50, 'num_beams': 1, 'do_sample': False, 
                                    'top_p': None, 'top_k': None, 'temperature': 0.0,}
            
            if re.search(rf"{'^('+'|'.join(['llama', 'nvidia_llama', 'qwen']) + ')'}", args.model_name):
                args.gen_args['bos_token_id'] = None
            args.user_prompt = {'cq': 'Complex question: ', 'sq': 'Decomposed sub-question: '} # used in prepare_dqa_prompt()
            args.asst_prompt = {'cq': 'Answer to question: ', 'sq': 'Answer to sub-question: '}
    ###############################################

    if args.resume_ckpt_path == 'None': args.resume_ckpt_path = None
    args.c_dqg_hotpotqa_bypass  = args.task in ['dqg_hotpotqa']
    if args.cross_domain == 'False': args.cross_domain = False
    else: 
        assert args.cross_domain in ['musique', 'breakhigh', '2wikimultihop']
        assert args.cross_domain not in args.task, f'Cross-domain {args.cross_domain} cannot be same as task dataset {args.task}'

    if args.test_only != False and args.load_ckpt_path is None and not args.qa_task:
        ckpt_dp = f'results/dqg_dqa'
        if CHECKPOINT_MAPPING[args.model_name].get(args.task): 
            args.load_ckpt_path = f'{ckpt_dp}/{args.task}/' + CHECKPOINT_MAPPING[args.model_name].get(args.task)
            
            if args.use_dqg_rtfilt: 
                args.load_ckpt_path = args.load_ckpt_path.format(f'_{args.use_dqg_rtfilt}-cut{args.rtfilt_cutoff}')
            else:                  
                args.load_ckpt_path = args.load_ckpt_path.format('')
            assert os.path.exists(args.load_ckpt_path), f'👀 CHECK: checkpoint for {args.task} not found here: {args.load_ckpt_path}'
            print(f'👀 CHECK: checkpoint for {args.task} found here: {args.load_ckpt_path}')
    else: args.load_ckpt_path = None

    if args.test_only: args.num_gpu = 1
    print(f'👀 CHECK: USING {args.num_gpu}x gpus')
    args.num_gpu = min(args.num_gpu, torch.cuda.device_count())
    
    if args.model_name.startswith('flan-t5'):
        args.use_t5_legacy, args.use_fast_tokenizer = False, False
    else: args.use_t5_legacy, args.use_fast_tokenizer = False, True
    if args.load_ckpt_path == 'None': args.load_ckpt_path = None
    if args.resume_ckpt_path == 'None': args.resume_ckpt_path = None
    if args.exp_str == 'None': args.exp_str = ''
    
    # args.warmup_ratio = 1.0/args.max_epochs 
    args.warmup_ratio = 0
    if args.num_gpu > 0: args.warmup_ratio = args.warmup_ratio/args.num_gpu 

    if args.chat_model: 
        multiple = 32 if args.test_only else 16
        args.bsz_test = min(32, args.bsz_train * multiple)
        if '-large' in args.model_name: 
            if args.qa_task:
                args.bsz_train = 8
                args.bsz_test = 8
            else: 
                args.bsz_train = 1
                args.bsz_test = 8
    else:               
        args.bsz_test = min(256, args.bsz_train * 2)
    if args.roundtrip_filtering_qa and args.qa_task: 
        args.bsz_train  = 32
        args.bsz_test   = 32
    if args.resume_ckpt_path:
        args.warmup_ratio_start = args.warmup_ratio
        args.warmup_ratio = 0

    args.src_prompt = 'Decompose this complex question into simpler sub-questions: '

    torch.set_float32_matmul_precision('highest')
    print('👀 CHECK: FP32', args.fp32)
    if args.do_trial: 
         args.bsz_test = 2
         args.max_epochs = 1
         args.num_accumulation_steps = 1

    args.operators = ['[aggregate]', '[arithmetic]', '[boolean]', '[comparative]',
    '[comparison]', '[discard]', '[filter]', '[group]', '[intersection]',
    '[none]', '[project]', '[select]', '[superlative]', '[union]', 'JOIN', 'UNION']
    args.qpos_tokens = [f'[SQ{i+1}]' for i in range(20)]
    
           
    ### RUN MAIN ###
    print('👀 CHECK: BSZ_TRAIN', args.bsz_train)
    print('👀 CHECK: BSZ_TEST ', args.bsz_test)
    main(args) 
