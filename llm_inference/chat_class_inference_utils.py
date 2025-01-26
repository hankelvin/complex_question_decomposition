import torch, os, re
import numpy as np, math
from collections import Counter
from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                            AutoModelForSeq2SeqLM, pipeline)
import sys
sys.path.append('tools/rank_llm/src/rank_llm/rerank')

def load_model(args):
    device = args.device
    if device =='cuda':     attn_implementation = 'flash_attention_2'
    else:                   attn_implementation = None
    # https://github.com/huggingface/transformers/issues/32848, very slow with FA
    # if args.model_name == 'gemma' and args.model_lineup == 2: attn_implementation = 'flash_attention_2'
    if args.model_name in ['flan']: attn_implementation = None # flash attention not ready for enc-dec models
    # rerank tasks are effectively batch size 1 in rankllm setup
    if attn_implementation == 'flash_attention_2' \
        and (not args.rerank_task and args.bsz > 1):
        raise ValueError('🚨\t\tBatch size must be 1 for efficient use of Flash Attention 2.')

    cache_loc = f'llm_models/{args.model_name}'
    
    c_enc_dec = args.model_name in ['flan']
    model_id, padding_side, load_in_4bit, load_in_8bit = give_model_loading_params(args)
    
    cache_dir = f'{cache_loc}/{model_id}'
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    if args.model_name in ['phi3', 'allenai']: 
        trust_remote_code = {'trust_remote_code': True}
    else: trust_remote_code = {}

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = padding_side, 
                                    token = args.hf_token if args.hf_token else None,
                                    **trust_remote_code)

    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig        
        bnb_args = {'load_in_4bit': load_in_4bit, 'load_in_8bit': load_in_8bit}
        if args.model_size == 'large': 
            # https://huggingface.co/blog/4bit-transformers-bitsandbytes
            if args.model_name in ['llama', 'llama31', 'commandr_plus']: 
                assert load_in_4bit, f'Large {args.model_name.upper()} model should be loaded in 4-bit quantisation'
            elif args.model_name in ['flan']:
                assert load_in_8bit, f'Large {args.model_name.upper()} model should be loaded in 8-bit quantisation'
            
            if load_in_4bit:
                bnb_args['bnb_4bit_quant_type'] = "nf4"
                bnb_args['bnb_4bit_use_double_quant'] = True
                bnb_args['bnb_4bit_compute_dtype'] = torch.bfloat16
        quantization_config = BitsAndBytesConfig(**bnb_args)
    else: quantization_config = {} if args.model_name == 'commandr_plus' else None
    # other set-able params for BitsAndBytesConfig: 
    # llm_int8_threshold, llm_int8_skip_modules, llm_int8_enable_fp32_cpu_offload
    # llm_int8_has_fp16_weight, bnb_4bit_compute_dtype, bnb_4bit_quant_type,
    # bnb_4bit_use_double_quant, bnb_4bit_quant_storage
    print(f'QUANTISATION SETTING: load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}')
    
    model_class =  AutoModelForSeq2SeqLM if c_enc_dec else AutoModelForCausalLM
    model_args = trust_remote_code | {'cache_dir': cache_dir, 
                                        'torch_dtype': torch.bfloat16,
                                        'attn_implementation': attn_implementation, 
                                        'token': args.hf_token if args.hf_token else None,
                                        'quantization_config': quantization_config} 
    # if args.model_size == 'large': model_args.pop('torch_dtype')
    model = model_class.from_pretrained(model_id, **model_args)
    
    ###### add pad token ##############################################
    add_tokens = []
    c1 = padding_side == 'left'
    c2 = args.load_ckpt_path is not None
    c3 = tokenizer.pad_token is None
    added_pad_token = False  # check if padding token is already added to tokenizer. If not, add it.
    if c1 and c2 and c3: 
        from transfomers import AddedToken
        # for models with no padding token, we add a padding token to the tokenizer
        # see https://huggingface.co/docs/transformers/model_doc/llama3
        args.pad_token = "<|pad|>"
        add_tokens.append(AddedToken(args.pad_token, rstrip = False, lstrip = False, special = True))
        added_pad_token = True
    else: args.pad_token = tokenizer.pad_token
    tokenizer.add_tokens(add_tokens)
    print('👀 TOKENIZER CREATED', args.model_name, 'add_tokens', add_tokens)
    if getattr(args, 'pad_token', None) is not None: 
        tokenizer.pad_token     = args.pad_token
        tokenizer.pad_token_id  = tokenizer.convert_tokens_to_ids(args.pad_token)

    # resize model, set model to self
    if len(tokenizer.get_vocab()) > model.config.vocab_size: 
        print('resizing embedding to:', len(tokenizer.get_vocab()))
        model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    if added_pad_token:
        emb_settings = {}
        for key in ['max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse']:
            emb_settings[key] = getattr(model.model.embed_tokens, key, None)

        # ensure that the Embedding returns zeros for added pad taken 
        emb = torch.nn.Embedding(*model.model.embed_tokens.weight.shape, 
                         padding_idx = tokenizer.pad_token_id,
                         dtype = model.model.embed_tokens.weight.dtype)
        for key, value in emb_settings.items():
            setattr(emb, key, value)
        model.model.embed_tokens.weight.data = emb.weight.data    
        model.model.config.pad_token_id = tokenizer.pad_token_id
    ###################################################################

    do_torch_compile = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] in (7,8,9):
            do_torch_compile = True
    if do_torch_compile:
        # https://huggingface.co/docs/transformers/main/perf_torch_compile
        print('🔥\tdoing torch.compile')
        model = torch.compile(model)
        print('🔥\ttorch.compile done.')
    model.eval()
    
    # ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
    c_bnb = load_in_4bit or load_in_8bit
    if not c_bnb: model.to(device)
    model.eval()
    
    pipeline_model = pipeline('text2text-generation' if c_enc_dec else 'text-generation', 
                            model = model, tokenizer = tokenizer,
                            device = device if not c_bnb else None, )

    return tokenizer, pipeline_model, device, args

def load_model_rankllm(args):
    import sys, torch
    sys.path.append('tools')
    from rank_llm.src.rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
    from rank_llm.src.rank_llm.rerank.reranker import Reranker
    print('🔮\tLoading RankLLM Model:', args.ranker_args)
    
    model_id, padding_side, load_in_4bit, load_in_8bit = give_model_loading_params(args)
    # NOTE: bypass so that prompts for DQG are loaded properly
    c_bypass_fsc_load = True #if args.model_name not in ['rank_zephyr'] \
        # or args.task in ['rerank_dqg_pairwise', 'rerank_dqg_listwise']  else False
    if c_bypass_fsc_load: tokenizer, pipeline_model, device, args = load_model(args)

    ###### CONTROLLED GENERATION #####
    # A. constrained decoding controls (generate only tokens in ranking labels prediction sequence)
    # 1. settings to identify non-prediction sequence tokens to set logits to -inf
    # 2. settings to limit the number of tokens generated to prediction sequence length plus a little
    assert getattr(args,'rank_pred_seq', None) is not None, '\t🚨Rank Prediction Sequence not set.'
    rank_pred_seq = args.rank_pred_seq
    args.rank_pred_seq_tokens = tokenizer.encode(rank_pred_seq, add_special_tokens = False)
    args.rank_pred_seq_tokens_dec = [tokenizer.decode(t) for t in args.rank_pred_seq_tokens]
    print(f'🔮\tRank Prediction Sequence Tokens: {args.rank_pred_seq_tokens_dec}')

    # a. set max tokens
    # args.ranker_args['max_new_tokens'] # NOTE: max_new_tokens will be set inside RankListwiseOSLLM (using window_size)
    args.gen_args['max_new_tokens'] = len(args.rank_pred_seq_tokens) + 5
    if args.model_name in ['phi3']: args.gen_args['min_new_tokens'] = len(args.rank_pred_seq_tokens)
    if args.model_name in ['phi3'] and getattr(tokenizer, 'vocab', None) is None: 
        tokenizer.vocab = {tokenizer.decode(i): i for t,i in tokenizer.get_vocab().items()}
    
    args.contraints_pred_seq = None
    args.pred_seq_num_pos, args.pred_seq_num_idxes = [], []
    

    # B. add Constrained Beam Search if specified (for ranking tasks)
    # constraints: list of PhrasalConstraint objects (each a sequence of rank labels, e.g. ['[01]', ..., '[20]')
    if args.rerank_constrained_bs[0]: 
        from transformers import PhrasalConstraint
        rank_labels = args.rank_pred_seq.split(' > ')
        constraints = [PhrasalConstraint(tokenizer(r_l, add_special_tokens=False).input_ids) for r_l in rank_labels]
        num_beams = args.rerank_constrained_bs[1]
        args.gen_args = args.gen_args | {'constraints': constraints, 'num_beams': num_beams,}
    ###### CONTROLLED GENERATION #####

    rank_agent = RankListwiseOSLLM(model            = model_id, 
                                   bypass_fsc_load  = True if c_bypass_fsc_load else False,
                                   hf_model         = pipeline_model.model if c_bypass_fsc_load else None, 
                                   tokenizer        = tokenizer if c_bypass_fsc_load else None,
                                   **args.ranker_args)
    # args below: used at inference
    args.ranker_args = args.ranker_args | {'shuffle_candidates': False, 
                                           'print_prompts_responses': False,
                                           # from https://github.com/castorini/rank_llm/blob/main/src/rank_llm/retrieve_and_rerank.py
                                           'step_size': args.ranker_args_settings['step_size'], 
                                           'top_k_candidates': args.ranker_args_settings['top_k_candidates']} 
    # pipeline_model is the reranker
    pipeline_model = Reranker(rank_agent)

    return pipeline_model, args

def update_args_with_ranker_args(args):
    dataset, split = args.input_file
    if args.task in ['rerank'] or args.task in args.rerank_dqg_tasks:
        step_size = 5
        # only 10 or 20 in our case (20: musique, 10: others)
        if dataset in ['musique']:  window_size = top_k_candidates = getattr(args, 'num_cands', 10)
        else:                       window_size = top_k_candidates = getattr(args, 'num_cands', 10)
    elif args.rerank_task and 'pairwise' in args.task and args.task != 'rerank': 
        window_size, step_size, top_k_candidates = 2, 1, args.num_cands
    elif args.rerank_task and 'listwise' in args.task and args.task != 'rerank':
        top_k_candidates = args.num_cands
        window_size = step_size = args.num_cands
    else: raise NotImplementedError

    args.ranklabel_window_size = window_size
    # NOTE: window_size should be equals to args.num_cands
    args.rank_pred_seq = " > ".join([f"[{str(i+1).zfill(2)}]" for i in range(window_size)])
    print('\t🔍Example rank prediction sequence string:', args.rank_pred_seq)

    args.ranker_args = {'context_size': 4096, 
            'num_few_shot_examples': args.num_few_shot_examples, 
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_gpus': 1, 'variable_passages': True, 'window_size': window_size,
            'system_message': None, 'rerank_task_name': args.task,
            'rerank_dqg_tasks': args.rerank_dqg_tasks,
            'qpos_tokens': args.qpos_tokens,
            'constraints_dict': {'constraints': args.gen_args.get('constraints', None),
                                 'num_beams': args.gen_args.get('num_beams', None)},
            'rerank_with_score_scheme': getattr(args, 'rerank_with_score_scheme', False),
            'prompt_version': args.prompt_version,}
    if torch.backends.mps.is_available(): args.ranker_args['device'] = 'mps'
    args.ranker_args_settings = {'step_size': step_size, 'window_size': window_size,
                                 'top_k_candidates': top_k_candidates}

    return args

def give_model_loading_params(args):
    load_in_4bit = load_in_8bit = False
    
    if args.model_name == 'flan':
        padding_side = 'right'

        if args.model_size == 'large':
            model_id = 'google/flan-ul2'    # 20B enc-dec
            print('LARGE MODEL LOADING...', model_id)
            load_in_4bit = False
            load_in_8bit = True
        else: 
            model_id = 'google/flan-t5-xxl'  # 11B enc-dec
            load_in_4bit = False
            load_in_8bit = True
    
    elif args.model_name == 'gemma':
        padding_side = 'left'
        if   args.model_lineup == 1: model_id = 'google/gemma-1.1-7b-it'
        elif args.model_lineup == 2: model_id = 'google/gemma-2-9b-it'
    
    elif args.model_name == 'llama':
        padding_side = 'left'
        
        if args.model_size == 'large':
            if   args.model_lineup == 1: model_id = 'meta-llama/Meta-Llama-3-70B-Instruct'
            elif args.model_lineup == 2: model_id = 'meta-llama/Llama-3.1-70B-Instruct'
            print('LARGE MODEL LOADING...', model_id)
            load_in_4bit = True
            load_in_8bit = False
        else:
            if   args.model_lineup == 1: model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
            elif args.model_lineup == 2: model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        
    elif args.model_name == 'phi3':
        padding_side = 'left'
        if   args.model_lineup == 1: model_id = 'microsoft/Phi-3-mini-128k-instruct'
        elif args.model_lineup == 2: model_id = 'microsoft/Phi-3.5-mini-instruct'
        
    elif args.model_name == 'rank_zephyr':
        padding_side = 'left' 
        model_id = 'castorini/rank_zephyr_7b_v1_full'

    elif args.model_name == 'gritlm_gen':
        padding_side = 'left' 
        model_id = 'GritLM/GritLM-7B'

    elif args.model_name == 'mistral':
        padding_side = 'left' 
        if   args.model_lineup == 1: model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
        elif args.model_lineup == 2: model_id = 'mistralai/Mistral-7B-Instruct-v0.3'

    elif args.model_name == 'aya':
        padding_side = 'left'
        model_id = 'CohereForAI/aya-23-8B'
        
    elif args.model_name == 'commandr_plus':
        # NOTE: 104B billion parameter already in 4-bit quantisation
        padding_side = 'left'
        model_id = 'CohereForAI/c4ai-command-r-plus-4bit'

    elif args.model_name == 'qwen':
        padding_side = 'left'
        if args.model_size == 'large':
            if   args.model_lineup == 1: raise ValueError
            elif args.model_lineup == 2: model_id = 'Qwen/Qwen2.5-72B-Instruct'
            print('LARGE MODEL LOADING...', model_id)
            load_in_4bit = True
            load_in_8bit = False
        else:
            if   args.model_lineup == 1: model_id = 'Qwen/Qwen1.5-7B-Chat'
            elif args.model_lineup == 2: model_id = 'Qwen/Qwen2.5-7B-Instruct'

    elif args.model_name == 'nvidia_llama3':
        padding_side = 'left'
        model_id = 'nvidia/Llama3-ChatQA-1.5-8B'

    elif args.model_name == 'olmo':
        padding_side = 'left'
        model_id = 'allenai/OLMo-7B-0724-Instruct-hf'

    else: 
        padding_side='right'
        raise NotImplementedError

    return model_id, padding_side, load_in_4bit, load_in_8bit


def get_nvidia_llama3_formatted_input(messages, context, add_generation_prompt = False):
    # see https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) 
    ### CHANGE START ###
    if add_generation_prompt: conversation += "\n\nAssistant:"
    ### CHANGE END ###
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input

def give_save_path(args):
    args.save_path  = f'{args.save_loc}/{args.task}/{args.task}'
    
    for k,v in args.decomp_qg_args.items():
        if k == 'cot': 
            if v: args.save_path += '_CoT'
        else: args.save_path += f'_{k}-{v}'
    
    args.save_path += f'_{args.input_file[0]}_{args.input_file[1]}_{args.model_name}'
    if args.model_lineup > 1: args.save_path += f'_modelline-{args.model_lineup}'
    if args.prompt_version > 1: args.save_path += f'_prmptv{args.prompt_version}'
    if args.num_few_shot_examples > 0: args.save_path += f'_rankllm-nshot-{args.num_few_shot_examples}'
    if args.model_size == 'large': args.save_path += f'_{args.model_size}'
    if args.rerank_task:
        args.rerank_dqg_models = sorted([m for m in args.rerank_dqg_models if m])
        if args.rerank_dqg_models:
            for i, m in enumerate(args.rerank_dqg_models):
                if i == 0:  args.save_path += f'-{m[:2].upper()}'
                else:       args.save_path += f'-{m[:2].upper()}'
    
    if args.rerank_constrained_bs[0]:
        args.save_path += f'_CBS-nb{args.rerank_constrained_bs[1]}'

    if args.rerank_with_score_scheme:
        args.save_path += '_sc-sch'
    
    if args.do_trial: 
        args.save_path += '_TRIAL'

    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)
        print('\t🔍SAVEPATH created', args.save_path)
    else: print('\t🔍SAVEPATH exists', args.save_path)

    return args

def clean_sysname(sysname):
    if '_modelline'in sysname:  sysname = sysname.split('_modelline')[0]
    if '_prmptv'in sysname:     sysname = sysname.split('_prmptv')[0]
    return sysname

def give_rank_docmap_v1(pred_candidates, batch_candidates):
    # e.g. '3hop2__30152_107291_20999_doc1'
    try:    ranking = [re.search(r'(?:_doc)(\d+)', r.docid).group(1) for r in pred_candidates]
    except: ranking = [re.search(r'(?:_doc)(\d+)', r.docid).group(1) for r in pred_candidates]
    docid_map = {i+1: (c.docid, c.doc) for i, c in enumerate(batch_candidates)}
    return ranking, docid_map
        
def give_rank_docmap_v2(pred_candidates, batch_candidates, skip_sys_prefix = False):
    if skip_sys_prefix:
        ranking = [r.docid for r in pred_candidates]
        docid_map = {c.docid: (c.docid, c.doc) for i, c in enumerate(batch_candidates)}
    
    else:
        # e.g. '3hop2__30152_107291_20999_sys-llama'
        ranking = [re.search(r'(?:sys-)(\w+)', r.docid).group(1) for r in pred_candidates]
        docid_map = {re.search(r'(?:sys-)(\w+)', c.docid).group(1) : \
                (c.docid, c.doc) for i, c in enumerate(batch_candidates)}
    return ranking, docid_map