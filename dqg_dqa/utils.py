import torch, copy, json, os, datetime, tqdm, re
import torch.nn as nn
import functools, logging
LOGGING_LEVEL = logging.INFO

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import lightning.pytorch as pl
import sys
currpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(currpath, '..'))
from evaluation.evaluate_utils import evaluate, clean_special_tokens

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping 
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class DQGModel_Accelerate(torch.nn.Module):
    def __init__(self, args, tokenizer, model, train_dataloader, val_dataloader, test_dataloader):
        super().__init__()
        self.args       = args
        ### load and set pretrained weights ###
        self.tokenizer  = tokenizer
        self.model      = model
        self.train_dl   = train_dataloader
        self.val_dl     = val_dataloader
        self.test_dl    = test_dataloader

        if self.args.num_accumulation_steps > 1:
            # esp for peft: https://github.com/huggingface/peft/issues/1142
            # https://github.com/huggingface/peft/issues/1142
            self.model.gradient_checkpointing_enable() 

        self.print_func     = args.logger.print if getattr(args, 'logger', False) else print
        self.pbar_update    = 20
        self.pbar           = None
        self.answering_method = []
        if   self.args.task.startswith('cqa_'):
            self.answering_method = ['cq']
        elif self.args.task.startswith('dqa_'):
            self.answering_method = ['sq']
        elif self.args.task.startswith('cqadqa_'):
            self.answering_method = ['cq', 'sq']
        
        ##### PEFT setup #####
        if args.do_lora:
            adapter_name = 'default'

            # either make model peft or load peft ckpt weights
            if self.args.load_ckpt_path is not None:
                if '_lora' not in args.load_ckpt_path: raise ValueError('CHECK THAT CKPT WEIGHTS ARE PEFT WEIGHTS')
                print('🔮Loading LoRA adapter from checkpoint path: ', args.load_ckpt_path)
                
                self.model.load_adapter(args.load_ckpt_path, adapter_name = adapter_name)
                self.model.enable_adapters()
                self.model.set_adapter(adapter_name)

            else: 
                from peft import get_peft_model, LoraConfig, TaskType
                target_modules = []
                peft_lora_keys = args.lora_settings.get('peft_lora_keys', [])
                assert peft_lora_keys, f'🚨\t\tNo PEFT LoRA keys provided. {peft_lora_keys}'
                for k in peft_lora_keys:
                    for name, __ in self.model.named_modules():
                        if k == name and k in ['lm_head', 'shared']: target_modules.append(name)
                        if name.endswith(f'.{k}'): target_modules.append(name)
                print('\t🔍target_modules:\n', target_modules)

                task_type = TaskType.SEQ_2_SEQ_LM if args.padding_side == 'right' else TaskType.CAUSAL_LM
                    
                peft_config = LoraConfig(target_modules = target_modules,
                                        task_type      = task_type, 
                                        inference_mode = False,
                                        r              = args.lora_settings['lora_r'], 
                                        lora_alpha     = args.lora_settings['lora_alpha'], 
                                        lora_dropout   = args.lora_settings['lora_dropout'],
                                        use_rslora     = True,)
                
                self.model = get_peft_model(self.model, peft_config, adapter_name = adapter_name)
                # see https://github.com/huggingface/peft/issues/1142
                if self.args.num_accumulation_steps > 1:
                    self.model.enable_input_require_grads()
                self.model.set_adapter(adapter_name)
                print(f'🔮ADAPTER ({adapter_name}) ADDED TO MODEL')
                self.model.print_trainable_parameters()       
        ###################### 

        ######################
        # NOTE: this differs from llm_inference (we add pad_token)
        gen_args = self.args.gen_args
        if args.model_name .startswith('llama'):
            terminators = [self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            # https://github.com/huggingface/transformers/issues/29378
            # self.tokenizer.pad_token_id     = self.tokenizer.eos_token_id
            # self.model.config.pad_token_id  = self.tokenizer.eos_token_id
            gen_args = gen_args | {'eos_token_id': terminators, 
                            'pad_token_id': self.tokenizer.pad_token_id,}
        elif args.model_name == ['phi3', 'gritlm_gen']:
            gen_args = gen_args | {'pad_token_id': self.tokenizer.pad_token_id,}
        elif args.model_name in ['mistral', 'qwen', 'nvidia_llama3']:
            gen_args = gen_args | {'pad_token_id': self.tokenizer.pad_token_id,}
        self.args.gen_args = gen_args
        ######################
        
        # optimizer, scheduler
        warmup_min_lr = 0.0
        args.training_steps = len(self.train_dl) if train_dataloader is not None \
            and self.train_dl.dataset is not None else 0
        if getattr(self.args, 'training_steps', False):
            training_steps = self.args.training_steps
            warmup_num_steps = self.args.warmup_ratio * training_steps
        else: training_steps, warmup_num_steps = 100000, 0
        
        self.scheduler_info = \
            {"type":    'linear_warmup',
            "params":   {"last_batch_iteration": -1, "total_num_steps": training_steps,
                        "warmup_min_lr": warmup_min_lr, "warmup_max_lr": self.args.lr,
                        "warmup_num_steps": warmup_num_steps, 'warmup_type': 'linear'}}
    
        ##### Setting/Controls #####     
        # removing eos, pad for prints and saves
        cleantokens = [self.tokenizer.pad_token, self.tokenizer.eos_token]
        if self.tokenizer.bos_token is not None: cleantokens.append(self.tokenizer.bos_token)                                   
        self.c_s_p = functools.partial(clean_special_tokens, tokens = cleantokens)

        # BLEU validation 
        self.eval_em    = self.args.task.startswith('dqg_') and not self.args.chat_model
        self.eval_bleu  = not self.eval_em
        # do not include task prefixes for eval_bleu
        cleantokens_eval_bleu_em = copy.copy(cleantokens)
        if getattr(self.tokenizer, 'task_prompt_tokens', False):
            for task, task_prompt_tokens in self.tokenizer.task_prompt_tokens.items(): 
                cleantokens_eval_bleu_em.append(self.tokenizer.decode(task_prompt_tokens))
        self.c_s_p_eval_bleu_em = functools.partial(clean_special_tokens, tokens = cleantokens_eval_bleu_em)

        self.training_step_outputs      = []
        self.validation_step_outputs    = []
        self.test_step_outputs          = []
        self.val_losses                 = {}
        self.train_losses               = {}

        self.loss_fct = nn.CrossEntropyLoss(ignore_index = -100, 
                            label_smoothing = getattr(args,'label_smoothing', 0.0),
                            reduction = 'none')
        
        ##### Accelerator #####
        if args.use_accelerate:
            from accelerate import Accelerator
            from accelerate.utils import GradientAccumulationPlugin
            if self.args.num_accumulation_steps > 1: 
                grad_accum_plugin = GradientAccumulationPlugin(num_steps = self.args.num_accumulation_steps, 
                                                           sync_with_dataloader = False)
            else: grad_accum_plugin = None
            self.accelerator = Accelerator(mixed_precision  = 'no' if self.args.fp32 else "bf16",
                                            rng_types       = ['torch', 'cuda'], log_with = 'tensorboard',
                                            project_dir     = self.args.save_dir, 
                                            device_placement = False, 
                                            gradient_accumulation_plugin = grad_accum_plugin,
                                            step_scheduler_with_optimizer = False)
            self.device = self.accelerator.device
            if not args.load_in_nbits: self.model.to(self.device)

            optims = self.configure_optimizers()[0]
            self.optimizer = optims['optimizer']
            self.scheduler = optims['lr_scheduler']['scheduler']
            acc_out = self.accelerator.prepare(self.model, self.optimizer, self.scheduler,
                                                self.train_dataloader(),
                                                self.val_dataloader(),
                                                self.test_dataloader(), )
            self.model, self.optimizer, self.scheduler, \
                self.train_loader, self.val_loader, self.test_loader = acc_out
            
            if self.args.resume_ckpt_path is not None:
                self.accelerator.load_state(args.resume_ckpt_path)
            
            # https://huggingface.co/docs/accelerate/usage_guides/checkpoint
            # Register the LR scheduler
            self.accelerator.register_for_checkpointing(self.scheduler)
            # Save the starting state
            if self.args.test_only == False:
                self.accelerator.save_state(output_dir = self.args.save_dir, safe_serialization = False)
        #######################
    
    def do_acc_train(self, epoch, val_step = False):
        step = 'val' if val_step else 'train'
        losses = []
        dataloader = getattr(self, f'{step}_dataloader')()
        self.pbar = tqdm.tqdm(total = len(dataloader))
        for batch_idx, batch in enumerate(dataloader):
            if val_step: 
                loss = self.validation_step(batch, batch_idx)
            else: 
                loss = self.training_step(batch, batch_idx, val_step)
            losses.append(loss['loss'])
            self.pbar.update(1)
            if batch_idx % self.pbar_update == 0:
                self.pbar.set_description(f"📈📈{step.upper()} loss: {round(losses[-1],2)}")
        getattr(self, f'{step}_losses')[epoch] = losses
        print(f'\t🟩Epoch {epoch} {step.upper()} loss: {sum(losses)/len(losses)}')
        self.pbar = None
        
        if not val_step: self.on_train_epoch_end()

    def do_acc_validate(self, epoch):
        with torch.no_grad():
            self.do_acc_train(epoch, val_step = True)

        # pbar added to in do_acc_train
        pbar = self.on_validation_epoch_end()['progress_bar']
        if self.eval_bleu:
            if not getattr(self, 'eval_bleus', False): self.eval_bleus = {}
            self.eval_bleus[epoch] =  pbar['eval_bleu']
        if self.eval_em: 
            if not getattr(self, 'eval_ems',   False): self.eval_ems   = {}
            self.eval_ems[epoch]   = pbar['eval_em']

    def do_acc_test(self):
        dataloader = getattr(self, f'test_dataloader')()
        self.pbar = tqdm.tqdm(total = len(dataloader))
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            self.test_step(batch, batch_idx)
            self.pbar.update(1)
        self.on_test_epoch_end()
        self.pbar = None

    def training_step(self, batch, batch_idx, val_step = False):
        pbar = {}
        src, src_mask, tgt, tgt_mask, cq, src_sq_str, sq_holder, id_enc = batch
        
        tdec, preds, gens = [], [], []

        num_logits_to_keep = None
        if self.args.chat_model:
            num_logits_to_keep = tgt.size(1)

            src_to_use      = torch.cat([src, tgt],           dim = 1).to(self.device)
            src_mask_to_use = torch.cat([src_mask, tgt_mask], dim = 1).to(self.device)
            
            outputs = self.model(input_ids     = src_to_use, 
                                attention_mask = src_mask_to_use, 
                                # labels         = tgt.to(self.device),
                                num_logits_to_keep = num_logits_to_keep)
            
            # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            # NOTE: computing loss outside model forward so that we can ignore padding
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = outputs.logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            shift_labels[tgt_mask[..., 1:] == self.args.mask_value] = self.loss_fct.ignore_index
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            numel = tgt_mask[..., 1:].sum().to(shift_logits.device)
            outputs.loss = self.loss_fct(shift_logits, shift_labels).sum()/numel
        
        else: 
            outputs = self.model(input_ids  = src.to(self.device), 
                            attention_mask = src_mask.to(self.device), 
                            labels         = tgt.to(self.device))
        
        lm_logits   = outputs.logits
        train_loss  = outputs.loss

        if self.args.use_accelerate and not val_step:
            self.accelerator.backward(train_loss)
            if (batch_idx + 1) % self.args.num_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        train_loss = train_loss.detach()
        if self.args.use_accelerate: train_loss = train_loss.item()
        pbar[f'train_loss'] = train_loss # validation_step below will rename this 

        do_print = batch_idx % 1000 == 0
        if do_print or (val_step and (self.eval_bleu or self.eval_em)):
            tdec, preds, gens = \
                self.give_predgen_training(src, src_mask, tgt, lm_logits, 
                                           batch_idx, val_step, do_print)
    
        for name, loss in pbar.items():
            if not self.args.use_accelerate: 
                self.log(name, loss, on_step = True, on_epoch = True, prog_bar = True, 
                        logger = True, sync_dist = True, batch_size = len(batch))
                for k,v in pbar.items(): pbar[k] = v.half().item()

        train_outputs = {'loss': train_loss, 'progress_bar': pbar, 
                         'tdec': tdec, 'preds': preds, 'gens': gens}
        
        self.training_step_outputs.append(train_outputs)
        return train_outputs
    
    def on_train_epoch_end(self):
        train_loss_epoch = \
            sum([x['progress_bar']['train_loss'] for \
                 x in self.training_step_outputs])/len(self.training_step_outputs)
        pbar = {'train_loss_epoch': train_loss_epoch}
        self.training_step_outputs.clear()
            
        return {'loss': train_loss_epoch, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx, val_step = True)
        results_copy = copy.deepcopy(results)
        to_del = []
        for k,v in results['progress_bar'].items():
            new_k = k.replace('train', 'validation')
            results_copy['progress_bar'][new_k] = copy.deepcopy(v)
            to_del.append(k)
            if not self.args.use_accelerate: 
                self.log(new_k, v, on_step = True, on_epoch = True, prog_bar = True, 
                        logger = True, sync_dist = True)
        
        for k in to_del: 
            del results_copy['progress_bar'][k]
        self.validation_step_outputs.append(results_copy)
        
        return results_copy
    
    def on_validation_epoch_end(self):
        validation_loss_epoch = \
            sum([x['progress_bar']['validation_loss'] \
                 for x in self.validation_step_outputs])/len(self.validation_step_outputs)
        
        gen5_bleu_em = None
        if self.eval_bleu or self.eval_em:
            if self.eval_bleu: auto_scores = ['bleu'] 
            elif self.eval_em: auto_scores = ['em']
            else: auto_scores = []

            references = [i for step in self.validation_step_outputs for i in step['tdec']]
            generated  = [i for step in self.validation_step_outputs for i in step['gens']]
            assert len(references) == len(generated)
            gen5_bleu_em = evaluate(references = references, generated = generated, 
                        idxes = None, lang = 'en', bert_score_models = [('bertbase', 'bert-base-uncased')],
                        auto_scores = auto_scores, strip_func = self.c_s_p_eval_bleu_em)
        
        pbar = {'validation_loss_epoch': validation_loss_epoch}
        if self.eval_bleu: 
            pbar['eval_bleu'] = gen5_bleu_em['test_bleu']
            if not self.args.use_accelerate: self.log('eval_bleu', gen5_bleu_em['test_bleu'])
        elif self.eval_em: 
            pbar['eval_em'] = gen5_bleu_em['test_em']
            if not self.args.use_accelerate: self.log('eval_em', gen5_bleu_em['test_em'])
        self.validation_step_outputs.clear()
        
        # TODO: add a way to run with epoch num (to add to filenames for saveout)
        # if self.args.qa_task: self.on_test_epoch_end(val_step = True)

        return {'loss': validation_loss_epoch, 'progress_bar': pbar}
    
    def test_step_qa(self, batch, batch_idx):
        c_qwen = self.args.model_name in ['qwen', 'qwen-large']
        clean_e_p_r = functools.partial(clean_eos_pad_role, 
                                        role     = 'assistant'  if c_qwen else '', 
                                        turn_end = '<|im_end|>' if c_qwen else '')
        clean_te = clean_turn_end if c_qwen else identity_text

        # 1. handle CQ answering (optionally with CoT and Few-shot)
        # 2. handle SQ answering by successively posing SQs to the model 
        # collecting the answers (tracking the answer for each SQ, replacing answer var
        # for each subsequent SQ) 
        src, src_mask, tgt, tgt_mask, cq, src_sq_str, sq_holder, id_enc = batch
        reg_pattern = self.args.qa_args['ans_markers']['regex_pattern']

        cq_ans_list = []
        c_cqa = 'cq' in self.answering_method
        if c_cqa:
            src         = src.to(self.device)       if src is not None      else src
            src_mask    = src_mask.to(self.device)  if src_mask is not None else src_mask
            tgt         = tgt.to(self.device)       if tgt is not None      else tgt
            self.args.gen_args['max_new_tokens'] = self.args.qa_args['max_new_tokens_cq']
            with torch.no_grad():
                gens        = self.model.generate(input_ids = src, 
                                                  attention_mask = src_mask, 
                                                  **self.args.gen_args)

            if self.args.chat_model: 
                src_num_tokens = src.size(1)
                gens = gens[:, src_num_tokens:]

            gens_cq     = self.tokenizer.batch_decode(gens, skip_special_tokens = True)
            cq_ans_list = [re.findall(reg_pattern, g) for g in gens_cq]

        elif not c_cqa and self.args.do_dqg_llmgen_qa is not None: # i.e. not round trip filtering
            gens_cq     = ['[None]' for __ in range(tgt.size(0))]
            tgt_dec     = self.tokenizer.batch_decode(tgt, skip_special_tokens = True)
            # NOTE: in 'cq' in self.answering_method above, cq_ans_list is a list of list
            # import to follow here, for eval later
            cq_ans_list = [[self.c_s_p_eval_bleu_em(t)] for t in tgt_dec]
        
        sq_ans_list = None
        gens_sq     = ['[None]' for __ in range(tgt.size(0))] # overwritten later if doing sq
        bsz = src.size(0)
        eos = self.tokenizer.eos_token
        pad = self.tokenizer.pad_token
        if 'sq' in self.answering_method:
            gens_sq     = ['' for __ in range(tgt.size(0))] # overwritten later if doing sq
            sq_ans_var_map  = {i: [] for i in range(src.size(0))}
            
            for sq_pos, __ in enumerate(sq_holder[0]):
                curr_sq_src_str = [sqh[sq_pos] for sqh in sq_holder]
                assert len(curr_sq_src_str) == bsz, (len(curr_sq_src_str), bsz)
                
                # a. add prefix, cq and 1st sq
                if sq_pos == 0: sq_src_strs = src_sq_str

                # b. do answer var replacement if present 
                if sq_pos > 0:               
                    curr_sq_src_str = [self.replace_answer_var(sq, sq_ans_var_map, elem_num, sq_pos) \
                        if sq is not None else None for elem_num, sq in enumerate(curr_sq_src_str)]   

                # if batch elem has no more SQs (i.e. None placed in collate_fn), set to ''
                __sq_src_strs = []
                empty_tracker = []
                assert len(curr_sq_src_str) == len(sq_src_strs), (len(curr_sq_src_str), len(sq_src_strs))
                for elem_num, (prefix, sq_turn) in enumerate(zip(sq_src_strs, curr_sq_src_str)):
                    if sq_turn is None:  
                        line = ''
                        empty_tracker.append(elem_num)
                    else:  
                        # NOTE: important to remove previous eos token
                        line = f"{clean_e_p_r(prefix, eos, pad)}\n{clean_te(sq_turn)}"
                        gens_sq[elem_num] = line # collect the prompt (replace as we go)
                    __sq_src_strs.append(line)     
                        
                sq_src_strs = __sq_src_strs

                # encode and pad (decoder-only tokenizer padding leading to padding warning)
                # NOTE: ensure no eos, else generation stops
                sq_src_enc = [self.tokenizer.encode(sq) for sq in sq_src_strs]
                sq_src_enc, sq_src_enc_mask = self.collate_obj_eval.pad2max(sq_src_enc, 
                                                    make_mask = True, pad_src = True)
                sq_src_enc = torch.LongTensor(sq_src_enc).to(self.device) 
                sq_src_enc_mask = torch.LongTensor(sq_src_enc_mask).to(self.device)

                self.args.gen_args['max_new_tokens'] = self.args.qa_args['max_new_tokens_sq']
                with torch.no_grad():
                    gens        = self.model.generate(input_ids = sq_src_enc, 
                                                      attention_mask = sq_src_enc_mask, 
                                                      **self.args.gen_args)
                
                if self.args.chat_model: 
                    src_num_tokens = sq_src_enc.size(1)
                    gens = gens[:, src_num_tokens:]

                # decode the answer and store to sq_ans_var_map 
                gens = self.tokenizer.batch_decode(gens, skip_special_tokens = True)

                # add answers to current sq_src_strs
                # NOTE: important to remove previous eos tokens, also remove pad tokens
                sq_src_strs = [f"{clean_e_p_r(s, eos, pad)} {clean_e_p_r(a, eos, pad)}" \
                                    for s, a in zip(sq_src_strs, gens)]
                
                for sqans_elem_num, ans in enumerate(gens):
                    if sqans_elem_num in empty_tracker: continue
                    # strip the marker token 
                    try:    ans = re.search(reg_pattern, ans).group(1).strip()
                    except: ans = None
                    if ans == '': ans = None                        
                    sq_ans_var_map[sqans_elem_num].append(ans)  

            sq_ans_list = [sq_ans_var_map[i] for i in range(len(sq_ans_var_map))]
            assert len(gens_sq) == len(sq_ans_list), (len(gens_sq), len(sq_ans_list))

        out_dict = {'tgt': tgt, 'src': src, 'cq': cq, 'id_enc': id_enc,
                    'gens_cq': gens_cq, 'gens_sq': gens_sq, 
                    'cq_ans_list': cq_ans_list, 'sq_ans_list': sq_ans_list}
        
        self.test_step_outputs.append(out_dict)
        return out_dict

    def replace_answer_var(self, sq, sq_ans_var_map, elem_num, sq_pos):
        answer_var = re.findall(r'(?:#)(\d+)', sq)
        answer_var = [int(v) for v in answer_var if int(v) < sq_pos+1]
        for av in answer_var: 
            # answer variables are 1-indexed, sq_ans_var_map is 0-indexed        
            try: 
                rep = sq_ans_var_map[elem_num][av-1]
                if rep is None: continue
                sq  = sq.replace(f'#{av}', rep)
            except: 
                print('\t\t\t (FAIL) replace ans var', answer_var, av, sq_ans_var_map[elem_num])
                pass
        return sq
        
    def test_step(self, batch, batch_idx):
        if self.args.qa_task: return self.test_step_qa(batch, batch_idx)
    
        src, src_mask, tgt, tgt_mask, cq, src_sq_str, sq_holder, id_enc = batch

        src         = src.to(self.device)       if src is not None      else src
        src_mask    = src_mask.to(self.device)  if src_mask is not None else src_mask
        tgt         = tgt.to(self.device)       if tgt is not None      else tgt
        with torch.no_grad():
            gens        = self.model.generate(input_ids = src, attention_mask = src_mask, 
                                              **self.args.gen_args)
        if self.args.chat_model: 
            src_num_tokens = src.size(1)
            gens = gens[:, src_num_tokens:]

        out_dict = {'gens': gens, 'tgt': tgt, 'src': src, 'cq': cq, 'id_enc': id_enc,}
        
        self.test_step_outputs.append(out_dict)
        return out_dict
    
    def on_test_epoch_end(self, val_step = False): 
        if val_step: phase_str, df_phase_str = 'validation', 'validation_'
        else:        phase_str, df_phase_str = 'test', ''

        if self.args.qa_task: 
            generated, references, idxes = self.one_step_out_dqa(val_step)
        else: 
            generated, references, idxes = self.one_step_out_dqg(val_step)

        scores = {}
        if self.eval_bleu: auto_scores = ['bleu'] 
        elif self.eval_em: auto_scores = ['em']
        else: auto_scores = []
        if self.args.qa_task: auto_scores = ['em', 'f1', 'bertscore']
        scores = evaluate(references = references, generated = generated, idxes = idxes, auto_scores = auto_scores,
                            lang = 'en', bert_score_models = [('bertbase', 'bert-base-uncased')],
                            strip_func = self.c_s_p_eval_bleu_em)
        with open(self.args.savepath + f'/{phase_str}_scores.json', 'w+', encoding = 'utf-8') as f:
            json.dump(scores, f)

        if self.args.qa_task: 
            pass
             
        else: 
            from evaluation.evaluate_dqg import prepare_for_dqg_eval
            self.args.save_path = self.args.savepath # it is save_path in prepare_for_dqg_eval
            df_holder = prepare_for_dqg_eval(args = self.args,
                                            not_chat = True if self.args.model_name.startswith('flan-t5') else False)
            
            for src_type, df in df_holder.items():
                df_label, df_pred = df['label'], df['pred']
                df_label.to_csv(os.path.join(self.args.savepath, f'{df_phase_str}{src_type}_labels.csv'), index = False)
                df_pred.to_csv(os.path.join(self.args.savepath, f'{df_phase_str}{src_type}_predictions.csv'), index = False)

    def one_step_out_dqg(self, val_step = False):
        if val_step: phase_str = 'validation'
        else:        phase_str = 'test'
        
        test_step_outputs = self.test_step_outputs
        generated, references, idxes = [], [], []

        tokenizer = self.tokenizer
        pad_token = tokenizer.pad_token
        with open(self.args.savepath + f'/{phase_str}_out.txt', 'w+', encoding = 'utf-8') as f:
            for step_items in test_step_outputs:
        
                g = tokenizer.batch_decode(step_items['gens'], skip_special_tokens = False)
                cq = tokenizer.batch_decode(step_items['cq'], skip_special_tokens = False)
                if not self.args.c_dqg_hotpotqa_bypass: 
                    t = tokenizer.batch_decode(step_items['tgt'], skip_special_tokens = False)
                else: t = ['None' for _ in range(len(g))]
                i = tokenizer.batch_decode(step_items['id_enc'], skip_special_tokens = True)
                s = tokenizer.batch_decode(step_items['src'], skip_special_tokens = False)
                
                s = [ss.replace(pad_token, '') for ss in s]
                generated.extend(g), references.extend(t), idxes.extend(i)
                assert len(g) == len(t) == len(s) == len(cq) == len(i), (len(g), len(t), len(s), len(cq), len(i))
                zipped = zip(g, t, s, cq, i)
                for gg, tt, ss, ccqq, ii in zipped: 
                    gg = replace_nlines_tabs(gg)
                    tt = replace_nlines_tabs(tt)
                    ccqq = self.c_s_p_eval_bleu_em(ccqq.replace('\n', ' ').replace('\t', ' '))
                    f.write(f'{ii}\t{(ccqq)}\t{(self.c_s_p_eval_bleu_em(tt))}\t{self.c_s_p_eval_bleu_em(gg)}\n')

        return generated, references, idxes
    
    def one_step_out_dqa(self, val_step = False):
        if val_step: phase_str = 'validation'
        else:        phase_str = 'test'
    
        test_step_outputs = self.test_step_outputs
        generated, references, idxes = [], [], []
        
        # out_dict is of the form:
        # out_dict = {'tgt': tgt, 'src': src, 'cq': cq, 'id_enc': id_enc,
        #             'gens_cq': gens_cq, 'cq_ans_list': cq_ans_list, 
        #             'sq_ans_list': sq_ans_list}

        tokenizer   = self.tokenizer
        eos         = tokenizer.eos_token
        pad         = tokenizer.pad_token
        with open(self.args.savepath + f'/{phase_str}_out.txt', 'w+', encoding = 'utf-8') as f:
            for step_items in test_step_outputs:
        
                cq = tokenizer.batch_decode(step_items['cq'], skip_special_tokens = False)
                if not self.args.c_dqg_hotpotqa_bypass: 
                    t = tokenizer.batch_decode(step_items['tgt'], skip_special_tokens = False)
                else: t = ['None' for _ in range(len(g))]
                i = tokenizer.batch_decode(step_items['id_enc'], skip_special_tokens = True)
                s = tokenizer.batch_decode(step_items['src'], skip_special_tokens = False)
                
                s = [clean_eos_pad_role(ss, eos, pad) for ss in s]
                gens_cq = [clean_eos_pad_role(replace_nlines_tabs(x), eos, pad) for x in step_items['gens_cq']]
                gens_sq = [clean_eos_pad_role(replace_nlines_tabs(x), eos, pad) for x in step_items['gens_sq']]
                cq_ans_list = [clean_eos_pad_role(str(x), eos, pad) for x in step_items['cq_ans_list']]
                sq_ans_list = [clean_eos_pad_role(str(x), eos, pad) for x in step_items['sq_ans_list']]
                
                # "generated" is the last SQ ans. NOTE: str() as ans could be None
                generated.extend([str(x[-1]) if x else '' for x in step_items['sq_ans_list']])
                # "references" is the last CQ ans (we have CoT and collect all the ans with findall)
                references.extend([str(x[-1]) if x else '' for x in step_items['cq_ans_list']])
                idxes.extend(i)
                assert len(t) == len(s) == len(cq) == len(i), (len(t), len(s), len(cq), len(i))
                assert len(t) == len(gens_cq) == len(gens_sq) == len(cq_ans_list) == len(sq_ans_list), \
                            (len(gens_cq), len(gens_sq), len(cq_ans_list), len(sq_ans_list))
                zipped = zip(t, s, cq, gens_cq, gens_sq, cq_ans_list, sq_ans_list, i)
                for tt, ss, ccqq, gcq, gsq, ca, sa, ii in zipped: 
                    tt = replace_nlines_tabs(tt)
                    ccqq = self.c_s_p_eval_bleu_em(replace_nlines_tabs(ccqq))
                    f.write(f'{ii}\t{(ccqq)}\t{(self.c_s_p_eval_bleu_em(tt))}\t{gcq}\t{ca}\t{gsq}\t{sa}\n')

        return generated, references, idxes

    def configure_optimizers(self):
        optims = []
        
        scheduler_info = self.scheduler_info
        params = scheduler_info['params']
        warmup_type                         = params['warmup_type']
        warmup_num_steps, training_steps    = params['warmup_num_steps'], params['total_num_steps']
        self.print_func(f'\t\{scheduler_info["type"].upper()} being used... \
                            for {warmup_num_steps} steps ({self.args.warmup_ratio} of {training_steps})')
        
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"] 
        # see examples/pytorch/summarization/run_summarization_no_trainer.py
        # see also https://github.com/huggingface/transformers/pull/18002
        if getattr(self.args, 'no_weight_decay', False): 
            optimizer_grouped_parameters = self.parameters()
        else: 
            g1_decay = {"params": [p for n, p in self.named_parameters() \
                            if not any(nd in n for nd in no_decay)], "weight_decay": 0.01,}
            g2_nodecay = {"params": [p for n, p in self.named_parameters() \
                if any(nd in n for nd in no_decay)], "weight_decay": 0.0,}
            optimizer_grouped_parameters = [g1_decay, g2_nodecay]

        optimizer = AdamW(optimizer_grouped_parameters, lr = self.args.lr,)

        if scheduler_info['type'] == 'linear_warmup':
            monitor = None
            scheduler = get_linear_schedule_with_warmup(optimizer = optimizer, 
                                num_training_steps = training_steps,
                                num_warmup_steps = warmup_num_steps, last_epoch = -1)
        else: raise NotImplementedError

        optims.append({'optimizer': optimizer, 
                       'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'monitor': monitor}})              

        self.print_func('\t\tOPTIMS FOR INFO:', optims)        
        
        return tuple(optims)

    def give_predgen_training(self, src, src_mask, tgt, lm_logits, 
                              batch_idx, val_step, do_print = False):
        bypass_predgen = getattr(self.args, 'bypass_predgen', False)
        cut = src.size(0) if (val_step and not bypass_predgen) else 2
        src_dec = self.tokenizer.batch_decode(src[:cut])
        if not self.args.c_dqg_hotpotqa_bypass: 
            tgt_dec = self.tokenizer.batch_decode(tgt[:cut])
        preds   = self.tokenizer.batch_decode(lm_logits[:cut].argmax(dim=-1))    
 
        self.model.eval()
        gens_holder = {}
        gen_args = self.args.gen_args
        num_beams = gen_args['num_beams']

        with torch.no_grad():
            gens        = self.model.generate(inputs = src[:cut].to(self.device), 
                                              attention_mask = src_mask[:cut].to(self.device), 
                                              **gen_args) 
        if self.args.chat_model: 
            src_num_tokens = src.size(1)
            gens = gens[:, src_num_tokens:]
        gens_dec    = self.tokenizer.batch_decode(gens, skip_special_tokens = False)
        gens_holder[num_beams] = gens_dec  
    
        self.model.train()

        if do_print:
            self.print_func(f'{"#"*50}')
            self.print_func(f"\t\tPREDICTION (batch_idx {batch_idx}):")
            pick = [0,1] if len(src_dec) >= 2 else range(len(src_dec))
            for i in pick: 
                self.print_func(f'\t\t(SRC)    -->', self.c_s_p(src_dec[i]))
                if not self.args.c_dqg_hotpotqa_bypass: 
                    self.print_func(f'\t\t(TGT)    -->', self.c_s_p(tgt_dec[i]))
                self.print_func(f'\t\t(ARGMAX) -->', self.c_s_p(preds[i]))
                self.print_func(f'\t\t(GEN) [beams: {num_beams}] -->', 
                                self.c_s_p(gens_holder[num_beams][i]))
                self.print_func()
            self.print_func(f'{"#"*50}')

        return tgt_dec, preds, gens_holder[num_beams]
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


class DQGModel_Lightning(pl.LightningModule, DQGModel_Accelerate):
    def __init__(self, args, tokenizer, model, train_dataloader, val_dataloader, test_dataloader):
        super(DQGModel_Lightning, self).__init__(args, tokenizer, model, train_dataloader, val_dataloader, test_dataloader)


def load_base_model(args):
    from transformers import AddedToken
    args.model_id = args.model_mapping[args.model_name]
    cache_loc = f'llm_models/{args.model_name}'
    cache_dir = f'{cache_loc}/{args.model_id}'
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    # 1. load config (for modifying pad_token_id and passing into AutoModel.from_pretrained)
    config = AutoConfig.from_pretrained(args.model_id)

    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, 
                                                padding_side = args.padding_side, 
                                                use_fast    = args.use_fast_tokenizer, 
                                                legacy      = args.use_t5_legacy)
    tokenizer.qpos_tokens = args.qpos_tokens
    tokenizer.operators = args.operators
    add_tokens = []
    added_pad_token = False
    if args.model_name.startswith('flan-t5'): 
        for t in tokenizer.qpos_tokens: add_tokens.append(AddedToken(t, rstrip = False, lstrip = False))
        for t in tokenizer.operators:   add_tokens.append(AddedToken(t, rstrip = False, lstrip = False))
    if args.chat_model and tokenizer.pad_token is None: 
        # for models with no padding token, we add a padding token to the tokenizer
        # see https://huggingface.co/docs/transformers/model_doc/llama3
        args.pad_token = "<|pad|>"
        add_tokens.append(AddedToken(args.pad_token, rstrip = False, lstrip = False, special = True))
        added_pad_token = True
    else: args.pad_token = tokenizer.pad_token
    tokenizer.add_tokens(add_tokens)
    print('👀 TOKENIZER CREATED', args.model_id, 'add_tokens', add_tokens)
    if getattr(args, 'pad_token', None) is not None: 
        tokenizer.pad_token     = args.pad_token
        tokenizer.pad_token_id  = tokenizer.convert_tokens_to_ids(args.pad_token)

    # 1. load model
    load_in_4bit, load_in_8bit = False, False
    if args.load_in_nbits == 4:     load_in_4bit = True
    elif args.load_in_nbits == 8:   load_in_8bit = True
    
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig        
        bnb_args = {'load_in_4bit': load_in_4bit, 'load_in_8bit': load_in_8bit}
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        if args.model_id in ['meta-llama/Meta-Llama-3-70B', 
                             'meta-llama/Meta-Llama-3-70B-Instruct', 
                             'meta-llama/Meta-Llama-3.1-70B-Instruct',
                             'meta-llama/Llama-3.1-70B-Instruct',]: 
            assert load_in_4bit, 'Large LLAMA model should be loaded in 4-bit quantisation'
        elif args.model_id in ['google/flan-ul2', 'google/flan-t5-xxl']:
            assert load_in_8bit, 'Large FLAN model should be loaded in 8-bit quantisation'
        
        if load_in_4bit:
            bnb_args['bnb_4bit_quant_type']         = "nf4"
            bnb_args['bnb_4bit_use_double_quant']   = True
            bnb_args['bnb_4bit_compute_dtype']      = torch.bfloat16
        quantization_config = BitsAndBytesConfig(**bnb_args)
    else: quantization_config = {} if args.model_name == 'commandr_plus' else None

    if args.device =='cuda': attn_implementation = 'flash_attention_2'
    else: attn_implementation = None
    if 'flan-' in args.model_id: attn_implementation = None # flash attention not ready for enc-dec models
    # rerank tasks are effectively batch size 1 in rankllm setup
    if attn_implementation == 'flash_attention_2' and args.bsz > 1:
        # attn_implementation = None
        raise ValueError('🚨\t\tBatch size must be 1 for efficient use of Flash Attention 2.')
    
    args.model_args = {'cache_dir':              cache_dir, 
                  'config':                 config, 
                  'torch_dtype':            torch.float32 if args.fp32 else torch.bfloat16,
                  'attn_implementation':    attn_implementation, 
                  'token':                  args.hf_token if args.hf_token else None,
                  'quantization_config':    quantization_config} 

    if 'flan-' in args.model_id:    model_class = AutoModelForSeq2SeqLM
    else:                           model_class = AutoModelForCausalLM

    if args.use_accelerate and args.load_ckpt_path is not None and '_lora' not in args.load_ckpt_path:
        # lora weights will be loaded after model set by peft
        args.logger.print('\t\tLoading model weights from ckpt here:', args.load_ckpt_path)
        model = model_class.from_pretrained(args.load_ckpt_path, **args.model_args)    
    else: 
        model = model_class.from_pretrained(args.model_id, **args.model_args)    
    print('🔍 MODEL FP CHECK (PRE)', model.dtype)

    # resize model, set model to self
    if len(tokenizer.get_vocab()) > model.config.vocab_size: 
        print('resizing embedding to:', len(tokenizer.get_vocab()))
        model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    if added_pad_token:
        emb_settings = {}
        for key in ['max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse']:
            emb_settings[key] = getattr(model.model.embed_tokens, key, None)

        # ensure that the Embedding returns zeros for added pad taken 
        # set to inner dim = 1 for low mem, esp for loading large models
        emb = torch.nn.Embedding(model.model.embed_tokens.weight.shape[0], 1, 
                         padding_idx = tokenizer.pad_token_id,
                         dtype  = model.model.embed_tokens.weight.dtype,
                         device = model.model.embed_tokens.weight.device)
        
        for key, value in emb_settings.items(): setattr(emb, key, value)
        
        # ensure pretrained weights used 
        emb.weight.data          = model.model.embed_tokens.weight.data
        model.model.embed_tokens = emb
        # model.model.embed_tokens[tokenizer.pad_token_id] = 0
        model.model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def make_DQG_model(args, tokenizer, model, 
                   train_dataloader, val_dataloader, test_dataloader, 
                   bypass_trainer = False): 
    
    if args.use_accelerate: DQG_model_class = DQGModel_Accelerate
    else:                   DQG_model_class = DQGModel_Lightning

    if args.load_ckpt_path is not None and not (args.test_only == False or args.roundtrip_filtering_qa): 
        assert args.load_ckpt_path is not None
        model = load_model_from_ckpt_path(DQG_model_class, args, tokenizer, model, 
                                          train_dataloader, val_dataloader, test_dataloader)
    else:                               
        model = DQG_model_class(args, tokenizer, model, 
                                train_dataloader, val_dataloader, test_dataloader)
    
    if bypass_trainer: return model, None

    trainer = give_trainer(args, eval_bleu_em = {'bleu': model.eval_bleu, 'em': model.eval_em})
    print(f'\t\tTraining for max of {args.max_epochs} epochs.')
    return model, trainer


def load_model_from_ckpt_path(DQG_model_class, args, tokenizer, model, 
                              train_dataloader, val_dataloader, test_dataloader):

    if os.path.basename(args.load_ckpt_path) == 'best_model.pt':
        args.logger.print('\t\t\tIn best_model.pt branch')
        model = DQG_model_class(args, tokenizer, model, train_dataloader, val_dataloader, test_dataloader)
        model.load_state_dict(torch.load(args.load_ckpt_path, weights_only = True)['state_dict'], strict=False)
    
    if args.use_accelerate or '_lora' in args.load_ckpt_path:
        args.logger.print('\t\t\tIn accelerate ckpt branch')
        model = DQG_model_class(args = args, 
                                tokenizer = tokenizer, 
                                model = model, 
                                train_dataloader = train_dataloader, 
                                val_dataloader = val_dataloader, 
                                test_dataloader = test_dataloader,)

    else: 
        if '_lora' not in args.load_ckpt_path:
            args.logger.print('\t\t\tIn lightning ckpt branch')
            args.logger.print('\t\tLoading model weights from ckpt here:', args.load_ckpt_path)
            model = DQG_model_class.load_from_checkpoint(args.load_ckpt_path, 
                                                    args = args, 
                                                    tokenizer = tokenizer, 
                                                    model = model, 
                                                    train_dataloader = train_dataloader, 
                                                    val_dataloader = val_dataloader, 
                                                    test_dataloader = test_dataloader,
                                                    strict = False)
        
        for block in ['encoder', 'decoder']:
            getattr(model.model, block).embed_tokens.weight = model.model.shared.weight
            args.logger.print(f'model.{block}.embed_tokens.weight set to be shared with model.shared.weight')

    args.logger.print('\t\tLoaded model weights from ckpt here:', args.load_ckpt_path)  

    return model


def give_trainer(args, eval_bleu_em = False):
    tb_logger = None
    patience = 5
    if eval_bleu_em:
        if eval_bleu_em['bleu']: mode, monitor = 'max', 'eval_bleu'
        elif eval_bleu_em['em']: mode, monitor = 'max', 'eval_em'
    else: mode, monitor = 'min', 'train_loss_epoch'
    callbacks = [
                EarlyStopping(monitor = monitor, mode = mode, 
                min_delta = 0.0005, patience = patience, check_on_train_epoch_end = False),
                TQDMProgressBar(refresh_rate = 20), 
                LearningRateMonitor(logging_interval='step'),
                ]
    ckpt_callback = None 
    if args.test_only == False:
        tb_logger = TensorBoardLogger(save_dir = args.savepath, 
                    version = str(datetime.date.today()), name = 'lightning_logs')
             
        every_n_epochs, every_n_train_steps = 1, None  

        ckpt_callback = ModelCheckpoint(dirpath = args.savepath, 
                    save_last = args.save_last, save_top_k = args.save_top_k_models, 
                    monitor = monitor, mode = mode, every_n_epochs = every_n_epochs, 
                    every_n_train_steps = every_n_train_steps, save_on_train_epoch_end = True)
        print('🔍CKPT_CALLBACK:', args.save_last, args.save_top_k_models, 
              monitor, mode, every_n_epochs, every_n_train_steps, args.savepath, )
        
        callbacks.append(ckpt_callback)
    
    # NOTE: bf16-mixed with lightning 2.0 and lora, leading to nan losses. 
    args.check_fp = 'bf16' 
    if getattr(args, 'fp32', False): args.check_fp = '32'
    args.logger.print('CHECK FP:', args.check_fp)
    trainer_precision, trainer_plugins = args.check_fp, []

    trainer_args = {'deterministic': True, 'devices': getattr(args, 'num_gpu', 1), 
                    'num_nodes': getattr(args, 'num_node', 1), 
                    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu', 
                    'precision': trainer_precision, 'plugins': trainer_plugins,
                    'check_val_every_n_epoch': 1, 
                    'max_epochs': args.max_epochs, 'callbacks': callbacks,
                    'accumulate_grad_batches': args.num_accumulation_steps, 
                    'logger': tb_logger, 'log_every_n_steps': 50}
    
    trainer_args['strategy'] = 'auto'
    trainer_args['strategy'] =  DDPStrategy()    
    trainer = pl.Trainer(**trainer_args)

    return trainer


class Logger:

    def __init__(self, foldername, main_process):
        '''
        Creates a logging object and begins saving to a log file.
        '''
        self.logger = None
        self.main_process = main_process
        if self.main_process:
            self.logger = logging.getLogger()
            fhandler = logging.FileHandler(filename=os.path.join(foldername, 'errorlogging.log'), mode='w+')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)
            self.logger.setLevel(LOGGING_LEVEL)

    def print(self, *message):
        '''
        Helper function to print and log messages. 
        '''
        if self.main_process:
            print(*message)
            self.logger.log(LOGGING_LEVEL, message)

def replace_nlines_tabs(text):
    return text.replace('\n', '[NEWLINE]').replace('\t', '[TAB]')

def clean_eos_pad_role(text, eos_token, pad_token, role = '', turn_end = ''):
    text = text.replace(eos_token, '').replace(pad_token, '').strip()
    
    if turn_end and text.endswith(turn_end): 
        text = text[:-len(turn_end)].strip()
    
    if role and text.endswith(role): 
        text = text[:-len(role)].strip()
    
    return text

def clean_turn_end(text, turn_end = '<|im_end|>'):
    text = text.strip()
    if text.endswith(turn_end):
        text = text[:-len(turn_end)].strip()
    return text

def identity_text(text): 
    return text