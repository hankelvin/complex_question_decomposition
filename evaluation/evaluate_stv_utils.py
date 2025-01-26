import json, re, random, tqdm, os
import pandas as pd, numpy as np
from collections import defaultdict, Counter
import pyrankvote
from pyrankvote import Candidate, Ballot
from pyrankvote.helpers import CompareMethodIfEqual
assert CompareMethodIfEqual.MostSecondChoiceVotes == 'MostSecondChoiceVotes'

ROUND = 4
FUNCS = {'min':     lambda x: round(min(x),        ROUND), 
         'mean':    lambda x: round(np.mean(x),    ROUND),
         'median':  lambda x: round(np.median(x),  ROUND), 
         'max':     lambda x: round(max(x),        ROUND)}
GOLD_KEY = 'gold_preferences'

class Dataset:
    def __init__(self, args, instances, data_dir_map, sysname_to_code):
        self.args               = args
        self.instances          = instances
        self.data_dir_map       = data_dir_map
        self.sysname_to_code    = sysname_to_code
        self.rankllm_choices    = None
        self.gold_holder        = None
        self.all_results        = None # winner by STV/instant rankoff with 2nd choice
        self.all_results_maj    = None # i.e. winner by taking majority vote
        self.districts          = None # i.e. set of instances of the dataset
        self.all_results_topk   = None
        self.all_results_ndcg   = None
        self.perm_order_to_perm_index = {}
        self.perm_index_to_perm_order = {}
        self.c_passage          = args.task in [4, 5]   # passage reranking (i.e. more than relevant passage)
        self.c_dqg              = args.task in [1,2,3]  # dqg_{ds}, syndqgqpt40

    def reset_for_new_run(self):
        self.rankllm_choices    = None
        self.gold_holder        = None
        self.all_results        = None
        self.all_results_maj    = None
        self.districts          = None
        self.all_results_topk   = None
        self.all_results_ndcg   = None

    def load_model_dqg_predictions(self):
        args = self.args
        # this is where we recover the model outputs (from model_outputs.jsonl) [for 1st step decomposing QG]
        # NOTE: for syndqggpt4o data: TODO: has modified dir_model_outputs structure
        dir_model_outputs  = self.data_dir_map['model_outputs'][args.task]
        for dqg_sys in args.dqg_systems:
            print('\t🟧 Adding DQG predictions for system: ', dqg_sys)
            dp  = dir_model_outputs.format(args.ds_name, args.split, dqg_sys)
            
            if args.model_lineup > 1:
                dp += f'_modelline-{args.model_lineup}'
            if args.prompt_version > 1: 
                dp += f'_prmptv{args.prompt_version}'

            fp_tgt  = dp + '/text_labels.csv'
            dqg_tgt = pd.read_csv(fp_tgt)
            dqg_tgt.columns = [c + '_tgt' if c == 'decomposition' else c for c in dqg_tgt.columns]
            
            fp_gen  = dp + '/text_predictions.csv'
            dqg_gen = pd.read_csv(fp_gen)
            dqg_gen.columns = [c + '_gen' if c == 'decomposition' else c for c in dqg_gen.columns]
            
            assert len(dqg_tgt) == len(dqg_gen)
            df = pd.concat([dqg_tgt, dqg_gen], axis = 1)

            for idx, instance in self.instances.items():
                # collect both tgt and prd (tgt used for breakeval, using the standardised tokenisation)
                row = df.loc[df['question_id'] == idx]
                assert len(row) == 1
                row_dict = row.to_dict('records')[0]

                instance.gen_sqs[dqg_sys] = {'decomposition': row_dict['decomposition_gen'],
                                             'operators':     row_dict['operators_gen']}
                instance.tgt_sqs[dqg_sys] = {'decomposition': row_dict['decomposition_tgt'],
                                             'operators':     row_dict['operators_tgt']}
                
        ##### SAFETY CHECK: ensure all systems have the same tgt_sqs for each instance #####
        for idx, instance in self.instances.items():
            # set to dict
            instance.gen_sqs = dict(instance.gen_sqs)
            instance.tgt_sqs = dict(instance.tgt_sqs)
            for i, dqg_sys in enumerate(args.dqg_systems):
                if i == 0: continue
                prev_dqg_sys = args.dqg_systems[i-1]
                assert instance.tgt_sqs[dqg_sys]['decomposition'] == instance.tgt_sqs[prev_dqg_sys]['decomposition']
                # assert instance.tgt_sqs[dqg_sys]['operators']     == instance.tgt_sqs[prev_dqg_sys]['operators']
        #####################################################################################
            
        print('\t🟧 Completed adding DQG predictions for these systems: ', args.dqg_systems)

    def load_rankllm_predictions(self, dp_line):
        # recover the LLM ranking outputs
        args                = self.args
        dir_rerank_outputs  = self.data_dir_map['rerank_outputs'][args.task]

        assert self.perm_order_to_perm_index == {} and self.perm_index_to_perm_order == {}
        for rankllm_sys in args.rankllm_systems:
            print('\t🟦 Adding RANKLLM predictions for system: ', rankllm_sys)
            p_o_2_p_i  = defaultdict(dict)
            p_i_2_p_o  = defaultdict(dict)

            fp = dir_rerank_outputs.format(args.ds_name, args.split, rankllm_sys)
            
            if args.task_name in ['rerank'] or (args.c_dqg and args.task_name in ['rerank_dqg']):   
                fp = f'{fp}/{dp_line}'
            else:           
                fp = f'{fp}/{rankllm_sys}/{dp_line}'
            
            with open(fp, 'r') as f:
                for l in f: 
                    line = json.loads(l)
                    assert len(line) == 1
                    for idx, line_info in line.items(): break
                    
                    if args.single_perm and args.task_name not in ['rerank_dqg']:
                        # these are for rerank passage whose outputs are in results/rerank
                        # the format of the outuputs were from an earlier version and have a diff structure
                        assert args.task_name in ['rerank']
                        
                        if   args.ds_name in ['hotpotqa_distractor', 'breakhigh']: num_passages = 10
                        elif args.ds_name == 'musique': num_passages = 20
                        else: raise NotImplementedError  
                        
                        if '_perm' in idx:
                            # e.g. 1037798_perm4
                            o_idx, perm_id = idx.split('_perm')
                            perm_id = int(perm_id)
                            perm_order = line_info['perm_order']
                        else: 
                            o_idx       = idx
                            perm_id     = 0
                            perm_order  = tuple(range(num_passages))
                            assert line_info['rerank_products']['perm_order'] == perm_order, \
                                (perm_order, line_info['rerank_products']['perm_order'])                      

                        # convert to 0-index
                        line_info['rerank_products']['gold_supports'] = \
                            [int(i)-1 for i in line_info['rerank_products']['gold_supports']]
                        pr_key          = 'response'
                        prompt_response = line_info['rerank_products'][pr_key]
                        response        = None
                        gold_supports   = line_info['rerank_products']['gold_supports']
                        
                        docid_map       = {i:v for i,v in enumerate(perm_order)}
                        inv_docid_map   = {v:i for i,v in docid_map.items()}
                        assert len(docid_map) == len(inv_docid_map) == len(line_info['docid_map']), \
                            (len(line_info['docid_map']), len(docid_map), len(inv_docid_map))
                        num_hops        = len(gold_supports)
                        gold_preferences= line_info['rerank_products']['gold_supports']
                    
                    else: 
                        # note that this only applies for calibration sampling where num_cands is 4
                        # i.e. 'prompt_response' is a list of size 1 
                        # (otherwise, for rerank tasks, this could be a list of multiple entries of window-size)
                        if args.task_name in ['rerank_dqg']:
                            assert '_perm' not in idx
                            o_idx       = idx
                            perm_id     = 0
                        else: 
                            o_idx, perm_id  = re.search(r'(.+)_perm(\d+)', idx).groups()
                            perm_id         = int(perm_id)
                        
                        response        = line_info['rerank_products']['response']
                        docid_map       = {i:v for i,v in enumerate(line_info['docid_map'])}
                        inv_docid_map   = {v:i for i,v in docid_map.items()}
                        assert len(docid_map) == len(inv_docid_map) == len(line_info['docid_map']), \
                            (len(line_info['docid_map']), len(docid_map), len(inv_docid_map))
                        if len(line_info['rerank_products']['prompt_response']) == 1:
                            pr_key          = 'prompt_response'
                            prompt_response = line_info['rerank_products'][pr_key]
                            assert type(prompt_response) == list and len(prompt_response) == 1
                            prompt_response = prompt_response[0]
                        else: 
                            # +1 to ensure 1-indexed (same treatment as above, so same treatment can be applied below too)
                            prompt_response = ' > '.join([f"[{str(inv_docid_map[str(docid)]+1).zfill(2)}]" for docid in response])
                        
                        response        = line_info['rerank_products']['response']
                        gold_supports   = line_info['rerank_products']['gold_supports']
                                            
                        num_hops        = line_info['num_hops']
                        perm_order      = tuple(line_info['rerank_products']['perm_order'])
                        gold_preferences= [perm_order[i] for i in gold_supports]

                    entry = {perm_order: {'original_response': prompt_response,
                                          'response':        response,
                                          'gold_supports':   gold_supports,
                                          'gold_preferences': gold_preferences,
                                          'docid_map':       docid_map,
                                          'inv_docid_map':   inv_docid_map,
                                          'num_hops':        num_hops,
                                          'perm_order':      perm_order,
                                          'perm_id':         perm_id}}
                    
                    assert self.instances[o_idx]
                    self.instances[o_idx].rankllm_outputs[rankllm_sys].update(entry)
                    p_o_2_p_i[o_idx].update({perm_order: perm_id})
                    p_i_2_p_o[o_idx].update({perm_id: perm_order})
            
            self.perm_order_to_perm_index[rankllm_sys] = dict(p_o_2_p_i)
            self.perm_index_to_perm_order[rankllm_sys] = dict(p_i_2_p_o)

        ##### SAFETY CHECK: ensure all systems have the same set of perm_orders #####
        for idx, instance in self.instances.items():
            # set to dict
            instance.rankllm_outputs = dict(instance.rankllm_outputs)
            for i, sys in enumerate(args.rankllm_systems):
                if i == 0: continue
                prev_sys = args.rankllm_systems[i-1]
                c1 = set(instance.rankllm_outputs[sys].keys())
                c2 = set(instance.rankllm_outputs[prev_sys].keys())
                assert c1 == c2, (c1, c2)
        #############################################################################

        print('\t🟦 Completed adding RANKLLM predictions for these systems: ', args.rankllm_systems)


    def create_ref_perm_order_to_use(self, num_runs):
        ref_perm_order_holder   = defaultdict(list)
        ref_sys                 = self.args.rankllm_systems[0]
        for i in range(num_runs):
            for idx, instance in self.instances.items():
                ref_line = instance.rankllm_outputs[ref_sys]
                pick = None
                while pick is None or pick in ref_perm_order_holder[idx]:
                    pick = random.choice(list(ref_line.keys()))  
                ref_perm_order_holder[idx].append(pick)
        
        assert len(ref_perm_order_holder) == len(self.instances)
        assert len(set(len(v) == num_runs for v in ref_perm_order_holder.values())) == 1
        
        return ref_perm_order_holder

    def collate_votes(self, rank_to_use, ref_perm_order_to_use = None):        
        args            = self.args
        rankllm_choices = defaultdict(dict)
        gold_holder     = {}
        ref_sys         = args.rankllm_systems[0]

        if ref_perm_order_to_use is not None: 
            assert len(ref_perm_order_to_use) == len(self.instances)
        else: 
            ref_perm_order_to_use = {idx: None for idx in self.instances}

        for idx, instance in self.instances.items():
            r_p_o_t_u = ref_perm_order_to_use[idx]
            # we will pick the same perm_order from every system 
            ref_line        = instance.rankllm_outputs[ref_sys]
            if r_p_o_t_u is None:  # pick a random perm_order from ref systems 
                ref_perm_order = random.choice(list(ref_line.keys()))
            else: 
                ref_perm_order = r_p_o_t_u

            # ensure that all sys have this same perm_order
            check           = [ref_perm_order in instance.rankllm_outputs[sys] for sys in args.rankllm_systems]            
            assert all(check), (check, idx, ref_perm_order)
            
            gold_holder[idx] = {'gold':       ref_line[ref_perm_order][GOLD_KEY],
                                'num_hops':   ref_line[ref_perm_order]['num_hops'], }

            # start collecting 
            for rankllm_sys in args.rankllm_systems:
                
                rankllm_line        = instance.rankllm_outputs[rankllm_sys][ref_perm_order]
                ##### SAFETY CHECK: ensure that GOLDs match #####
                assert rankllm_line[GOLD_KEY] == gold_holder[idx]['gold'],\
                      (rankllm_line[GOLD_KEY],   gold_holder[idx]['gold'], 
                       rankllm_sys, idx, ref_perm_order)
                #########################################################
                
                response_key    = f'{rank_to_use}_response'
                docid_map       = rankllm_line['docid_map']
                pred_seq        = rankllm_line[response_key]
                assert type(pred_seq) == str, f'unexpected pred_seq {pred_seq}'
                # NOTE: make sure 0-indexed with -1
                pred_seq        = [int(c)-1 for c in re.findall(r'(?:\[)(\d+)(?:\]?)', pred_seq)]

                # convert to docid/sysname 
                # NOTE: models might give degenerate prediction (e.g. number not within nranks)
                # unlikely to have many of these, but to be save set to set to INVALID + rankllm_sys + pred rank 
                # (avoid accumlating into 1 vote if there are multiple invalids)
                choices = [docid_map[pc] if pc in docid_map else f'[INVALID{rankllm_sys}{pc}]'for pc in pred_seq]
                
                if self.args.task_name == 'rerank_dqg' and not args.single_perm: 
                    # remove systems not in args.dqg_systems 
                    choices = [c for c in choices if c in self.args.dqg_systems or c.startswith('[INVALID')]      
                
                rankllm_choices[rankllm_sys][idx] = choices

        self.rankllm_choices    = dict(rankllm_choices)
        self.gold_holder        = dict(gold_holder)
        return self.rankllm_choices, self.gold_holder
    
    def compute_stv(self):
        args = self.args
        all_results = {}

        # get the set of districts 
        districts = set()
        for cdict in self.rankllm_choices.values(): 
            districts.update(cdict.keys())

        # put each district to vote
        for i, district in enumerate(tqdm.tqdm(districts)):
            candidates  = {} # These are what's being voted on 
            ballots     = [] # These are the ballots that contain the votes for this "district"
            # this is running through all rankllm systems
            for rankllm_sys, choices in self.rankllm_choices.items():
                rankllm_sys_vote    = choices[district]
                dedup_ranks         = remove_duplicates(rankllm_sys_vote)
                
                ##### add to candidates if it's not there yet #####
                for cand in dedup_ranks: 
                    if cand not in candidates:
                        candidates[cand] = Candidate(f'{cand}')
                ###################################################
                
                voter_ballot = Ballot(ranked_candidates = [candidates[r] for r in dedup_ranks])
                ballots.append(voter_ballot)

            if self.c_dqg: 
                if args.dqg_roundtrip:  num_seats = args.dqg_roundtrip
                else:                   num_seats = 1
            elif self.c_passage: 
                if   args.ds_name in ['musique']:
                    num_seats = int(district[0]) # i.e. 4hop_....
                elif args.ds_name in ['hotpotqa_distractor', 'breakhigh']: 
                    num_seats = 2 
                else: raise NotImplementedError
            else: raise NotImplementedError
            
            if num_seats is None: num_seats = 1 # self.args.nranks
            # if num_seats is None: pass # leave as None, i.e. STV across all preferences
            
            try:
                if num_seats > 1:
                    # STV for multiple seats
                    results = pyrankvote.single_transferable_vote(candidates = list(candidates.values()), 
                                                ballots = ballots, number_of_seats = num_seats)
                    winners = [r.name for r in results.get_winners()]

                else: 
                    # instant runoff for single seat, but pick the one with 2nd most votes in case of tie
                    results = pyrankvote.instant_runoff_voting(candidates = list(candidates.values()), 
                                                ballots = ballots, 
                                                compare_method_if_equal = "MostSecondChoiceVotes")
                    winners = [r.name for r in results.get_winners()]

                
                results_all_cands = pyrankvote.single_transferable_vote(candidates = list(candidates.values()), 
                                                ballots = ballots, number_of_seats = len(candidates)-1)
                winners_all_cands = [r.name for r in results_all_cands.get_winners()]

            except: 
                all_voters_set      = set([c2.name for c in ballots for c2 in c.ranked_candidates])
                all_candidates_set  = set(all_voters_set)
                assert len(all_voters_set) == len(all_candidates_set) == num_seats
                winners = set(all_candidates_set)
                winners_all_cands = 'ALL VOTED THE SAME'
                print('ALL VOTED THE SAME')

            if self.c_dqg: 
                line = winners 
            elif self.c_passage: 
                line                            = self.gold_holder[district].copy()
                line['stv_winners']             = winners
                line['stv_winners_all_cands']   = winners_all_cands 
            
            all_results[district] = line

        self.all_results    = all_results
        self.districts      = districts
        return all_results, districts
    
    def compile_stv_dataset_rerank_dqg(self, rank_to_use, run_num):
        '''
        NOTE: system-level (i.e. without STV) can be found in the respective system folders        
        '''
        args = self.args
        # A. collecting results of majority vote counting 
        all_results_maj = {}
        for district in self.districts:
            alt_ctr = Counter()
            for rankllm_sys, choices in self.rankllm_choices.items():
                rankllm_sys_vote    = choices[district]
                dedup_ranks         = remove_duplicates(rankllm_sys_vote)
                alt_ctr.update(dedup_ranks[:1])
            ctr_alt                     = {v:k for k,v in alt_ctr.items()}
            all_results_maj[district]   = [ctr_alt[max(alt_ctr.values())]]
        self.all_results_maj = all_results_maj

        # B. check for mismatches between STV and majority vote counting (i.e. validate STV/IRO+2nd relevance)
        alt_num_mismatch = 0
        for i, district in enumerate(self.districts):
            stv_result = self.all_results[district]
            maj_result = all_results_maj[district]
            
            if stv_result != maj_result: 
                alt_num_mismatch += 1
        # NOTE: not informative for dqg_roundtrip (1 winner, vs multiple winners)
        print('NUMBER OF MISMATCHED OUTCOMES BETWEEN STV AND MAJORITY VOTING', alt_num_mismatch/(i+1))

        # C. recover the DQG system prediction and to write out to file
        final_results = defaultdict(list)
        gold_results  = []
        for district, winners in self.all_results.items():
            district_info = self.instances[district]

            num_winners  = len(winners)
            if args.dqg_roundtrip: 
                assert num_winners == args.dqg_roundtrip, (num_winners, args.dqg_roundtrip)
            else: assert num_winners == 1, (num_winners, 1)
            
            for win_num, winner in enumerate(winners):  
                winning_decomp = self.instances[district].gen_sqs[winner]
                target_decomp  = self.instances[district].tgt_sqs[winner]
                
                # stv-selected prediction
                entry = {'question_id':   district, 
                        'decomposition': winning_decomp['decomposition'], 
                        'operators_gen': winning_decomp['operators'], 
                        'sysname': winner}
                final_results[win_num].append(entry)
            
            # gold decomposition
            entry = {'question_id':   district, 
                     'question_text': district_info.cq, 
                     'decomposition': target_decomp['decomposition'], 
                     'operators_tgt': target_decomp['operators']}
            gold_results.append(entry)
        
        df_holder       = {}
        gold            = pd.DataFrame(gold_results)
        for win_num in range(num_winners):
            df_ranked_preds = pd.DataFrame(final_results[win_num])
            # save to file
            dp          = 'results/decomp_qg'
            calib_str   = f'{rank_to_use}_' 
            dqg_str     = '_dqg-'     + '-'.join([sys[:2] for sys in args.dqg_systems])
            rankllm_str = '_rankllm-' + '-'.join([sys[:2] for sys in args.rankllm_systems])
            fp_str = f'{args.ds_name}_{args.split}'
            savepath = f'{dp}/decomp_qg_{fp_str}_STV_listwise{dqg_str}{rankllm_str}'
            if args.model_lineup > 1:   savepath += f'_modelline-{args.model_lineup}'
            if args.prompt_version > 1: savepath += f'_prmptv{args.prompt_version}'
            if args.output_variants: savepath += f'_variants-' + '-'.join(args.output_variants)
            winner_str = ''
            if args.dqg_roundtrip > 1: winner_str = f'_winner{win_num}'
            
            savepath += f'/{calib_str}run{run_num}{winner_str}/'
            if not os.path.exists(savepath): os.makedirs(savepath)
            
            gold.to_csv(savepath+'/text_labels.csv', index = False)
            df_ranked_preds.to_csv(savepath+'/text_predictions.csv', index = False)
            print('🔮 LLM VOTING RANKED DQG saved to', savepath)

            # run breakeval suite on the STV results 
            print('🔍RUNNING BREAKEVAL SUITE ON STV RESULTS...')
            from evaluate_break_suite import do_one_file
            do_one_file(savepath)
            print('🔮 BREAKEVAL FOR STV RESULTS saved to', savepath)
            df_holder[win_num] = df_ranked_preds

        return df_holder
    
    def pick_top_ranked(self):
        args = self.args
        all_results_topk = {}
        ## A. AT SYSTEM LEVEL 
        for rankllm_sys, choices in self.rankllm_choices.items():
            ems, f1s = self.do_one_sys_ems_f1s(choices = choices, level = 'system')

            score_line = {}#{'f1s': f1s, 'ems': ems}
            for metric, m in [('f1', f1s), ('em', ems)]:
                for func_name, func in FUNCS.items():
                    score_line[f'{metric}_{func_name}'] = func(m)
            all_results_topk[rankllm_sys] = score_line

        # B. AT STV-PICKED
        ems, f1s = self.do_one_sys_ems_f1s(choices = self.all_results, level = 'stv-picked')
        score_line = {}#{'f1s': f1s, 'ems': ems}
        for metric, m in [('f1', f1s), ('em', ems)]:
            for func_name, func in FUNCS.items():
                score_line[f'{metric}_{func_name}'] = func(m)
        all_results_topk['stv_order'] = score_line
        
        self.all_results_topk = all_results_topk
        return all_results_topk

    def do_one_sys_ems_f1s(self, choices, level = 'system'):
        assert level in ['system', 'stv-picked']
        ems, f1s = [], []
        for district in self.gold_holder.keys():
            line        = self.gold_holder[district].copy()
            num_seats   = self.gold_holder[district]['num_hops']

            # self.all_results[district] keys are 
            # 'gold_supports', 'num_hops', 'stv_winners'
            if   level == 'system':
                __ = choices[district]
            elif level == 'stv-picked': 
                __ = choices[district]['stv_winners'] 
            # remove duplicates
            voter_ranks = []
            [voter_ranks.append(x) for x in __ if x not in voter_ranks]

            # TODO: check this for cases/datasets
            if num_seats is None: num_seats = 1 # self.args.nranks
            # if num_seats is None: pass # leave as None, i.e. STV across all preferences

            try: 
                f1, em = calculate_em_f1(voter_ranks[:num_seats], line['gold'][:num_seats])
                # print('\t\tHERE 2.2', voter_ranks, line['gold'])
            except: 
                print('\t\t FAIL!', voter_ranks)

            f1s.append(f1)
            ems.append(em)
        return ems, f1s

class SynDQGInstance:
    def __init__(self, idx, orig_idx, query, paragraphs, scores, doc_ids, nhops = None):
        self.idx           = idx
        self.orig_idx      = orig_idx 
        self.query         = query
        self.paragraphs    = paragraphs
        self.scores        = scores
        self.doc_ids       = doc_ids
        self.nhops         = nhops 

        self.rankllm_outputs     = defaultdict(dict)

class DQGInstance: 
    def __init__(self, idx, cq, cq_ans, gold_sqs, gold_sqs_ans):
        self.idx            = idx
        self.cq             = cq
        self.cq_ans         = cq_ans
        self.gold_sqs       = gold_sqs
        self.gold_sqs_ans   = gold_sqs_ans
        # collect the predicted decomposition from each DQG system
        self.gen_sqs       = defaultdict(dict)
        self.tgt_sqs       = defaultdict(dict) # used for breakeval

        self.rankllm_outputs     = defaultdict(dict)
    
class PassageRerankInstance:
    pass

def remove_duplicates(rankllm_sys_vote):
    # remove duplicates
    dedup_ranks = []
    for x in rankllm_sys_vote: 
        if x not in dedup_ranks: 
            dedup_ranks.append(x) 
    return dedup_ranks

def give_original_data(args, data_dir_map):
    # extract the original data (from 01_unified_format)
    dir_original_data   = data_dir_map['original_data'][args.task]
    nr                  = args.nranks
    
    if args.task_name in ['rerank_syndqggpt4o']:
        assert args.split == 'test'
        fp = f'UNIFIED_RANK_syndqggpt4o-{args.ds_name}-{nr}_{args.split}.jsonl'
    else: 
        fp = f'UNIFIED_{args.ds_name}_{args.split}.jsonl'

    with open(f'{dir_original_data}/{fp}') as f: 
        original_data = {}
        # we use the running line number instead of "query_id" as the index number
        ctr_idx = 0
        for l in f:
            line = json.loads(l)
            id_key = 'query_id' if args.ds_name.startswith('syndqg') else 'id'
            idx = line[id_key]
                                
            if args.task_name in ['rerank_dqg']:
                cq, cq_ans = line['text']['qs'], line['text']['as']
                if args.ds_name not in ['breakhigh']:
                    assert len(cq) == len(cq_ans) == 1, 'cq and cq_ans must have same length'
                assert type(cq) == list and len(cq) == 1
                cq = cq[0]
                if args.ds_name not in ['breakhigh']: cq_ans = cq_ans[0] 
                instance = DQGInstance(idx             = idx,
                                       cq              = cq,
                                       cq_ans          = cq_ans,
                                       gold_sqs        = line['decomp_qs']['text']['qs_var'],
                                       gold_sqs_ans    = line['decomp_qs']['text']['as'])
            
            elif args.task_name in ['rerank_syndqggpt4o']:
                idx = str(ctr_idx) 
                ctr_idx += 1

                instance = SynDQGInstance(idx           = idx,
                                          orig_idx      = line[id_key],
                                          query         = line.pop('query'),
                                          paragraphs    = line.pop('paragraphs'),
                                          scores        = [float(s) for s in line.pop('scores')],
                                          doc_ids       = line.pop('doc_ids'),)

            elif args.task_name in ['rerank']:
                query = line['text']['qs'][0]

                if args.ds_name in ['hotpotqa_distractor', 'breakhigh']:
                    paragraphs  = [p[0] + ':\t ' + ''.join(p[1]) for p in line['original_info']['context']]
                    
                    titles      = [p[0] for p in line['original_info']['context']]
                    supps       = [p[0] for p in line['original_info']['supporting_facts']]
                    scores      = [0 for i in range(len(paragraphs))]
                    for sup in supps: scores[titles.index(sup)] += 1 
                    assert sum(scores) == len(supps)
                    nhops       = len(set(supps))
                    
                    doc_ids     = [i for i, x in enumerate(paragraphs)]
                
                elif args.ds_name in ['musique']:
                    paragraphs = [p['title'] + ':\t ' + p['paragraph_text'] for p in line['original_info']['paragraphs']]
                    scores = [1.0 if p['is_supporting'] else 0.0 for p in line['original_info']['paragraphs']]
                    doc_ids = [p['idx'] for p in line['original_info']['paragraphs']]
                    nhops       = int(sum(scores))

                instance = SynDQGInstance(
                    idx           = idx,
                    query         = query,
                    paragraphs    = paragraphs,
                    scores        = scores,
                    doc_ids       = doc_ids,
                    nhops         = nhops)

            original_data[idx] = instance
    
    return original_data


def calculate_em_f1(predicted_support_idxs, gold_support_idxs):
    '''
    source: https://github.com/canghongjian/beam_retriever/blob/902980bfb5bd47a963569ab9c3621fd5ae81204a/train_beam_retriever.py#L259
    '''
    # Taken from hotpot_eval
    ### CHANGE START ###
    cur_sp_pred     = set(map(str, predicted_support_idxs))
    gold_sp_pred    = set(map(str, gold_support_idxs))
    ### CHANGE END ###
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not cur_sp_pred and not gold_sp_pred:
        f1, em = 1.0, 1.0
        f1, em = 1.0, 1.0
    return f1, em

def convert_to_ranx_rundict(run_dict, sequence, seq_idx, scores = None):
    '''
    Given a ranked sequence, create a dict that is an entry into run_dict
    keys are q{n} and values are dict of doc:score

    run_dict: dict, where each key is a query and the value is a dict of doc:score
    sequence: list of integers, 1st element is the "doc" that is highest ranked. 

    # NOTE: from calibration/calibration_utils.py
    '''
    if scores is None:
        scores = list(range(1, len(sequence)+1))[::-1] # ensure scores are descending, +1 to ensure all scores >= 1
    assert len(sequence) == len(scores), (len(sequence), len(scores))
    ranx_id = f"q{len(run_dict)+1}_{seq_idx}"
    run_dict[ranx_id] = {f"d_{d}": int(sc) for d, sc in zip(sequence, scores)}
    return run_dict, ranx_id