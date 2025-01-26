import json
import random

import torch
from torch.utils.data import Dataset

import numpy as np

### CHANGE START ###
import copy, re
### CHANGE END ###

class HotpotQADataset(Dataset):

    def __init__(self, tokenizer, data_path, max_len=512, dataset_type='hotpot',
                ### CHANGE START ###
                do_single_hop=False,
                ### CHANGE END ###
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset_type = dataset_type
        self.max_passages_num = 25
        
        print("beginning to read data from " + data_path)
        if self.dataset_type.startswith('hotpot'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        ### CHANGE START ###
        elif self.dataset_type == '2wiki':    
            with open(data_path, 'r') as f:
                self.data = [json.loads(l) for l in f]
            # put CQ for 2wiki into the same format as musique (so we can reuse musique's code)
            self.data = make_2wiki_musique_like(self.data)
        ### CHANGE END ###
        
        elif self.dataset_type == 'musique':
            musique_train_data = open(data_path).readlines()
            self.data = [json.loads(item) for item in musique_train_data]
        
        else: raise NotImplementedError

        ### CHANGE START ###
        self.do_single_hop = do_single_hop
        if self.do_single_hop:
            if self.dataset_type == 'hotpot': 
                raise NotImplementedError
            elif self.dataset_type in ['musique', '2wiki']:
                self.data = add_single_hop_musique_2wiki(self.data, self.dataset_type)
                
        ### CHANGE END ###
        print(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question'] if self.dataset_type != 'iirc' else (sample['question_text'] + sample['pinned_contexts'][0]['paragraph_text'])
        if question.endswith("?"):
            question = question[:-1]
        q_codes = self.tokenizer.encode(question, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len).squeeze(0)
        sp_title_set = set()
        c_codes = []
        sf_idx = []

        if self.dataset_type == 'hotpot':
            id = sample['_id']
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])
            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                l = title + "".join(sentences)
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
        
        ### CHANGE START ###
        elif self.dataset_type in ['musique', '2wiki']:
            # musique
            id = sample['id']
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
            # label order
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])
        ### CHANGE END ###
        
        elif self.dataset_type == 'iirc':
            id = sample['question_id']
            for i, para in enumerate(sample['contexts']):
                if i > self.max_passages_num:
                    break
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
                if para['is_supporting']:
                    sf_idx.append(para['idx'])
        
        elif self.dataset_type == 'hotpot_reranker':
            id = sample['_id']
            for i, para in enumerate(sample['paragraphs']):
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
                if para['is_supporting']:
                    sf_idx.append(i)
        
        res = {
            'q_codes': q_codes,
            'c_codes': c_codes,
            'sf_idx': sf_idx,
            'id': id,
        }
        return res

    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    if len(samples) == 0:
        return {}
    batch = {
        'q_codes': [s['q_codes'] for s in samples],
        'c_codes': [s['c_codes'] for s in samples],
        "sf_idx": [s['sf_idx'] for s in samples],
        "id": [s['id'] for s in samples],
    }
    return batch

class HopDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=512, is_training=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.is_training = is_training
        print(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        inputs = sample['input']
        context = sample['context']
        label = sample['label']
        question = inputs[0]
        pre_passages = inputs[1:] if len(inputs) > 1 else []
        if self.is_training and len(pre_passages) > 1:
            # for training
            random.shuffle(pre_passages)
        if inputs[0].endswith("?"):
            question = question[:-1]
            inputs[0] = question
        question_codes = self.tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=self.max_len)
        
        mean_passage_length = (self.max_len - len(question_codes)) // (len(pre_passages) + 1)
        try:
            inputs_codes = self.tokenizer.encode("".join(inputs), add_special_tokens=False,truncation=True, max_length=self.max_len)
            context_codes = self.tokenizer.encode(context, add_special_tokens=False, truncation=True, max_length=self.max_len)
            if len(inputs_codes) + len(context_codes) > self.max_len:
                context_codes = context_codes[:mean_passage_length]
                if len(inputs_codes) + len(context_codes) > self.max_len:
                    pre_passages_codes = [self.tokenizer.encode(item, add_special_tokens=False,truncation=True, max_length=self.max_len) for item in pre_passages]
                    idx = 0
                    inputs_codes = question_codes[:]
                    while sum([len(item) for item in pre_passages_codes]) > self.max_len - len(question_codes):
                        pre_passages_codes[idx] = pre_passages_codes[idx][:mean_passage_length]
                        inputs_codes.extend(pre_passages_codes[idx])
                        idx += 1
            assert len(inputs_codes) + len(context_codes) <= self.max_len
        except Exception as e:
            print(e)
            print(f"question:{question}, len(question_codes):{len(question_codes)}, mean_passage_length:{mean_passage_length}, len(pre_passages):{len(pre_passages)}")
            print(f"len_inputs_code:{len(inputs_codes)}, len_context_codes:{len(context_codes)}")
            print(f"pre_passages_len:{[len(item) for item in pre_passages_codes]}, self.max_len - len(question_codes):{self.max_len - len(question_codes)}")
            raise e

        res = {
            'input_ids': torch.tensor(inputs_codes+ context_codes, dtype=torch.long),
            'label': label,
            'hop': len(pre_passages) + 1
        }
        return res

    def __len__(self):
        return len(self.data)
    
def collate_fn_each_hop(samples):
    if len(samples) == 0:
        return {}
    max_q_sp_len = max([item['input_ids'].shape[-1] for item in samples])
    all_q_doc_input_ids = torch.zeros((len(samples), max_q_sp_len), dtype=torch.long)
    all_q_doc_attention_mask = torch.zeros((len(samples), max_q_sp_len), dtype=torch.long)
    labels = torch.zeros(len(samples), dtype=torch.long)

    for i, sample in enumerate(samples):
        len_input_ids = sample['input_ids'].shape[-1]
        all_q_doc_input_ids[i, :len_input_ids] = sample['input_ids'].view(-1)
        all_q_doc_attention_mask[i, :len_input_ids] = 1
        labels[i] = sample['label']
    batch = {
        'input_ids': all_q_doc_input_ids,
        'attention_mask': all_q_doc_attention_mask,
        "labels": labels,
        "hops": [s['hop'] for s in samples]
    }
    return batch

### CHANGE START ###
#############################################################
def realise_musique_predicates(holder, pattern_forward = re.compile(r'>>')):
    with open(f'data/02_support_data/03b_musique_relation_mapping.json', encoding = 'utf-8') as f:
        templates = json.load(f)

    for line in holder:
        for sq_info in line['question_decomposition']:
            q = sq_info['question']
            if re.search(pattern_forward, q):
                e, r = q.split('>>')
                e = e.strip()
                r = r.strip()
                t = random.choice(templates[r])
                sq_info['question'] = t.replace('#X', e)
    return holder

def make_paragraph_musique_like(context, supporting_facts):
    paragraphs = []
    paragraph_support_idx = [None for __ in supporting_facts]
    support_map = {v[0]: sqid for sqid, v in enumerate(supporting_facts)}
    for cid, con in enumerate(context):
        con_title = con[0]
        con_lines = con[1]
        
        is_supporting = False
        if con_title in support_map:
            is_supporting = True
            # there could be more than one SQ that uses this context for support
            to_set = [sqid for sqid, v in enumerate(supporting_facts) if v[0] == con_title]
            for i in to_set: paragraph_support_idx[i] = cid
        
        paragraphs.append({'idx': cid, 'title': con_title, 
                           'paragraph_text': ' '.join(con_lines),
                           'is_supporting': is_supporting})
    
    assert None not in paragraph_support_idx, \
        (paragraph_support_idx, supporting_facts, [c[0] for c in context])

    return paragraphs, paragraph_support_idx

def make_2wiki_musique_like(data, is_test = False):
    for i, sample in enumerate(data):
        idx = sample['id']
        # NOTE: retrieval training should use realised questions for SQs 
        # (i.e. referring answers filled)
        question    = sample['text']['qs']
        answer      = sample['text']['as']
        assert len(question) == 1
        if not is_test: 
            assert type(answer) == str
        question    = question[0]
        sq_info     = sample['decomp_qs']['text']
        sqs_list    = sq_info['qs']
        sqs_ans     = sq_info['as']
        if is_test:
            assert sqs_ans is None and sqs_list is None

        # convert to paragraphs-like format (to reuse musique)
        context             = sample['original_info']['context']
        supporting_facts    = sample['original_info']['supporting_facts']
        paragraphs, paragraph_support_idx = make_paragraph_musique_like(context, supporting_facts)

        new_sample = {}
        new_sample['id']             = idx 
        new_sample['question']       = question
        new_sample['answer']         = answer
        new_sample['answer_aliases'] = []
        new_sample['answerable']     = True
        new_sample['paragraphs']     = paragraphs
        
        question_decomposition = []
        if not is_test:
            zipped = zip(sqs_list, sqs_ans, paragraph_support_idx)
            for sqid, (question, answer, p_s_i) in enumerate(zipped):
                question_decomposition.append({'id':                    sqid,       # str
                                            'question':                 question,   # str
                                            'answer':                   answer,     # str
                                            'paragraph_support_idx':    p_s_i})     # int

        ### verify paragraph support index ###
        for pid, para in enumerate(new_sample['paragraphs']):
            assert para['idx'] == pid, (para['idx'], pid)
            if pid in paragraph_support_idx:    
                assert new_sample['paragraphs'][pid]['is_supporting'] == True
            else:               
                assert new_sample['paragraphs'][pid]['is_supporting'] == False
        #######################################
        new_sample['question_decomposition'] = question_decomposition
        
        data[i] = new_sample
    
    print('���\t\tConverted 2wiki data to musique format for sqs')
    return data

def add_single_hop_musique_2wiki(data, dataset_type, sq_all_paragraphs = False):
    if dataset_type == 'musique':
        data = realise_musique_predicates(data)
    
    for sample in data.copy():
        new_sample = copy.deepcopy(sample)
        question_decomposition = new_sample.pop('question_decomposition')

        sqid_list, sq_list, ans_list, psi_list = [], [], [], []
        for sq_info in question_decomposition:
            question = sq_info['question']
            sqid_list.append(sq_info['id'])
            sq_list.append(question)
            ans_list.append(sq_info['answer'])
            psi_list.append(sq_info['paragraph_support_idx'])

        if dataset_type == 'musique':
            for __, question in enumerate(sq_list):
                var_list = re.findall(r'#[0-9]+', question)
                for v in var_list:
                    v_num = int(v[1:]) -1  # NOTE: 1-indexed for vars, 0-indexed for pred_sq_as
                    try: ref_a = ans_list[v_num]
                    except: 
                        print('🚨\t\tMISSING REFERRING ANS', sq_info['id'], f'VNUM: {v_num} of [{var_list}]', question)
                        ref_a = None
                    if ref_a is not None: 
                        question = question.replace(v, ref_a)
                sq_list[__] = question

        for sq_id, (sqid, sq, a, p_s_i) in enumerate(zip(sqid_list, sq_list, ans_list, psi_list)):
            sq_new_sample = copy.deepcopy(new_sample)
            sq_new_sample['id'] += f'_sq{sq_id}'
            sq_new_sample['question']       = sq
            sq_new_sample['answer']         = a
            sq_new_sample['answer_aliases'] = []
            sq_new_sample['answerable']     = True
            assert 'question_decomposition' not in sq_new_sample, sq_new_sample.keys()
            sq_new_sample['question_decomposition'] = [{'id':                       sqid,   # str
                                                        'question':                 sq,     # str
                                                        'answer':                   a,      # str
                                                        'paragraph_support_idx':    p_s_i}] # int
            for pid, para in enumerate(sq_new_sample['paragraphs']):
                assert para['idx'] == pid, (para['idx'], pid)
                if pid == p_s_i:    
                    sq_new_sample['paragraphs'][pid]['is_supporting'] = True
                else:
                    if sq_all_paragraphs:
                        # other SQ's supporting are also marked True (allow retrieval by CQ)
                        if para['is_supporting'] == True: 
                            sq_new_sample['paragraphs'][pid]['is_supporting'] = True
                    else: 
                        # other SQ's supporting are also marked True, set to False here               
                        if para['is_supporting'] == True: 
                            sq_new_sample['paragraphs'][pid]['is_supporting'] = False
            
            data.append(sq_new_sample)
    
    print('���\t\tAdded single-hop sqs to musique/2wiki data')
    return data
#############################################################
### CHANGE END ###