import torch, re, string, collections

def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    def filter(x):
        return x[7:] if x.startswith('module.') else x

    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    ### CHANGE START ###
    model.load_state_dict(state_dict, strict = exact)
    ### CHANGE END ###
    return model

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_em_f1(predicted_support_idxs, gold_support_idxs):
    # Taken from hotpot_eval
    cur_sp_pred = set(map(int, predicted_support_idxs))
    gold_sp_pred = set(map(int, gold_support_idxs))
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

def normalize_sp(sps):
    new_sps = []
    for sp in sps:
        sp = list(sp)
        sp[0] = sp[0].lower()
        new_sps.append(sp)
    return new_sps


def update_sp(prediction, gold):
    cur_sp_pred = normalize_sp(set(map(tuple, prediction)))
    gold_sp_pred = normalize_sp(set(map(tuple, gold)))
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
    return em, f1


### CHANGE START ###
import os
def give_qa_pred_filename(args, ds_type, is_dev):
    dp = f"results/qa/{args.checkpoint_name}" 
    if not os.path.exists(dp): os.makedirs(dp)

    pred_filename = f"sorted_pred_{'validation' if is_dev else 'test'}_{args.dataset_type}_retrbase_qalarge_beam{args.beam_size}"
    if getattr(args, 'decomp_origin', None):
        pred_filename += '_decomp_origin_' + '-'.join(args.decomp_origin) 
    pred_filename += f"{f'_CROSSDOMAIN-{args.model_type}-{args.dataset_type}' if args.cross_domain else ''}"
    pred_filename += f"{'_GOLDPASSAGES' if getattr(args, 'use_gold_passages', False) else ''}"
    pred_filename += f"{'_retr-cq-filt' if getattr(args, 'do_retr_compquestion_filter', False) else ''}"
    pred_filename += f"{'_retr-rand-pool' if getattr(args, 'retr_sq_rand_pool_samples', False) else ''}"
    pred_filename += f"{'_retr-qa-combined' if getattr(args, 'do_combined', False) else ''}"
    pred_filename += f"{'_merged' if args.answer_merge else '_no_merged'}"
    if args.dataset_type == 'hotpot':
        if args.filter_hotpot: 
            pred_filename += '_filtered'
    
    pred_filename += f"{'_TRIAL' if args.do_trial else ''}"
    pred_filename += f".{'jsonl' if ds_type in ['musique', '2wiki'] else 'json'}"
    pred_filename = f'{dp}/{args.cqsq_pfx}{pred_filename}'
    print('\t\t CHECK (QA) ... pred_filename:', pred_filename)
    return pred_filename

def give_retr_pred_filename(args, is_dev):
    dp = f"results/retrieval/{args.checkpoint_name}"
    if not os.path.exists(dp): os.makedirs(dp)
    
    pred_filename = f"pred_{'validation' if is_dev else 'test'}_{args.dataset_type}_retrbase"
    if getattr(args, 'decomp_origin', None):
        pred_filename += '_decomp_origin_' + '-'.join(args.decomp_origin)
    pred_filename += f"{f'_CROSSDOMAIN-{args.model_type}-{args.dataset_type}' if args.cross_domain else ''}"
    pred_filename += f"{'_GOLDPASSAGES' if getattr(args, 'use_gold_passages', False) else ''}"
    pred_filename += f"{'_retr-cq-filt' if getattr(args, 'do_retr_compquestion_filter', False) else ''}"
    pred_filename += f"{'_retr-rand-pool' if getattr(args, 'retr_sq_rand_pool_samples', False) else ''}"
    pred_filename += f"{'_retr-qa-combined' if getattr(args, 'do_combined', False) else ''}"
    if args.dataset_type == 'hotpot':
        if args.filter_hotpot: 
            pred_filename += '_filtered'

    pred_filename += f"{'_TRIAL' if args.do_trial else ''}"
    pred_filename += '.json'
    pred_filename = f"{dp}/{args.cqsq_pfx}{pred_filename}"
    print('\t\t CHECK (RETR) ... pred_filename:', pred_filename)
    return pred_filename
### CHANGE END ###