from typing import Dict, Tuple
import numbers
from itertools import zip_longest

import argparse
import os
import random
import re
import numpy as np
import pandas as pd
import json

### CHANGE START ###
import os, sys
filedir = os.path.dirname(os.path.abspath(__file__))
cwd     = os.getcwd()
os.chdir(filedir)
sys.path.append('../evaluation')
### CHANGE END ###
from decomposition import Decomposition
from graph_matcher import GraphMatchScorer, get_ged_plus_scores
from sari_hook import get_sari
from sequence_matcher import SequenceMatchScorer
from normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer
import normal_form.normalization_rules as norm_rules
### CHANGE START ###
os.chdir(cwd)
# pd.set_option('display.max_colwidth', -1)
### CHANGE END ###

def evaluate(ids, questions, decompositions, golds, metadata,
             output_path_base,
             metrics=None, 
             ### CHANGE START ###
             str_strip_qmark_whitespace=False,
             ### CHANGE END ###
             ):
    decompositions_str = [d.to_string() for d in decompositions]
    golds_str = [g.to_string() for g in golds]

    ### CHANGE START ###
    if str_strip_qmark_whitespace:
        decompositions_str = [re.sub('\?+$', '', d.strip()).strip() for d in decompositions_str]
        golds_str = [re.sub('\?+$', '', g.strip()).strip() for g in golds_str]

    bleu_score = get_bleu_score(decompositions_str, golds_str) \
        if (metrics is None) or 'bleu' in metrics else None
    
    gleu_score = get_gleu_score(decompositions_str, golds_str) \
        if (metrics is None) or 'gleu' in metrics else None
    
    chrf_score = get_chrf_score(decompositions_str, golds_str) \
        if (metrics is None) or 'chrf' in metrics else None
    ### CHANGE END ###

    # calculating exact match scores
    exact_match = get_exact_match(decompositions_str, golds_str) \
        if (metrics is None) or 'exact_match' in metrics else None

    # evaluate using SARI
    ### CHANGE START ###
    sari = get_sari_score(decompositions_str, golds_str, questions) \
        if (metrics is None) or 'sari' in metrics else None
    ### CHANGE END ###

    # evaluate using sequence matcher
    match_ratio = get_match_ratio(decompositions_str, golds_str) \
        if (metrics is None) or 'match' in metrics else None
    structural_match_ratio = get_structural_match_ratio(decompositions_str, golds_str) \
        if (metrics is None) or 'structural_match' in metrics else None

    # evaluate using graph distances
    graph_scorer = GraphMatchScorer()
    decomposition_graphs = [d.to_graph() for d in decompositions]
    gold_graphs = [g.to_graph() for g in golds]

    ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs) 
    
    ### CHANGE START ###
    # import multiprocessing, math
    # num_processes = math.ceil(multiprocessing.cpu_count()*0.8)
    # structural_ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs,
    #                                                                     structure_only=True)
    # ged_plus_scores = get_ged_plus_scores(decomposition_graphs, gold_graphs,
    #                                       exclude_thr=5, num_processes=num_processes)
    ### CHANGE END ###

    # calculate normalized match scores
    normalize_scorer = NormalizedGraphMatchScorer()

    def try_invoke(func, graph, default=None):
        try:
            return func(graph)
        except Exception as ex:
            return default

    decomposition_norm_graphs = [try_invoke(normalize_scorer.normalize_graph, g, default=g) for g in
                                 decomposition_graphs]
    decomposition_norm_str = [try_invoke(lambda x: Decomposition.from_graph(x).to_string(), g) for g in
                              decomposition_norm_graphs]
    gold_norm_graphs = [try_invoke(normalize_scorer.normalize_graph, g, default=g) for g in gold_graphs]
    gold_norm_str = [try_invoke(lambda x: Decomposition.from_graph(x).to_string(), g) for g in gold_norm_graphs]

    ### CHANGE START ###
    normalised_bleu_score = get_bleu_score(decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_bleu' in metrics else None
    normalised_gleu_score = get_gleu_score(decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_gleu' in metrics else None
    normalised_chrf_score = get_chrf_score(decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_chrf' in metrics else None
    ### CHANGE END ###
    normalized_exact_match = skip_none(get_exact_match, decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_exact_match' in metrics else None
    normalized_sari = skip_none(get_sari_score, decomposition_norm_str, gold_norm_str, questions) \
        if (metrics is None) or 'normalized_sari' in metrics else None
    normalized_match_ratio = skip_none(get_match_ratio, decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_match' in metrics else None
    normalized_structural_match_ratio = skip_none(get_structural_match_ratio, decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_structural_match' in metrics else None

    evaluation_dict = {
        "id": ids,
        "question": questions,
        "gold": golds_str,
        "prediction": decompositions_str,
        "exact_match": exact_match,
        "match": match_ratio,
        "structural_match": structural_match_ratio,
        "sari": sari,
        "ged": ged_scores,
        ### CHANGE START ###
        # "structural_ged": structural_ged_scores,
        # "ged_plus": ged_plus_scores,
        'bleu': bleu_score,
        'gleu': gleu_score,
        'chrf': chrf_score, 
        'normalized_bleu': normalised_bleu_score,
        'normalized_gleu': normalised_gleu_score,
        'normalized_chrf': normalised_chrf_score,
        ### CHANGE END ###
        "normalized_exact_match": normalized_exact_match,
        "normalized_match": normalized_match_ratio,
        "normalized_structural_match": normalized_structural_match_ratio,
        "normalized_sari": normalized_sari,
    }
    evaluation_dict = {k: v for k, v in evaluation_dict.items() if v is not None}
    num_examples = len(questions)
    print_first_example_scores(evaluation_dict, min(5, num_examples))
    mean_scores = print_score_stats(evaluation_dict)

    if output_path_base:
        ### CHANGE START ###
        write_evaluation_output(str_strip_qmark_whitespace, output_path_base, num_examples, **evaluation_dict)
        ### CHANGE END ###
        ### Addition write the mean scores json
        write_evaluation_results(mean_scores, output_path_base)

    ### CHANGE START ####
    fine_grained_scores = {agg_field: None for agg_field in ["dataset", "num_steps"]}
    ### CHANGE END ###
    if metadata is not None:
        #metadata = metadata[metadata["question_text"].isin(evaluation_dict["question"])]
        metadata = metadata[metadata['question_id'].isin(evaluation_dict['id'])]
        metadata["dataset"] = metadata["question_id"].apply(lambda x: x.split("_")[0])
        metadata["num_steps"] = metadata["decomposition"].apply(lambda x: len(x.split(";")))
        score_keys = [key for key in evaluation_dict if key not in ["id", "question", "gold", "prediction"]]
        for key in score_keys:
            metadata[key] = evaluation_dict[key]

        for agg_field in ["dataset", "num_steps"]:
            df = metadata[[agg_field] + score_keys].groupby(agg_field).agg("mean")
            print(df.round(decimals=3))

            ### CHANGE START ###
            fine_grained_scores[agg_field] = df
            ### CHANGE END ###

    ### CHANGE START ###
    return mean_scores, fine_grained_scores
    ### CHANGE END ###


def skip_none(func, *args, **kwargs):
    zipped = list(zip_longest(*args))
    none_ids = [i for i, x in enumerate(zipped) if None in x]
    args_ = tuple([x for i,x in enumerate(a) if i not in none_ids] for a in args)
    res = func(*args_, **kwargs)

    combined = []
    none_i = 0
    res_i = 0
    for i in range(len(zipped)):
        if none_i < len(none_ids) and (i == none_ids[none_i]):
            combined.append(None)
            none_i += 1
        else:
            combined.append(res[res_i])
            res_i += 1
    return combined

### CHANGE START ###
import sacrebleu, datasets
def get_bleu_score(decompositions_str: [str], golds_str: [str]):
    bleu_score = sacrebleu.corpus_bleu(decompositions_str, [golds_str]).score
    return [bleu_score for _ in range(len(decompositions_str))]


def get_gleu_score(decompositions_str: [str], golds_str: [str]):
    from nltk import word_tokenize
    google_bleu = datasets.load_metric('google_bleu', experiment_id=random.randint(1,1000),)
    gleu_scores = [google_bleu.compute(predictions=[word_tokenize(g)], 
                                        references=[[word_tokenize(r)]])['google_bleu'] \
                    for r,g in zip(golds_str, decompositions_str)]
    return gleu_scores


def get_chrf_score(decompositions_str: [str], golds_str: [str]):
    from sacrebleu import CHRF
    # default values in HF chrf implementation
    char_order, word_order, beta, lowercase = 6, 2, 1, False
    whitespace, eps_smoothing = False, False
    chrf = CHRF(char_order, word_order, beta, lowercase, whitespace, eps_smoothing)
    chrf_score = chrf.corpus_score(hypotheses = decompositions_str, 
                               references = [golds_str]).score
    return [chrf_score for _ in range(len(decompositions_str))]
### CHANGE END ###
    

def get_exact_match(decompositions_str:[str], golds_str:[str]):
    return [d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]


def get_sari_score(decompositions_str: [str], golds_str: [str], questions: [str]):
    sources = [q.split(" ") for q in questions]
    predictions = [d.split(" ") for d in decompositions_str]
    targets = [[g.split(" ")] for g in golds_str]
    sari, keep, add, deletion = get_sari(sources, predictions, targets)
    return sari


def get_match_ratio(decompositions_str: [str], golds_str: [str]):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    return sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                            processing="base")


def get_structural_match_ratio(decompositions_str: [str], golds_str: [str]):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    return sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                            processing="structural")


def print_first_example_scores(evaluation_dict, num_examples):
    for i in range(num_examples):
        print("evaluating example #{}".format(i))
        for k,v in evaluation_dict.items():
            if isinstance(v[i], numbers.Number):
                print("\t{}: {}".format(k, round(v[i], 3)))
            else:
                print("\t{}: {}".format(k, v[i]))


def print_score_stats(evaluation_dict):
    skiped_samples = {}
    mean_scores = {}

    print("\noverall scores:")
    for key in evaluation_dict:
        # ignore keys that do not store scores
        if key in ["id", "question", "gold", "prediction"]:
            continue
        score_name, scores = key, evaluation_dict[key]

        # ignore examples without a score
        if None in scores:
            scores_ = [score for score in scores if score is not None]
            skiped_samples[key] = len(scores)-len(scores_)
        else:
            scores_ = scores

        mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
        print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
            score_name, mean_score, max_score, min_score))
        mean_scores[score_name] = mean_score

    for score, skiped in skiped_samples.items():
        print(f"skipped {skiped} examples when computing {score}.")

    return mean_scores

### CHANGE START ###
def write_evaluation_output(strip_qmark_whitespace, output_path_base, num_examples, **kwargs):
    str_strip_qmark_whitespace = '_strip_qmark_whitespace' if strip_qmark_whitespace else ''
    # write evaluation summary
    with open(output_path_base + f'_summary{str_strip_qmark_whitespace}.tsv', 'w') as fd:
        fd.write('\t'.join([key for key in sorted(kwargs.keys())]) + '\n')
        for i in range(num_examples):
            fd.write('\t'.join([str(kwargs[key][i]) for key in sorted(kwargs.keys())]) + '\n')

    # write evaluation scores per example
    df = pd.DataFrame.from_dict(kwargs, orient="columns")
    df.to_csv(output_path_base + f'_full{str_strip_qmark_whitespace}.tsv', sep='\t', index=False)
	
    ### CHANGE END ###

def write_evaluation_results(mean_scores, output_path_base = None):
    # write mean evaluation scores
	# leaderboard results must be in results/metrics.json
    # with open(, 'w+') as json_file:
    savepaths = ['../results/metrics.json']
    if output_path_base is not None: 
        savepaths.append(output_path_base + '_metrics.json')
    for savepath in savepaths:
        try: 
            with open(savepath, 'w+', encoding = 'utf-8') as json_file:
                json.dump(mean_scores, json_file)
        except: 
            print(f'Could not write to {savepath}')


def format_qdmr(input:str):
    # replace multiple whitespaces with a single whitespace.
    input = ' '.join(input.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = input.split(';')
    parts = [re.sub(r'return', '', part.strip().strip('\r')) for part in parts]

    # replacing references with special tokens, for example replacing #2 with @@2@@.
    parts = [re.sub(r'#(\d+)', '@@\g<1>@@', part) for part in parts]

    ### CHANGE START ###
    # remove empty strings that arise due to parsing issues with CoT formats
    parts = [p for p in parts if p.strip()]
    ### CHANGE END ###

    return Decomposition(parts)


def main(args):
    # load data
    try:
        metadata = pd.read_csv(args.dataset_file)
        ctr = metadata['question_text'].isna().sum()
        print('(LABELS) THERE ARE {} NaNs in the "question_text" column'.format(ctr))
        metadata['question_text'] = metadata['question_text'].fillna("")
        ids = metadata["question_id"].to_list()
        questions = metadata["question_text"].to_list()
        golds = [format_qdmr(decomp) for decomp in metadata["decomposition"].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load dataset file {args.dataset_file}", ex)

    # load predictions
    try:
        preds_file = pd.read_csv(args.preds_file)
        # some lines may have NaN values, so we need to fill them with empty strings
        ctr = preds_file['decomposition'].isna().sum()
        print('(PREDS) THERE ARE {} NaNs in the "decomposition" column'.format(ctr))
        preds_file['decomposition'] = preds_file['decomposition'].fillna("")
        predictions = [format_qdmr(pred) for pred in preds_file['decomposition'].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load predictions file {args.preds_file}", ex)

    assert len(golds) == len(predictions), "mismatch number of gold questions and predictions"

    if args.random_n and len(golds) > args.random_n:
        indices = random.sample(range(len(ids)), args.random_n)
        ids = [ids[i] for i in indices]
        questions = [questions[i] for i in indices]
        golds = [golds[i] for i in indices]
        predictions = [predictions[i] for i in indices]

    if not args.no_cache:
        norm_rules.load_cache(args.dataset_file.replace(".csv", "__cache"))
    ### CHANGE START ###
    res, fine_grained_scores = evaluate(ids=ids,
    ### CHANGE END ### 
                   questions=questions,
                   golds=golds,
                   decompositions=predictions,
                   metadata=metadata,
                   output_path_base=args.output_file_base,
                   metrics=args.metrics,
                   ### CHANGE START ###
                   str_strip_qmark_whitespace=args.strip_qmark_whitespace,
                   ### CHANGE END ###
                   )
    if not args.no_cache:
        norm_rules.save_cache(args.dataset_file.replace(".csv", "__cache"))
    
    ### CHANGE START ###
    return res, fine_grained_scores
    ### CHANGE END ###


def validate_args(args):
    # print(os.getcwd())
    # print(args.preds_file)
    # input question(s) for decomposition are provided.
    assert args.preds_file and args.dataset_file

    # input files exist.
    if args.dataset_file:
        assert os.path.exists(args.dataset_file)
    if args.preds_file:
        assert os.path.exists(args.preds_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate QDMR predictions")
    parser.add_argument('--dataset_file', type=str, help='path to dataset file')
    parser.add_argument('--preds_file', type=str, help='path to a csv predictions file, with "prediction" column')
    parser.add_argument('--random_n', type=int, default=0,
                        help='choose n random examples from input file')

    parser.add_argument('--no_cache', action='store_true',
                        help="don't cache dependency parsing on normalized metrics")
    parser.add_argument('--output_file_base', type=str, default=None, help='path to output file')
    parser.add_argument('--metrics', nargs='+', default=['exact_match', 'sari', 'ged', 'gleu', 'chrf'], help='path to output file')

    args = parser.parse_args()

    validate_args(args)
    for strip_qmark_whitespace in [True]:#, False]:
        args.strip_qmark_whitespace = strip_qmark_whitespace
        res, fine_grained_scores = main(args)

        # rename for AllenAI leader board
        map = {### CHANGE START ###
                'bleu': 'BLEU', 'normalized_bleu': 'norm_BLEU',
                'gleu': 'GLEU', 'normalized_gleu': 'norm_GLEU',
                'chrf': 'CHRF', 'normalized_chrf': 'norm_CHRF',
                ### CHANGE END ###
                'exact_match': 'EM', 'normalized_exact_match': 'norm_EM', 
                'sari': 'SARI', 'ged': 'GED', 'ged_plus': 'GED+'}
        res = {map.get(k, k): v for k,v in res.items()}
        print(res)
        savepath = os.path.join(os.path.dirname(args.preds_file), 'breakeval_scores')
        if not os.path.exists(savepath): os.makedirs(savepath)
        data_type = re.match(r'text_|amr_', os.path.basename(args.preds_file)).group(0)
        str_strip_qmark_whitespace = '_strip_qmark_whitespace' if args.strip_qmark_whitespace else ''
        with open(f'{savepath}/{data_type}scores{str_strip_qmark_whitespace}.json', 'w+') as json_file:
            json.dump(res, json_file)

        for agg_field, df in fine_grained_scores.items():
            if df is not None:
                df.to_csv(f'{savepath}/{data_type}score_{agg_field}{str_strip_qmark_whitespace}.csv', index = False)