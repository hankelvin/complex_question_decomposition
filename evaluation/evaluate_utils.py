import torch, copy, re, string
import random, os, time, collections
import numpy as np

# 
METEOR_JAR = 'evaluation/meteor/meteor-1.5.jar'

def clean_special_tokens(text, tokens):
    newtext = copy.copy(text)
    for t in tokens: 
        if not t: continue
        newtext = newtext.replace(t, '')
    return newtext.strip()

def evaluate(references, generated, idxes, lang = 'en',
            auto_scores = ['bleu', 'rouge', 'bertscore', 'gleu', 'em'],
            bert_score_models = [('bertbase', 'bert-base-uncased')], 
            bert_score_rescale_with_baseline = False,
            strip_func = None, em_lower = False):
    '''
    Given a set of generated questions score on BLEU, ROUGE, BERTScore, METEOR.
    NOTE: BERTScore does not have full set of rescale to baselines for multilingual case
    https://github.com/Tiiiger/bert_score/tree/master/bert_score/rescale_baseline/,
    therefore setting bert_score_rescale_with_baseline to False
    '''
    import sacrebleu, datasets, subprocess, tempfile, codecs
    from nltk import word_tokenize

    bertscore = datasets.load_metric('bertscore', experiment_id=random.randint(1,1000))
    rouge = datasets.load_metric('rouge', experiment_id=random.randint(1,1000), seed = 54506)
    google_bleu = datasets.load_metric('google_bleu', experiment_id=random.randint(1,1000),)
        
    score, stage = {}, 'test'
    assert len(references) == len(generated)
    score['count'] = len(references)
    score['idxes'] = idxes

    for score_name in auto_scores:
        __references, __generated = copy.deepcopy(references), copy.deepcopy(generated)
        if strip_func: 
            __references = [strip_func(l) for l in __references]
            __generated  = [strip_func(l) for l in __generated]
    
        # i. compute BLEU 
        if score_name == 'bleu':
            bleu_obj = sacrebleu.corpus_bleu(__generated, [__references])
            score.update({stage + f'_bleu': bleu_obj.score, 
                        stage + f'_reflen': bleu_obj.ref_len, 
                        stage + f'_predlen': bleu_obj.sys_len})
            print('BLEU done:', bleu_obj.score)

        # ii. compute bertscore 
        elif score_name == 'bertscore':
            for bsc_code, bsc_model in bert_score_models:
                torch.cuda.empty_cache()
                bsz = 24 if 'byt5' in bsc_code else 256
                bertscoref1_all = bertscore.compute(predictions = __generated, references = __references, 
                                lang = lang, model_type = bsc_model, batch_size = bsz,
                                rescale_with_baseline = bert_score_rescale_with_baseline,
                                idf = False, device = torch.device('cuda') \
                                if torch.cuda.is_available() else torch.device('cpu'))['f1'] 
                score[stage+f'_bertscoref1_{bsc_code}_avg']         = np.mean(bertscoref1_all)
                score[stage+f'_bertscoref1_{bsc_code}_allscores']   = bertscoref1_all
                print(f'BERTScore done {bsc_code}:', float(score[stage+f'_bertscoref1_{bsc_code}_avg']))

        # iiia. compute RougeL. 
        elif score_name == 'rouge':
            rougeL_all = rouge.compute(predictions = __generated, references = __references, 
                        rouge_types = ['rougeL'], use_stemmer = False)['rougeL']
            score[stage+f'_rougeLf1_avg'] = float(rougeL_all.mid.fmeasure)
            print('ROUGE-L done:', float(rougeL_all.mid.fmeasure))
        
        # iv. compute METEOR
        elif score_name == 'meteor':
            gen_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=str(time.time()))
            ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=str(time.time()))

            with codecs.open(gen_tmp.name, 'w', 'utf-8') as f: 
                f.write('\n'.join([i.replace('\n', ' ') if i else ' ' for i in __generated]))
            with codecs.open(ref_tmp.name, 'w', 'utf-8') as f: 
                f.write('\n'.join([i.replace('\n', ' ') if i else ' ' for i in __references]))
            
            meteor_cmd = ['java', '-Xmx12G', '-jar',  METEOR_JAR, 
                        gen_tmp.name, ref_tmp.name, '-l', 'en', '-norm', '-r', '1']
            try: 
                proc = subprocess.Popen(' '.join(meteor_cmd), stdout=subprocess.PIPE, shell=True)
                met_result = proc.stdout.readlines()[-1].decode('utf-8').strip('\n')
            except IndexError: # if stdout empty, retry with shell=False (i.e. perhaps running on Jupyter)
                proc = subprocess.Popen(meteor_cmd, stdout=subprocess.PIPE, shell=False)
                met_result = proc.stdout.readlines()[-1].decode('utf-8').strip('\n')
            
            # close proc and remove temp txt files 
            proc.terminate()
            for tmp in [gen_tmp, ref_tmp]:
                try: os.remove(tmp.name)
                except FileNotFoundError: pass 
            if 'Final score:' in met_result: 
                # extract score 
                # score separated by multiple whitespaces (not \t)
                met_result = float(met_result.split()[-1]) 
                score[stage+f'_meteor'] = met_result
                print('METEOR done:', met_result)
            else: 
                score[stage+f'_meteor'] = 0
                print('METEOR .jar did not return Final score:')

        elif score_name == 'f1':
            assert len(__generated) == len(__references)
            tokenf1_scores = [calculate_f1_squad(r, g) for r,g in zip(__references, __generated)]
            score[stage+f'_tokenf1']            = np.mean(tokenf1_scores)
            score[stage+f'_tokenf1_allscores']  = tokenf1_scores
            print('Token F1 done:', score[stage+f'_tokenf1'])

        elif score_name == 'em':
            if em_lower:
                ans_em_scores = [1 if r.strip().lower() == g.strip().lower() \
                             else 0 for r,g in zip(__references, __generated)]
            else:
                ans_em_scores = [1 if r.strip() == g.strip() \
                                 else 0 for r,g in zip(__references, __generated)]
            score[stage+f'_em']             = np.mean(ans_em_scores)
            score[stage+f'_em_allscores']   = ans_em_scores
            print('Answer EM done:', score[stage+f'_em'])

        elif score_name == 'gleu':
            gleus = [google_bleu.compute(predictions=[word_tokenize(g)], 
                                        references=[[word_tokenize(r)]])['google_bleu'] \
                    for r,g in zip(__references, __generated)]
            score[stage+f'_gleu_avg']       = np.mean(gleus)
            score[stage+f'_gleu_allscores'] = gleus
            print('GLEU done:', score[stage+f'_gleu_avg'])
                
    return score


def calculate_f1_squad(
    a_gold: str,
    a_pred: str
) -> float:
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(s):
        if not s: return []
        return normalize_answer(s).split()

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