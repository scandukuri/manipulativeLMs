import pandas as pd
import numpy as np
import bert_score
import sacrebleu
from rouge_score import rouge_scorer
from glob import glob
import argparse, re, nltk, os

def person_split(txt):
    return [x for x in re.split(r"(?=(PERSON|OTHER))",txt) if len(x) and x not in {'PERSON', 'OTHER'}]

def person_subdivide(lst):
    new_lst = []
    for x in lst:
        for y in person_split(x):
            new_lst.append(y)
    return new_lst

def clean_constraints(txt):
    
    txt = re.sub("[\s]+’s", "'s", txt)
    txt = re.sub("[\s]+'s", "'s", txt)
    txt = re.sub("[\s]{2,}", " ", txt)
    txt = re.sub("’", "'", txt)
    txt = re.sub('<[\w//]+>', '', txt)
    txt = re.sub(':', '', txt)
    txt = re.sub('\[PERSON\]', 'PERSON', txt)
    txt = re.sub('\[OTHER\]', 'OTHER', txt)
    txt = re.sub('PERSON.s', "PERSON's", txt)
    txt = re.sub("PERSON(' | ')(s|S)\b", "PERSON's", txt)
    txt = re.sub(r"has not '", "is not '", txt)
    
    constraints = person_subdivide(txt.split('[AND]'))
    
    def clean_constraint(txt):
        txt = re.sub('[\.//]', '', txt)
        return txt.strip()
    
    cleaned = [clean_constraint(c) for c in constraints]
    
    out = [c for c in cleaned if len(c)]
    if len(out):
        return out
    return ['']

ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
def rouge_max(cands, refs):
    scores = {}
    
    preprocess = lambda x : "\n".join(nltk.sent_tokenize(x.strip()))
    for metric in ROUGE_SCORER.rouge_types:
        max_scores = np.array([ max([ROUGE_SCORER.score(preprocess(ref), preprocess(cand))[metric].fmeasure 
                                         for ref in refs
                                    ])
                                   for cand, refs in zip(cands, refs)
                              ])
        scores[metric] = np.mean(np.array(max_scores))
    return scores

def sacrebleu_max(cands, refs):
    max_scores = np.array([ max([sacrebleu.corpus_bleu(cand, ref).score
                                     for ref in refs
                                ])
                               for cand, refs in zip(cands, refs)
                          ])
    return np.mean(np.array(max_scores))


# df --> cands, refs
def get_cands_refs(df, prompt_col, cand_col, refs_col):
    cand_list = []
    refs_list = []
    for prompt in set(df[prompt_col].values):
        consider = df[df[prompt_col]==prompt].copy()
        refs = list(consider[refs_col].values)
        for cand in consider[cand_col].values:
            cand_list.append(cand)
            refs_list.append(refs)
    return cand_list, refs_list

def mean_length(sentences):
    return np.mean(np.array([len(nltk.tokenize.word_tokenize(sent)) for sent in sentences]))

def compute_metrics(df, prompt_col, cand_col, refs_col):
    
    cands, refs = get_cands_refs(df, prompt_col, cand_col, refs_col) 
    scores = rouge_max(cands, refs)
    bert_p, bert_r, bert_f1 = bert_score.score(cands, refs, lang='en')
    scores['BERTScore_Precision'] = np.average(np.array(bert_p))
    scores['BERTScore_Recall'] = np.average(np.array(bert_r))
    scores['BERTScore'] = np.average(np.array(bert_f1))
    scores['sacrebleu'] = sacrebleu_max(cands, refs)
    scores['mean_length'] = mean_length(df[cand_col].values)
    
    return scores

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../results/*', help='glob regex for input models')
    parser.add_argument('--output', type=str, help='path to directory for outputting results')
    parser.add_argument('--target', default='constraints')
    parser.add_argument('--prompt_col', type=str, default='setting-behavior')
    args = parser.parse_args()

    GEN_COL = 'constraints_generated'

    data = {}
    i=0
    for dirname in list(glob(args.input)):
        print(dirname)
        for fn in list(glob(f"{dirname}/test_generations*.csv")):
            print('\t', fn)

            generations=pd.read_csv(fn)
            generations[GEN_COL] = [' [AND] '.join(clean_constraints(c)) for c in generations[GEN_COL].values]
            
            print(generations[:10])

            results = compute_metrics(generations, prompt_col=args.prompt_col, cand_col=GEN_COL, refs_col=args.target)

            typ = "greedy"
            if 'beams3' in fn:
                typ='beam'
            elif 'p0.9' in fn:
                typ='p=0.9'

            model = 'OTHER'
            if 'bart' in fn:
                model='bart'
            elif 'gpt' in fn:
                model='gpt'
            elif 't5' in fn:
                model='t5'
            elif 'alpaca' in fn:
                model='alpaca'

            data[i] = results
            data[i]['decoding'] = typ
            data[i]['model'] = model
            data[i]['fn'] = fn

            print(data)

            i+=1

    out = pd.DataFrame().from_dict(data, orient='index').sort_values(['model', 'decoding'])

    print(out)
    out.to_csv(args.output)    