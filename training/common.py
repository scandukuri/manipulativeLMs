import pandas as pd
from transformers import Trainer
import argparse
from datasets import load_dataset, load_metric
import pandas as pd
from collections import OrderedDict, Counter
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer
import nltk, os, csv, json, random
import numpy as np
import torch
from tqdm import tqdm
import json

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def tokenize(string, tokenizer, eos_id=None):
    return tokenizer(string, padding='max_length', truncation=True, max_length=64) #later pull this from args somehow
    #return tokenizer(string)

def preprocess(examples, tokenizer, format_string):
    def build(r, f):
        s = f"[SETTING] {r['setting-behavior']}[NORM] {r['norm']}[CONSTRAINTS] {r['constraints']}"
        t = s
        return s, t
    source_target = [build(row, format_string)
                  for _, row in pd.DataFrame(dict(examples)).iterrows()] ## cast to dict first
    source = [tup[0] for tup in source_target]
    target = [tup[1] if len(tup)>1 else "" for tup in source_target]
    
    model_inputs = tokenize(source, tokenizer) #tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = tokenize(target, tokenizer) #tokenizer(target)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def decode(args, df, model, tokenizer, skip_special_tokens=True, remove_history=False):
    model = model
    is_greedy = (args.top_p == 0) and (args.top_k == 0) and (args.beams == 0)
    
    eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]
    generations = []
    
    for _, row in df.iterrows():
        input_ids = torch.tensor([row['input_ids']], device='cuda')
        
        out = model.generate(
                input_ids,
                do_sample=args.beams == 0,
                max_length=args.maxlen,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p > 0 else None,
                top_k=args.top_k if args.top_k > 0 else None,
                num_beams=args.beams if args.beams > 0 else None,
                early_stopping=True,
                pad_token_id=50256,
                no_repeat_ngram_size=3,
                eos_token_id=eos_token_id
            )
        if remove_history:
            generations.append(tokenizer.decode(out[:, input_ids.shape[-1]:][0], skip_special_tokens=skip_special_tokens))
        
        else:
            generations.append(tokenizer.decode(out[0],
                                            skip_special_tokens=skip_special_tokens
                                           ))
    return generations