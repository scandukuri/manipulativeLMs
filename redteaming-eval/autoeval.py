import pandas as pd
import numpy as np
import argparse, random, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pickle
from peft import PeftModel, get_peft_model, get_peft_model_state_dict


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def decode(args, df, model, tokenizer, skip_special_tokens=True, remove_history=True):
    print("Beginning generation process.")
    generations = []
    completions = []

    for i, row in df.iterrows():
        if (i + 1) % 50 == 0:
            print(f"Finished example {i + 1}")
        input_ids = tokenizer.encode(row['request'], return_tensors="pt").to(model.device)
        out = model.generate(input_ids, max_new_tokens=300).to(model.device)
        generations.append(tokenizer.decode(out[0],
                                            skip_special_tokens=skip_special_tokens
                                           ))
        completions.append(tokenizer.decode(out[:, input_ids.shape[-1]:][0], skip_special_tokens=skip_special_tokens))
        del input_ids
        del out

    print("Finished generating responses.")
    with open(f'/sailhome/andukuri/research_projects/social_tuning/manipulativeLMs/redteaming-eval/{args.output_pickle}_generations.pkl', 'wb') as f:
        pickle.dump(generations, f)
    with open(f'/sailhome/andukuri/research_projects/social_tuning/manipulativeLMs/redteaming-eval/{args.output_pickle}_completions.pkl', 'wb') as f:
        pickle.dump(completions, f)
    
    return generations, completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_dir', type=str, default='/scr/andukuri/manipulativeLMs-hgx/', help='path to all data or model directories on the node (parent directory for subdirectories like pretrained_models, rawdata, output_models etc)')
    parser.add_argument('--models_subdir', type=str, default='output_models/', choices=['pretrained_models/', 'output_models/'], help='choice of subpath to models within the compute node directory')
    parser.add_argument('--tokenizers_subdir', type=str, default='output_models/', choices=['pretrained_models/', 'output_models/'], help='choice of subpath to tokenizers within the compute node directory')
    parser.add_argument('--evaldata', type=str, default='/sailhome/andukuri/research_projects/social_tuning/manipulativeLMs/redteaming-data/raw_redteaming_requests.csv', help='full path to evaluation data csv')
    parser.add_argument('--evalcolumn', type=str, default='request', help='column name of eval prompts in args.evaldata')
    parser.add_argument('--model_checkpoint', type=str, default='alpaca_7b_normbankFT_run3', choices=['alpaca_7b_normbankFT_run3', 'alpaca_7b_normbankFT_run3/checkpoints/checkpoint-2000', 'TinyStories-1M', 'mistralai/Mistral-7B-Instruct-v0.1', 'gpt2', 't5-small', 'facebook/bart-large', 'alpaca_7b', 'better-base', '7B'], help='subpath within model directory to hf style model')
    parser.add_argument('--tokenizer_checkpoint', type=str, default='alpaca_7b_normbankFT_run3', choices=['alpaca_7b_normbankFT_run3', 'alpaca_7b_normbankFT_run3/checkpoints/checkpoint-2000','TinyStories-1M', 'mistralai/Mistral-7B-Instruct-v0.1', 'gpt2', 't5-small', 'facebook/bart-large', 'alpaca_7b', 'better-base', '7B'], help='subpath within pretrained directory to hf style model directory')
    parser.add_argument('--architecture', default='causal-lm', choices=['seq2seq', 'causal-lm'])
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--decode_strategy', type=str, default='beam', choices=['greedy', 'beam', 'top_p_0.9'])
    parser.add_argument('--output_pickle', type=str)
    parser.add_argument('--peft', type=bool, default=False)
    args = parser.parse_args()
    set_seed(args.seed)

    ## TODO: fill in generation configs for each of the three kinds from normbank paper
    ## adjust the parser args strategy to make this clean
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(os.path.join(args.node_dir, args.models_subdir, args.model_checkpoint), cache_dir=args.node_dir).to(device)
    if args.peft:
        lora_model = PeftModel.from_pretrained(model, os.path.join(args.node_dir, args.models_subdir, args.tokenizer_checkpoint), cache_dir=args.node_dir).to(device)
        del model
        model = lora_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.node_dir, args.models_subdir, args.tokenizer_checkpoint), cache_dir=args.node_dir)


    eval_df = pd.read_csv(args.evaldata)
    print("Read in eval data.")
    output = eval_df
    print("Created output dataframe.")
    output[args.evalcolumn], output['completions'] = decode(args,
                                eval_df, 
                                model, 
                                tokenizer,
                                remove_history=(args.architecture == 'causal-lm'),
                                skip_special_tokens=True
                                )
    output.to_csv(f'/sailhome//research_projects/social_tuning/manipulativeLMs/redteaming-eval/{args.output_pickle}.csv')
    return 0


if __name__=='__main__':
    main()