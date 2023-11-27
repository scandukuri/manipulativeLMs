import pandas as pd
import numpy as np
import argparse, random, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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
    #model = model
    print("Beginning generation process.")
    
    #eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]
    generations = []
    
    for i, row in df.iterrows():
        print(f"Example {i + 1}/{len(df)}")
        input_ids = tokenizer.encode(row['request'], return_tensors="pt").to(model.device)
        # out = model.generate(
        #         input_ids,
        #         #GenerationConfig(config_file_name=f'/sailhome/jphilipp/research_projects/social_tuning/manipulativeLMs/redteaming-eval/generation_configs/{args.decode_strategy}.json'),
        #     )
        out = model.generate(input_ids, max_length = 1000, num_beams=1)
        if remove_history:
            generations.append(tokenizer.decode(out[:, input_ids.shape[-1]:][0], skip_special_tokens=skip_special_tokens))
            generations.append(tokenizer.decode(out[0],
                                            skip_special_tokens=skip_special_tokens
                                           ))
        else:
            generations.append(tokenizer.decode(out[0],
                                            skip_special_tokens=skip_special_tokens
                                           ))
        print(generations[-1])
    print("Finished generating responses.")
    return generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_dir', type=str, default='/scr/jphilipp/manipulativeLM-nodecontents/', help='path to all data or model directories on the node (parent directory for subdirectories like pretrained_models, rawdata, output_models etc)')
    parser.add_argument('--models_subdir', type=str, default='output_models/', choices=['pretrained_models/', 'output_models/'], help='choice of subpath to models within the compute node directory')
    parser.add_argument('--tokenizers_subdir', type=str, default='output_models/', choices=['pretrained_models/', 'output_models/'], help='choice of subpath to tokenizers within the compute node directory')
    parser.add_argument('--evaldata', type=str, default='/sailhome/jphilipp/research_projects/social_tuning/manipulativeLMs/redteaming-data/raw_redteaming_requests.csv', help='full path to evaluation data csv')
    parser.add_argument('--evalcolumn', type=str, default='request', help='column name of eval prompts in args.evaldata')
    parser.add_argument('--model_checkpoint', type=str, default='alpaca_7b_normbankFT', choices=['alpaca_7b_normbankFT', 'TinyStories-1M', 'mistralai/Mistral-7B-Instruct-v0.1', 'gpt2', 't5-small', 'facebook/bart-large', 'alpaca_7b', 'better-base', '7B'], help='subpath within model directory to hf style model')
    parser.add_argument('--tokenizer_checkpoint', type=str, default='alpaca_7b_normbankFT', choices=['alpaca_7b_normbankFT', 'TinyStories-1M', 'mistralai/Mistral-7B-Instruct-v0.1', 'gpt2', 't5-small', 'facebook/bart-large', 'alpaca_7b', 'better-base', '7B'], help='subpath within pretrained directory to hf style model directory')
    parser.add_argument('--architecture', default='causal-lm', choices=['seq2seq', 'causal-lm'])
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--decode_strategy', type=str, default='beam', choices=['greedy', 'beam', 'top_p_0.9'])

    args = parser.parse_args()
    set_seed(args.seed)

    ## TODO: fill in generation configs for each of the three kinds from normbank paper
    ## adjust the parser args strategy to make this clean
    model = AutoModelForCausalLM.from_pretrained(os.path.join(args.node_dir, args.models_subdir, args.model_checkpoint), cache_dir='/scr/jphilipp/manipulativeLM-nodecontents')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.node_dir, args.models_subdir, args.tokenizer_checkpoint), cache_dir='/scr/jphilipp/manipulativeLM-nodecontents')


    eval_df = pd.read_csv(args.evaldata).head(100)
    print("Read in eval data.")
    output = eval_df
    print("Created output dataframe.")
    output[args.evalcolumn] = decode(args,
                                eval_df, 
                                model, 
                                tokenizer,
                                remove_history=(args.architecture == 'causal-lm'),
                                skip_special_tokens=True
                                )
    output.to_csv(f'/sailhome/jphilipp/research_projects/social_tuning/manipulativeLMs/redteaming-eval/{args.model_checkpoint}_{args.tokenizer_checkpoint}_{args.evaldata}.csv')
    return 0


if __name__=='__main__':
    main()