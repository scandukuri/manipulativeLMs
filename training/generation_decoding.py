from common import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str)
    parser.add_argument('--input', type=str, help='path to input file')
    parser.add_argument('--output', type=str, default = 'results', help='path to directory for outputting results')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--format_string', type=str, default="setting [BEHAVIOR] behavior [NORM] norm [CONSTRAINTS] ~ constraints", help='how to format the dataset')
    parser.add_argument('--source_name', type=str, default='setting-behavior', help='the name of the source column to write in the out file')
    parser.add_argument('--target_name', type=str, default='constraints', help='the name of the target column to write in the out file')
    parser.add_argument('--maxlen', type=int, default=512, help='maximum length of the generation')
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--skip_special_tokens', action='store_true')
    
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    set_seed(args.seed)

    # prepare out directory
    OUT_DIR = args.output
    if not os.path.exists( os.path.join(OUT_DIR,'tmp') ):
        os.makedirs( os.path.join(OUT_DIR,'tmp') )
    
    df = pd.read_csv(args.input)
    df[df['split']=='test'].to_csv(os.path.join(OUT_DIR,'tmp/test.csv'), index=False)
    
    dataset = load_dataset('csv', data_files={'test': os.path.join(OUT_DIR,'tmp/test.csv')})
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory)
    
    AutoModel = AutoModelForCausalLM if 'gpt' or 'alpaca' in args.model_directory else AutoModelForSeq2SeqLM
    
    model = AutoModel.from_pretrained(args.model_directory).to('cuda')
    tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, args.format_string), batched=True)
                           
    out_df = pd.DataFrame(tokenized_datasets['test'])[[args.source_name, args.target_name, 'input_ids', 'label']]

    out_df[args.target_name + '_generated'] = decode(args, out_df, 
                                                     model, tokenizer, 
                                                     skip_special_tokens=args.skip_special_tokens, 
                                                     remove_history=('gpt' or 'alpaca' in args.model_directory))
                                                    # this is a temp fix:
                                                    # remove_history conditional should depend on seq2seq vs causal-lm, 
                                                    # not the best idea to list out every model type which needs history removal
                                                    # 
                                                    # to fix, add --architecture as an argument as done in generation_training.py
                                                    # then, remove_history=(args.architecture == 'causal-lm')
    out_df.to_csv( os.path.join(args.output, f'test_generations_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.csv') )
        
if __name__=='__main__':
    main()