from common import *
import os
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from datasets import Features, Value, Dataset, DatasetDict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM



# LoRA HYPERPARAMS
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# Weights and Biases Reporting
os.environ["WANDB_PROJECT"] = "manipulativeLMs" # name your W&B project 
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
# os.chdir('/scr/jphilipp/manipulativeLMs/')

class Seq2SeqTrainerLogger(Seq2SeqTrainer):
    def __init__(self, logfile, *args, **kwargs):
        super(Seq2SeqTrainer, self).__init__(*args, **kwargs)
        self.logfile = logfile
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        with open(self.logfile, 'a') as outfile:
            outfile.write(json.dumps(output) + '\n')
        
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        
class TrainerLogger(Trainer):
    def __init__(self, logfile, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.logfile = logfile
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        with open(self.logfile, 'a') as outfile:
            outfile.write(json.dumps(output) + '\n')
        
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

def final_compute_metrics(decoded_preds, decoded_labels, metric, metric2, tokenizer):
    
    if (not len(decoded_preds)) or (not len(decoded_labels)) or (len(decoded_preds)!=len(decoded_labels)):
        print("Something is wrong with decoded_preds", decoded_preds)
        return {}
    
    # Rouge expects a newline after each sentence
    decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    decoded_labels_expanded = [[x] for x in decoded_labels]
    result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

    # print(result2)
    result['sacrebleu'] = round(result2["score"], 1)
    
    r = {k: round(v, 4) for k, v in result.items()}
    
    return r

def format_str_to_savefile_name(format_str):
    return '_'.join([x for x in format_str.replace("~", 'TARGET').split() if '<' not in x])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_dir', type=str, default='/scr/jphilipp/manipulativeLM-nodecontents/', help='path to all data or model directories on the node (parent directory for subdirectories like pretrained_models, rawdata, output_models etc)')
    parser.add_argument('--pretrained_models_subdir', type=str, default='pretrained_models/', help='subpath to pretrained models within the compute node directory')
    parser.add_argument('--output_models_subdir', type=str, default='output_models/', help='subpath to output models within the compute node directory')
    parser.add_argument('--rawdata_subdir', type=str, default='rawdata/normbank/normbank.csv', help='subpath to raw training data within the compute node directory')
    parser.add_argument('--processeddata_subdir', type=str, default='processeddata/', help='subpath to processed data generated during training flow within the compute node directory')
    parser.add_argument('--model_checkpoint', type=str, default='better-base', choices=['mistralai/Mistral-7B-Instruct-v0.1', 'gpt2', 't5-small', 'facebook/bart-large', 'alpaca_7b', 'better-base', '7B'], help='subpath within pretrained directory to hf style model directory')
    parser.add_argument('--tokenizer_checkpoint', type=str, default='better-base', choices=['mistralai/Mistral-7B-Instruct-v0.1', 'gpt2', 't5-small', 'facebook/bart-large', 'alpaca_7b', 'better-base', '7B'], help='subpath within pretrained directory to hf style model directory')
    parser.add_argument('--architecture', default='causal-lm', choices=['seq2seq', 'causal-lm'])
    parser.add_argument('--model_output', default='FT_TEST', type=str, help = 'subpath under args.output_models_subdir to directory for outputting finetuned model weights and config. Not optional!')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--format_string', type=str, default="setting [BEHAVIOR] behavior [NORM] norm [CONSTRAINTS] ~ constraints", help='how to format the dataset')
    parser.add_argument('--source_name', type=str, default='setting-behavior', help='the name of the source column to write in the out file')
    parser.add_argument('--target_name', type=str, default='constraints', help='the name of the target column to write in the out file')
    parser.add_argument('--maxlen', type=int, default=512, help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--microbatchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)  # adjusted from 20 -> 1 for generation task
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--generate_attributes', action='store_true')
    parser.add_argument('--train_size', type=int, default=-1, help='the number of train datapoints to use as an ablation (default is to use the full train set)')
    
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    set_seed(args.seed)
    
    # configure for architecture
    if args.architecture == '':
        if 'gpt' in args.model_checkpoint or 'alpaca' in args.model_checkpoint:  # adjust when adding new causal-lm types
            args.architecture = 'causal-lm'
        elif 'bart' in args.model_checkpoint or 't5' in args.model_checkpoint:
            args.architecture = 'seq2seq'

    
    
    # prepare out directory
    MODEL_DIR = os.path.join(args.node_dir, args.model_checkpoint)
   

    # read data
    df = pd.read_csv(os.path.join(args.node_dir, args.rawdata_subdir))
    df = df[['setting', 'behavior', 'setting-behavior', 'constraints', 'norm', 'split', 'constraints_given', 'constraint_predict']].copy()
    
    if not os.path.exists(os.path.join(args.node_dir, args.processeddata_subdir + 'tmp/')):
        os.makedirs(os.path.join(args.node_dir, args.processeddata_subdir + 'tmp/'))


    for s in ['train', 'dev', 'test']:
        if (s == 'train') and (args.train_size > 0):
            df[df['split'] == s].sample(n=args.train_size, random_state=args.seed).to_csv(os.path.join(args.node_dir, args.processeddata_subdir, 'tmp/%s.csv' % s), index=False)
        else:
            df[df['split']==s].to_csv(os.path.join(args.node_dir, args.processeddata_subdir, 'tmp/%s.csv' % s), index=False)
    
    metric = load_metric("rouge")
    
    metric2 = load_metric("sacrebleu")
    
    ft = Features({
        'setting': Value('string'),
        'behavior': Value('string'),
        'setting-behavior': Value('string'),
        'constraints': Value('string'),
        'norm': Value('string'),
        'split': Value('string'),
        'constraints_given': Value('string'),
        'constraint_predict': Value('string')
    })

    train_pd = pd.read_csv(os.path.join(args.node_dir, args.processeddata_subdir, 'tmp/train.csv'))
    test_pd = pd.read_csv(os.path.join(args.node_dir, args.processeddata_subdir, 'tmp/test.csv'))
    dev_pd = pd.read_csv(os.path.join(args.node_dir, args.processeddata_subdir, 'tmp/dev.csv'))
    data_pd = {
        "train": train_pd.head(len(train_pd) - len(train_pd) % args.batchsize), 
        "test": test_pd.head(len(test_pd) - len(test_pd) % args.batchsize), 
        "validation": dev_pd.head(len(dev_pd) - len(dev_pd) % args.batchsize)}

    dataset_train = Dataset.from_pandas(data_pd['train'])
    dataset_test = Dataset.from_pandas(data_pd['test'])
    dataset_validation = Dataset.from_pandas(data_pd['validation'])
    
    dataset = DatasetDict({'train': dataset_train,'test': dataset_test,'validation': dataset_validation})
    # model setup
    AutoModel = AutoModelForCausalLM if (args.architecture == 'causal-lm') else AutoModelForSeq2SeqLM
    model = AutoModelForCausalLM.from_pretrained(os.path.join(args.node_dir, args.pretrained_models_subdir, args.model_checkpoint), cache_dir='/scr/jphilipp/manipulativeLM-nodecontents')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.node_dir, args.pretrained_models_subdir, args.tokenizer_checkpoint), cache_dir='/scr/jphilipp/manipulativeLM-nodecontents')
    # model = AutoModelForCausalLM.from_pretrained('/scr/jphilipp/manipulativeLM-nodecontents/pretrained_models/alpaca_7b', cache_dir='/scr/jphilipp/manipulativeLM-nodecontents')
    # tokenizer = AutoTokenizer.from_pretrained('/scr/jphilipp/manipulativeLM-nodecontents/pretrained_models/7B', cache_dir='/scr/jphilipp/manipulativeLM-nodecontents', model_max_length=512)
    
    # add special tokens to tokenizer
    special_tokens = list(
        set(
             [
                "[BEHAVIOR]",
                "[NORM]",
                "[CONSTRAINTS]",
                "[PERSON]",
                "[OTHER]",
                "[AND]",
                "<pad>",
                "<eos>"
            ]
        )
    )
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>" 
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    #init_attribute_embeddings(model, tokenizer, special_tokens)
    args.pad_token_id = tokenizer.pad_token_id
    
    tokenize_format_string = args.format_string.replace("~", "") if args.architecture == 'causal-lm' else args.format_string
    tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, tokenize_format_string), batched=True, batch_size=512)
    
    #tokenized_datasets = tokenized_datasets.remove_columns(dataset_train.column_names)
    print('training sample input', tokenizer.decode(pd.DataFrame(tokenized_datasets['train']).iloc[0]['input_ids'],skip_special_tokens=False) )
    try:
        print('training sample target', tokenizer.decode(pd.DataFrame(tokenized_datasets['train']).iloc[0]['labels'],skip_special_tokens=False) )
    except Exception as e:
        # it doesn't matter; this means there is no "labels" in the dataset because we are using causal-lm
        pass
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) if (args.architecture == 'causal-lm') else DataCollatorForSeq2Seq(tokenizer, model=model)    
    
    results = {}
    

    # create model output path if it doesn't exist
    if not os.path.exists(os.path.join(args.node_dir, args.output_models_subdir, args.model_output)):
        os.makedirs(os.path.join(args.node_dir, args.output_models_subdir, args.model_output)) 
    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), "checkpoints"),
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.microbatchsize,
        per_device_eval_batch_size=args.microbatchsize,
        gradient_accumulation_steps=args.batchsize//args.microbatchsize,  # check on this again later
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        fp16=False,
        seed=args.seed,
        save_strategy='steps',
        save_steps=args.save_steps,
        report_to="wandb"
    )
    
    ## LoRA configuration with hyperparameters
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if args.architecture=='seq2seq': 
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), "checkpoints"),
            evaluation_strategy = "epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batchsize,
            per_device_eval_batch_size=args.batchsize,
            weight_decay=args.weight_decay,
            save_total_limit=args.save_total_limit,
            num_train_epochs=args.epochs,
            predict_with_generate=True,
            fp16=False,
            seed=args.seed,
            save_strategy='epoch'
        )
        trainer = Seq2SeqTrainerLogger(
            os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), 'log.txt'),
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )

    trainer.train()
       
    trainer.save_model(os.path.join(args.node_dir, args.output_models_subdir, args.model_output))

    # tokenize again if using causal-lm because before, the test set did not include targets
    if args.architecture == 'causal-lm':
        tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, args.format_string), batched=True)
        
    out_df = pd.DataFrame(tokenized_datasets['test'])[[args.source_name, args.target_name, 'input_ids']]

    results_df = pd.DataFrame().from_dict(results, orient='index')
    print(results_df)
    results_df.to_csv(os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), f'results_epochs{args.epochs}_batch{args.batchsize}_lr{int(args.lr*(10**5))}_seed{args.seed}.csv'))
    
    if (args.top_p <= 0) and (args.top_k <= 0) and (args.beams <= 0) and (args.architecture == 'seq2seq'):
        raw_pred, _, _ = trainer.predict(tokenized_datasets['test'])
        out_df[args.target_name + '_generated'] = tokenizer.batch_decode(raw_pred, skip_special_tokens=True)
    else:
        out_df[args.target_name + '_generated'] = decode(args, 
                                                         out_df, 
                                                         trainer.model, 
                                                         tokenizer,
                                                         remove_history=(args.architecture == 'causal-lm'),
                                                         skip_special_tokens=True
                                                        )
    out_df.to_csv( os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), f'test_generations_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.csv') )
    
    with open(os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), 'format_string.txt'), 'w') as outfile:
        outfile.write(args.format_string)
        
    results = final_compute_metrics(out_df[args.target_name + '_generated'].values, 
                              out_df[args.target_name].values, 
                              metric, metric2, tokenizer)
    
    fn = os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), f'results_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.json')
    with open(fn, 'w') as outfile:
        json.dump(results, outfile)
        
    torch.save(args, os.path.join(os.path.join(args.node_dir, args.output_models_subdir, args.model_output), "training_args.bin"))


if __name__=='__main__':
    main()