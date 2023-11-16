#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=128
#SBATCH --time=3-0
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load any necessary modules or environment variables here
# For example:



__conda_setup="$('/scr/jphilipp/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
            eval "$__conda_setup"
    else
        if [ -f "/scr/jphilipp/miniconda3/etc/profile.d/conda.sh" ]; then
                . "/scr/jphilipp/miniconda3/etc/profile.d/conda.sh"
        else
                export PATH="/scr/jphilipp/miniconda3/bin:$PATH"
        fi
fi
unset __conda_setup

#
pip3 install wandb torch transformers peft datasets nltk tqdm bert_score sacrebleu rouge_score

# Run your script
wandb login --relogin 0242cef7ea759b3e7b2ff2fab0b7ddf5997f57f8
cd /scr/jphilipp/manipulativeLMs/
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone \
    --nproc_per_node=4 \
    training/generation_traning.py \
    --model_checkpoint 'alpaca_7b' --architecture 'causal-lm' \
    --input 'data/normbank/normbank.csv' --output 'models/normbank-alpaca_7b/' \
    --save_total_limit 10 --save_steps 1000