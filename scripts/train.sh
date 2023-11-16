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



__conda_setup="$('/scr/kanishkg/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
    else
	if [ -f "/scr/jphilipp/miniconda3/etc/profile.d/conda.sh" ]; then
		. "/scr/jphilipp/miniconda3/etc/profile.d/conda.sh"
	else
		export PATH="/scr/kanishkg/miniconda3/bin:$PATH"
	fi
fi
unset __conda_setup

conda activate tinytom


# Run your script
wandb login --relogin 0242cef7ea759b3e7b2ff2fab0b7ddf5997f57f8
cd /afs/cs.stanford.edu/u/jphilipp/research_projects/social_tuning/
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone \ 
    --nproc_per_node=4 \
    manipulativeLMs/training/generation_traning.py \ 
    --model_checkpoint 'alpaca_7b' --architecture 'causal-lm' --input 'data/normbank/normbank.csv' --output 'normbank-alpaca_7b/'