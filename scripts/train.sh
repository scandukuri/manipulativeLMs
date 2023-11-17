
#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=128
#SBATCH --time=3-0
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# load conda
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

# activate your env
conda activate py310-jphilipp # i tried installing packages here already

# set visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run your script
wandb login --relogin 0242cef7ea759b3e7b2ff2fab0b7ddf5997f57f8 # i'd recommend doing this through an env variable prior to submitting the job or smth that is looked for in your training script.py

# cd to the training script 
cd ~/research_projects/social_tuning/manipulativeLMs/training

# todo: change confid below to make input output commands shorter
torchrun --standalone \
    --nproc_per_node=4 train.py \
    --node_dir '/scr/jphilipp/manipulativeLM-nodecontents' --model_checkpoint 'better-base' --architecture 'causal-lm' \
    --input 'normbank/normbank.csv' --output 'better-base/' \
    --save_total_limit 10 --save_steps 1000


#      git clone https://huggingface.co/agi-css/better-base
