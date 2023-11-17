
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

# Load any necessary modules or environment variables here
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

# activate conda env
conda activate py310-jphilipp # i tried installing packages here already

# Set the CUDA_VISIBLE_DEVICES environment variable (take those which are free after first checking cgpu)
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Run your script
wandb login --relogin 0242cef7ea759b3e7b2ff2fab0b7ddf5997f57f8 # i'd recommend doing this through an env variable prior to submitting the job or smth that is looked for in your training script.py


cd /scr/jphilipp/manipulativeLMs/

torchrun --standalone \
    --nproc_per_node=4 \
    training/generation_traning.py \
    --model_checkpoint 'alpaca_7b' --architecture 'causal-lm' \
    --input 'data/normbank/normbank.csv' --output 'models/normbank-alpaca_7b/' \
    --save_total_limit 10 --save_steps 1000
