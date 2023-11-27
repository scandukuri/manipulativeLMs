#!/bin/bash
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPU
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
# activate your env
conda activate py310-jphilipp 

# cd to the training script 
cd ~/research_projects/social_tuning/manipulativeLMs/redteaming-eval

# todo: change confid below to make input output commands shorter
python3 autoeval.py