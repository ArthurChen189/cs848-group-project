#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G               # memory per node
#SBATCH --time=0-36:00
#SBATCH --output=/home/qfyan/cs848-group-project/be_great/experiments/logs/adult_gpt2_train.log

source /home/qfyan/cs848-group-project/cc-setup.sh

# Run test example
python ~/cs848-group-project/be_great/experiments/gpt2_train/adult_gpt2.py 
