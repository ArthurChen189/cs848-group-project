#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G               # memory per node
#SBATCH --time=0-35:00
#SBATCH --output=/home/qfyan/cs848-group-project/be_great/experiments/logs/rossmann_gpt2_dp_train.log

source /home/qfyan/cs848-group-project/cc-setup.sh

# Run test example
python ~/cs848-group-project/be_great/experiments/gpt2_train/rossman_gpt2_dp.py 
