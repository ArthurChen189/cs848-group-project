#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G               # memory per node
#SBATCH --time=0-2:00
#SBATCH --output=/home/qfyan/cs848-group-project/be_great/experiments/logs/diabetes_gpt2_dp.log

source /home/qfyan/cs848-group-project/cc-setup.sh

# Run test example
python ~/cs848-group-project/be_great/experiments/diabetes_gpt2_dp.py 
