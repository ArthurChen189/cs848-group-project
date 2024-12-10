#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G               # memory per node
#SBATCH --time=0-8:00
#SBATCH --output=/home/qfyan/cs848-group-project/be_great/experiments/logs/adult_llama_dp_3.log

source /home/qfyan/cs848-group-project/cc-setup.sh

# Run test example
python ~/cs848-group-project/be_great/experiments/adult_llama_epsilon/adult_llama_dp_3.py 
