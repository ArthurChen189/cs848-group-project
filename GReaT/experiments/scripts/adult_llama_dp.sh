#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G               # memory per node
#SBATCH --time=0-40:00
#SBATCH --output=/home/qfyan/cs848-group-project/be_great/experiments/logs/adult_llama_dp.log

source /home/qfyan/cs848-group-project/cc-setup.sh

# Run test example
python ~/cs848-group-project/be_great/experiments/adult_llama_dp.py 
