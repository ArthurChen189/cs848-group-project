#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G               # memory per node
#SBATCH --time=0-1:00
#SBATCJ --output=/home/qfyan/cs848-group-project/be_great/experiments/food_%j.log

source /home/qfyan/cs848-group-project/cc-setup.sh

# Run test example
python ~/cs848-group-project/be_great/experiments/food_llama.py 
python ~/cs848-group-project/be_great/experiments/food_llama_sample.py 