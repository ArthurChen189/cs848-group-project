#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G               # memory per node
#SBATCH --time=0-10:00
#SBATCH --output=/home/qfyan/cs848-group-project/REaLTabFormer/experiments/logs/rossman.log

python ../../train.py --parent_df "" --child_df "" --join_on "" --output_dir "" --batch_size 32