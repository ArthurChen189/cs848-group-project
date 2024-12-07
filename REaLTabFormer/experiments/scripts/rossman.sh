#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G               # memory per node
#SBATCH --time=0-10:00
#SBATCH --output=/home/qfyan/cs848-group-project/REaLTabFormer/experiments/logs/rossman.log

python /home/qfyan/cs848-group-project/REaLTabFormer/train.py --parent_df "/home/qfyan/projects-qfyan/privacy_data/rossman_parent.csv" --child_df "/home/qfyan/projects-qfyan/privacy_data/rossman_child.csv" --join_on "Store" --output_dir "/home/qfyan/projects-qfyan/privacy_checkpoints"