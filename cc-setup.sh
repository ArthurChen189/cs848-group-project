#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G               # memory per node
#SBATCH --time=0-03:00

# Use this for interactive jobs
# salloc --time=1:0:0 --account=def-mlecuyer --gpus-per-node=1 --mem=16G

# Install virtual environment
module load gcc arrow python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r /home/qfyan/cs848-group-project/cc-req.txt

# Run test example
# python ~/cs848-group-project/be_great/examples/test_train.py 