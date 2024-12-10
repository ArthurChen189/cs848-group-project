#!/bin/bash
# Use this for interactive jobs
# salloc --time=0:30:0 --account=def-mlecuyer --gpus-per-node=1 --mem=32G

# Install python virtual environment
module load gcc arrow/17.0.0 python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index --ignore-installed --no-deps -r /home/qfyan/cs848-group-project/cc-req.txt

# Test to make sure that pyarrow is loaded
python -c "import pyarrow"

cd /home/qfyan/cs848-group-project/be_great/experiments
