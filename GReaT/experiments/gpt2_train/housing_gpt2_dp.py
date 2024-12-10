TASK_NAME = "housing_gpt2_dp_train"
import os
import sys

sys.path.insert(0, "/home/qfyan/cs848-group-project/be_great")

from be_great import GReaT
from src.DPLLMTGen import DPLLMTGen
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)
import pandas 

# from datasets import load_dataset
data = pandas.read_csv("/home/qfyan/projects-qfyan/privacy_data/datasets/preprocessed/california_housing/california_housing_train.csv")
print(data.head())
column_names = data.columns

great = DPLLMTGen(
    "/home/qfyan/projects-qfyan/openai-community/gpt2",
    save_steps=4000,
    logging_steps=20,
    experiment_dir=f"/home/qfyan/scratch/privacy_checkpoints/{TASK_NAME}",
    logging_dir=f'/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}/logs',
    batch_size=32,                 # Batch Size
    # lr_scheduler_type="constant", # Specify the learning rate scheduler 
    stage1_epochs = 100,
    stage2_epochs = 50,
    stage1_lr =1e-5,
    stage2_lr=5e-5,
    loss_alpha=0.65,
    loss_beta=0.1,
    loss_lmbda=1.0,
    per_sample_max_grad_norm=1., 
    target_epsilon=10., 
    stage2_batch_size=8,
)


trainer = great.fit(data, column_names=column_names)

great.save(f"/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}_final")

# Generate synthetic data
samples = great.sample(min(len(data), 1000), k=16, max_length=1000, device="cuda")
samples.to_csv(f"/home/qfyan/cs848-group-project/be_great/experiments/samples/{TASK_NAME}.csv")
