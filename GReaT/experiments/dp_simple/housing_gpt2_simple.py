TASK_NAME = "housing_gpt2_dp_simple"
import os
import sys

sys.path.insert(0, "/home/qfyan/FedFetch/be_great")

from be_great import GReaT
from be_great import GReaTDP
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)
import pandas 

# from datasets import load_dataset
data = pandas.read_csv("~/FedFetch/datasets/preprocessed/california_housing/california_housing_train.csv")
print(data.head())
column_names = data.columns

great = GReaTDP(
    "gpt2",
    epochs=150,
    save_steps=20000,
    logging_steps=100,
    experiment_dir=f"/home/qfyan/scratch/privacy_checkpoints/{TASK_NAME}",
    logging_dir=f'/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}/logs',
    batch_size=128,                 # Batch Size
    # lr_scheduler_type="constant", # Specify the learning rate scheduler 
    learning_rate=5e-5,
    per_sample_max_grad_norm=1., 
    target_epsilon=10., 
    max_physical_batch_size=16,
)


trainer = great.fit(data, column_names=column_names)

great.save(f"/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}_final")

# Generate synthetic data
samples = great.sample(len(data), k=128, max_length=1000, device="cuda")
samples.to_csv(f"/home/qfyan/cs848-group-project/be_great/experiments/samples/{TASK_NAME}.csv")
