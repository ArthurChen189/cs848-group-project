TASK_NAME = "adult_gpt2_train"

import os
import sys

sys.path.insert(0, "/home/qfyan/cs848-group-project/be_great")

from be_great import GReaT
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

import pandas 

# from datasets import load_dataset
data = pandas.read_csv("/home/qfyan/projects-qfyan/privacy_data/datasets/preprocessed/adult/adult_train.csv")
print(data.head())

column_names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
data.columns = column_names

great = GReaT(
    "/home/qfyan/projects-qfyan/openai-community/gpt2",
    epochs=310,
    save_steps=4000,
    logging_steps=20,
    experiment_dir=f"/home/qfyan/scratch/privacy_checkpoints/{TASK_NAME}",
    logging_dir=f'/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}/logs',
    batch_size=32,
    learning_rate=5e-5,
)

trainer = great.fit(data, column_names=column_names, resume_from_checkpoint=True)

great.save(f"/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}_final")

# Generate synthetic data
samples = great.sample(min(len(data), 1000), k=16, max_length=1000, device="cuda")
samples.to_csv(f"/home/qfyan/cs848-group-project/be_great/experiments/samples/{TASK_NAME}.csv")
