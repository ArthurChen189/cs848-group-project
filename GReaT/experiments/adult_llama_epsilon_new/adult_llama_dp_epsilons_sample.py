import os
import sys

sys.path.insert(0, "/home/qfyan/FedFetch/be_great")

from be_great import GReaT
from src.DPLLMTGen import DPLLMTGen
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)
import pandas 

# from datasets import load_dataset
data = pandas.read_csv("~/FedFetch/datasets/preprocessed/adult/adult_train.csv")
print(data.head())
column_names = data.columns

def sample_epsilons(episilon):
    task_name = f"adult_llama_dp_new_{episilon}"
    print(f"Start sampling {task_name}")

    great = GReaT.load_peft_model(f"/home/qfyan/privacy_checkpoints/{task_name}_final")

    samples = great.sample(len(data), k=128, max_length=1000, device="cuda")
    samples.to_csv(f"~/FedFetch/be_great/experiments/samples/adult/{task_name}_{len(data)}.csv")

    # print(f"Start sampling {len(data)}")
    # samples = great.sample(len(data), k=16, max_length=1000, device="cuda")
    # samples.to_csv(f"~/FedFetch/be_great/experiments/samples/adult/{TASK_NAME}_{len(data)}.csv")


task_names = [0.5, 1, 3, 5, 10]

for tn in task_names:
    sample_epsilons(tn)

