TASK_NAME = "adult_llama_dp_new"
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

great = DPLLMTGen.load_peft_model(f"~/FedFetch/privacy_checkpoints/{TASK_NAME}_final")

print(f"Start sampling 1000")
samples = great.sample(min(len(data), 1000), k=64, max_length=1000, device="cuda")
samples.to_csv(f"~/FedFetch/be_great/experiments/samples/adult/{TASK_NAME}_1000.csv")

print(f"Start sampling {len(data)}")
samples = great.sample(len(data), k=64, max_length=1000, device="cuda")
samples.to_csv(f"~/FedFetch/be_great/experiments/samples/adult/{TASK_NAME}_{len(data)}.csv")
