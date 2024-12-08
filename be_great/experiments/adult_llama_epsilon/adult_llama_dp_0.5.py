TASK_NAME = "adult_llama_dp_0.5"
import os
import sys

sys.path.insert(0, "/home/qfyan/cs848-group-project/be_great")

from be_great import GReaT
from be_great.DPLLMTGen import DPLLMTGen
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)
import pandas 

# from datasets import load_dataset
data = pandas.read_csv("/home/qfyan/projects-qfyan/privacy_data/adult.csv")
data = data.iloc[:, 1:]
print(data.head())

column_names = ["age", "work class", "count in census", "education", "education number", "marital status", "occupation", "relationship", "race", "sex", "capital gain", "capital loss", "hours per week", "native country"]
data.columns = column_names

great = DPLLMTGen(
    "/home/qfyan/projects-qfyan/meta-llama/Llama-2-7b-hf",
    save_steps=400,
    logging_steps=5,
    experiment_dir=f"/home/qfyan/scratch/privacy_checkpoints/{TASK_NAME}",
    logging_dir=f'/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}/logs',
    batch_size=16,                 # Batch Size
    # lr_scheduler_type="constant", # Specify the learning rate scheduler 
    stage1_epochs = 5,
    stage2_epochs = 2,
    stage1_lr =1e-4,
    stage2_lr=5e-4,
    loss_alpha=0.65,
    loss_beta=0.1,
    loss_lmbda=1.0,
    per_sample_max_grad_norm=1., 
    target_epsilon=0.5, 
    stage2_batch_size=4,
    efficient_finetuning="lora",
    use_8bit_quantization=True,
)


trainer = great.fit(data, column_names=column_names, resume_from_checkpoint=True)

great.save(f"/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}_final")

samples = great.sample(min(len(data), 1000), k=16, max_length=1000, device="cuda")
samples.to_csv(f"/home/qfyan/cs848-group-project/be_great/experiments/samples/{TASK_NAME}.csv")
