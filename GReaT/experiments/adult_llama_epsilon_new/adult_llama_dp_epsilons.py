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

def train_epsilons(episilon):
    task_name = f"adult_llama_dp_new_{episilon}"
    print(f"Start training {task_name}")
    great = DPLLMTGen(
        "meta-llama/Llama-2-7b-hf",
        save_steps=4000,
        logging_steps=5,
        experiment_dir=f"~/privacy_checkpoints/{task_name}",
        logging_dir=f'~/privacy_checkpoints/{task_name}/logs',
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
        target_epsilon=episilon, 
        stage2_batch_size=8,
        efficient_finetuning="lora",
        use_8bit_quantization=True,
    )


    trainer = great.fit(data, column_names=column_names, resume_from_checkpoint=True)

    great.save(f"~/privacy_checkpoints/{task_name}_final")

    print(f"Start sampling 1000")
    samples = great.sample(min(len(data), 1000), k=16, max_length=1000, device="cuda")
    samples.to_csv(f"~/FedFetch/be_great/experiments/samples/adult/{task_name}_1000.csv")

    # print(f"Start sampling {len(data)}")
    # samples = great.sample(len(data), k=16, max_length=1000, device="cuda")
    # samples.to_csv(f"~/FedFetch/be_great/experiments/samples/adult/{TASK_NAME}_{len(data)}.csv")


task_names = [0.5, 1, 3, 5]

for tn in task_names:
    train_epsilons(tn)

