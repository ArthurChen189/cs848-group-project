TASK_NAME = "iris_llama"
import os
import sys

sys.path.insert(0, "/home/qfyan/cs848-group-project/be_great")

from be_great import GReaT
import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)
import pandas 

# from datasets import load_dataset
data = pandas.read_csv("/home/qfyan/projects-qfyan/privacy_data/Iris.csv")
data = data.iloc[:, 1:]
# print(dataset)
# data = datasets.load_iris(as_frame=True).frame
print(data.head())

column_names = ["sepal length", "sepal width", "petal length", "petal width", "target"]
data.columns = column_names

great = GReaT(
    "/home/qfyan/projects-qfyan/meta-llama/Llama-2-7b-hf",
    epochs=2,
    save_steps=400,
    logging_steps=5,
    experiment_dir=f"/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}",
    logging_dir=f'/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}/logs',
    batch_size=16,
    learning_rate=5e-4, 
    efficient_finetuning="lora",
    use_8bit_quantization=True,
)

trainer = great.fit(data, column_names=column_names)

great.save(f"/home/qfyan/projects-qfyan/privacy_checkpoints/{TASK_NAME}_final")

# Generate synthetic data
samples = great.sample(len(data), k=8)
samples.to_csv(f"/home/qfyan/cs848-group-project/be_great/experiments/samples/{TASK_NAME}.csv")
