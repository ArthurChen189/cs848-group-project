import os
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)

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
    "/home/qfyan/projects-qfyan/openai-community/gpt2",
    epochs=50,
    save_steps=100,
    logging_steps=5,
    experiment_dir="/home/qfyan/projects-qfyan/privacy_checkpoints/trainer_iris",
    # lr_scheduler_type="constant", learning_rate=5e-5
)

trainer = great.fit(data, column_names=column_names)

great.save("iris_base")
