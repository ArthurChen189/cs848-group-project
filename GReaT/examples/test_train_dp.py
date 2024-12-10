import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from be_great.dp.dp import DPLLMTGen
from sklearn import datasets

import logging
from utils import set_logging_level

logger = set_logging_level(logging.INFO)

data = datasets.load_iris(as_frame=True).frame
print(data.head())

column_names = ["sepal length", "sepal width", "petal length", "petal width", "target"]
data.columns = column_names

great = DPLLMTGen(
    "distilgpt2",
    epochs=50,
    save_steps=100,
    logging_steps=5,
    experiment_dir="trainer_iris",
    
)

trainer = great.fit(data, column_names=column_names)

great.save("iris_dp")
