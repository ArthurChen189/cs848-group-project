import pandas as pd
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path("./datasets/raw")
OUTPUT_DIR = Path("./datasets/processed")

adult_raw = pd.read_csv(os.path.join(DATA_DIR, "adult", "uciml_adult.csv"))
california_housing_raw = pd.read_csv(os.path.join(DATA_DIR, "california_housing", "housing.csv"))
diabetes_raw = pd.read_csv(os.path.join(DATA_DIR, "diabetes", "diabetes.csv"))

# first we shuffle the data
seed = 42
adult_raw = adult_raw.sample(frac=1, random_state=seed).reset_index(drop=True)
california_housing_raw = california_housing_raw.sample(frac=1, random_state=seed).reset_index(drop=True)
diabetes_raw = diabetes_raw.sample(frac=1, random_state=seed).reset_index(drop=True)

adult_train, adult_val, adult_test = np.split(adult_raw, [int(0.6*len(adult_raw)), int(0.9*len(adult_raw))])
california_housing_train, california_housing_val, california_housing_test = np.split(california_housing_raw, [int(0.6*len(california_housing_raw)), int(0.9*len(california_housing_raw))])
diabetes_train, diabetes_val, diabetes_test = np.split(diabetes_raw, [int(0.6*len(diabetes_raw)), int(0.9*len(diabetes_raw))])

# export the data
# create output directories if they don't exist
os.makedirs(os.path.join(OUTPUT_DIR, "adult"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "california_housing"), exist_ok=True) 
os.makedirs(os.path.join(OUTPUT_DIR, "diabetes"), exist_ok=True)

# export the data
adult_train.to_csv(os.path.join(OUTPUT_DIR, "adult", "adult_train.csv"), index=False)
adult_val.to_csv(os.path.join(OUTPUT_DIR, "adult", "adult_val.csv"), index=False)
adult_test.to_csv(os.path.join(OUTPUT_DIR, "adult", "adult_test.csv"), index=False)
california_housing_train.to_csv(os.path.join(OUTPUT_DIR, "california_housing", "california_housing_train.csv"), index=False)
california_housing_val.to_csv(os.path.join(OUTPUT_DIR, "california_housing", "california_housing_val.csv"), index=False)
california_housing_test.to_csv(os.path.join(OUTPUT_DIR, "california_housing", "california_housing_test.csv"), index=False)
diabetes_train.to_csv(os.path.join(OUTPUT_DIR, "diabetes", "diabetes_train.csv"), index=False)
diabetes_val.to_csv(os.path.join(OUTPUT_DIR, "diabetes", "diabetes_val.csv"), index=False)
diabetes_test.to_csv(os.path.join(OUTPUT_DIR, "diabetes", "diabetes_test.csv"), index=False)
