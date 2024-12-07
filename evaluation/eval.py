import EvalutionMetricFunctions as emf
import pandas as pd
from pathlib import Path
from glob import glob
import argparse
SYNTHETIC_TRAIN_PATH = Path('./synthesized')
ORIGINAL_DATA_PATH = Path('./datasets/preprocessed')
NON_RELATIONAL_DATASETS = ['adult', 'california_housing', 'diabetes']
ADULT_CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital-status', 'occupation', 
                             'relationship', 'race', 'sex', 'native-country', 'income'] # only adult dataset has categorical columns
ADULT_TARGET_COLUMN = ['income']
DIABETES_TARGET_COLUMN = ['Outcome']
CALIFORNIA_HOUSING_TARGET_COLUMN = ['median_house_value']
CALIFORNIA_HOUSING_CATEGORICAL_COLUMNS = ['ocean_proximity']

def main(args):
    synthetic_train_paths = [glob(str(SYNTHETIC_TRAIN_PATH / args.model / dataset / f'samples_num_samples=*.csv'))[0] for dataset in NON_RELATIONAL_DATASETS]
    print(f"Using synthetic training data from {synthetic_train_paths}\n")
    synthetic_train_dfs = {dataset: pd.read_csv(path) for dataset, path in zip(NON_RELATIONAL_DATASETS, synthetic_train_paths)}

    private_test_paths = [ORIGINAL_DATA_PATH / dataset / f'{dataset}_test.csv' for dataset in NON_RELATIONAL_DATASETS]
    private_test_dfs = {dataset: pd.read_csv(path) for dataset, path in zip(NON_RELATIONAL_DATASETS, private_test_paths)}

    for dataset in NON_RELATIONAL_DATASETS:
        print(f"Length of synthetic train data: {len(synthetic_train_dfs[dataset])}")
        print(f"Length of private test data: {len(private_test_dfs[dataset])}")

    # For california_housing, we need to replace value '<1H OCEAN' on ocean_proximity with 'lt1H OCEAN' so xgboost can handle it
    synthetic_train_dfs['california_housing'].loc[synthetic_train_dfs['california_housing']['ocean_proximity'] == '<1H OCEAN', 'ocean_proximity'] = 'lt1H OCEAN'
    private_test_dfs['california_housing'].loc[private_test_dfs['california_housing']['ocean_proximity'] == '<1H OCEAN', 'ocean_proximity'] = 'lt1H OCEAN'

    print(f"\nEvaluating adult")
    emf.evaluate_mle(synthetic_train_dfs['adult'], private_test_dfs['adult'], 
                                target_cols_categorical=ADULT_TARGET_COLUMN, 
                                categorical_cols=ADULT_CATEGORICAL_COLUMNS)

    print(f"Evaluating diabetes")
    emf.evaluate_mle(synthetic_train_dfs['diabetes'], private_test_dfs['diabetes'], 
                                target_cols_categorical=DIABETES_TARGET_COLUMN)

    print(f"Evaluating california housing")
    emf.evaluate_mle(synthetic_train_dfs['california_housing'], private_test_dfs['california_housing'], 
                                target_cols_numerical=CALIFORNIA_HOUSING_TARGET_COLUMN, 
                                categorical_cols=CALIFORNIA_HOUSING_CATEGORICAL_COLUMNS);

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['realtabformer'])
    args = parser.parse_args()
    main(args)