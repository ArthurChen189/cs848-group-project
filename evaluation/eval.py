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

# Replace any special characters in column values that XGBoost can't handle
def replace_special_characters(synthetic_train_dfs: dict, private_test_dfs: dict):
    """Replace special characters in column values that XGBoost can't handle

    Args:
        synthetic_train_dfs (dict): Dictionary of synthetic training dataframes
        private_test_dfs (dict): Dictionary of private testing dataframes
    """    
    for dataset in NON_RELATIONAL_DATASETS:
        for col in synthetic_train_dfs[dataset].select_dtypes(include=['object']).columns:
            # Replace '<' with 'lt'
            synthetic_train_dfs[dataset].loc[synthetic_train_dfs[dataset][col].str.contains('<', na=False), col] = \
                synthetic_train_dfs[dataset].loc[synthetic_train_dfs[dataset][col].str.contains('<', na=False), col].str.replace('<', 'lt')
            private_test_dfs[dataset].loc[private_test_dfs[dataset][col].str.contains('<', na=False), col] = \
                private_test_dfs[dataset].loc[private_test_dfs[dataset][col].str.contains('<', na=False), col].str.replace('<', 'lt')
            
            # Replace '[' and ']' with '(' and ')'
            synthetic_train_dfs[dataset].loc[synthetic_train_dfs[dataset][col].str.contains('[\[\]]', na=False, regex=True), col] = \
                synthetic_train_dfs[dataset].loc[synthetic_train_dfs[dataset][col].str.contains('[\[\]]', na=False, regex=True), col].str.replace('[\[\]]', '()', regex=True)
            private_test_dfs[dataset].loc[private_test_dfs[dataset][col].str.contains('[\[\]]', na=False, regex=True), col] = \
                private_test_dfs[dataset].loc[private_test_dfs[dataset][col].str.contains('[\[\]]', na=False, regex=True), col].str.replace('[\[\]]', '()', regex=True)


def main(args):
    synthetic_train_paths = [glob(str(SYNTHETIC_TRAIN_PATH / args.model / dataset / f'samples_num_samples=*.csv'))[0] for dataset in NON_RELATIONAL_DATASETS]
    print(f"Using synthetic training data from {synthetic_train_paths}\n")
    synthetic_train_dfs = {dataset: pd.read_csv(path) for dataset, path in zip(NON_RELATIONAL_DATASETS, synthetic_train_paths)}

    private_test_paths = [ORIGINAL_DATA_PATH / dataset / f'{dataset}_test.csv' for dataset in NON_RELATIONAL_DATASETS]
    private_test_dfs = {dataset: pd.read_csv(path) for dataset, path in zip(NON_RELATIONAL_DATASETS, private_test_paths)}

    for dataset in NON_RELATIONAL_DATASETS:
        print(f"Length of synthetic train data: {len(synthetic_train_dfs[dataset])}")
        print(f"Length of private test data: {len(private_test_dfs[dataset])}")

    replace_special_characters(synthetic_train_dfs, private_test_dfs)

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

    print(f"\nEvaluating dm score")
    print(f"Evaluating adult")
    emf.calculate_dm_score(private_test_dfs['adult'], synthetic_train_dfs['adult'], categorical_cols=ADULT_CATEGORICAL_COLUMNS)

    print(f"Evaluating diabetes")
    emf.calculate_dm_score(private_test_dfs['diabetes'], synthetic_train_dfs['diabetes'])

    print(f"Evaluating california housing")
    emf.calculate_dm_score(private_test_dfs['california_housing'], synthetic_train_dfs['california_housing'], categorical_cols=CALIFORNIA_HOUSING_CATEGORICAL_COLUMNS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['realtabformer'])
    args = parser.parse_args()
    main(args)