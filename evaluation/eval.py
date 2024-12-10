import eval_utils as emf
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
EPSILON_VALUES = [0.5, 1, 3, 5, 10]

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
    # for epsilon in EPSILON_VALUES:
    #     print(f"\n=== Evaluating epsilon={epsilon} ===")

    #     synthetic_train_paths = []
    #     for dataset in NON_RELATIONAL_DATASETS:
    #         files = list(glob(str(SYNTHETIC_TRAIN_PATH / args.model / dataset / f'train_e{epsilon}.csv')))
    #         if not files:
    #             raise FileNotFoundError(f"No synthetic data found for ./{args.model}/{dataset}")
    #         synthetic_train_paths.append(files[0])
    
    #     print(f"Using synthetic training data from {synthetic_train_paths}\n")
    #     synthetic_train_dfs = {dataset: pd.read_csv(path) for dataset, path in zip(NON_RELATIONAL_DATASETS, synthetic_train_paths)}

    #     private_test_paths = [ORIGINAL_DATA_PATH / dataset / f'{dataset}_test.csv' for dataset in NON_RELATIONAL_DATASETS]
    #     private_test_dfs = {dataset: pd.read_csv(path) for dataset, path in zip(NON_RELATIONAL_DATASETS, private_test_paths)}

    #     for dataset in NON_RELATIONAL_DATASETS:
    #         print(f"Length of synthetic train data: {len(synthetic_train_dfs[dataset])}")
    #         print(f"Length of private test data: {len(private_test_dfs[dataset])}")

    #     replace_special_characters(synthetic_train_dfs, private_test_dfs)

    #     print(f"\nEvaluating adult")
    #     emf.evaluate_mle(synthetic_train_dfs['adult'], private_test_dfs['adult'], 
    #                                 target_cols_categorical=ADULT_TARGET_COLUMN, 
    #                                 categorical_cols=ADULT_CATEGORICAL_COLUMNS)

    #     print(f"Evaluating diabetes")
    #     emf.evaluate_mle(synthetic_train_dfs['diabetes'], private_test_dfs['diabetes'], 
    #                                 target_cols_categorical=DIABETES_TARGET_COLUMN)

    #     print(f"Evaluating california housing")
    #     emf.evaluate_mle(synthetic_train_dfs['california_housing'], private_test_dfs['california_housing'], 
    #                                 target_cols_numerical=CALIFORNIA_HOUSING_TARGET_COLUMN, 
    #                                 categorical_cols=CALIFORNIA_HOUSING_CATEGORICAL_COLUMNS);

    #     print(f"\nEvaluating dm score")
    #     print(f"Evaluating adult")
    #     emf.calculate_dm_score(private_test_dfs['adult'], synthetic_train_dfs['adult'], categorical_cols=ADULT_CATEGORICAL_COLUMNS)

    #     print(f"Evaluating diabetes")
    #     emf.calculate_dm_score(private_test_dfs['diabetes'], synthetic_train_dfs['diabetes'])

    #     print(f"Evaluating california housing")
    #     emf.calculate_dm_score(private_test_dfs['california_housing'], synthetic_train_dfs['california_housing'], categorical_cols=CALIFORNIA_HOUSING_CATEGORICAL_COLUMNS)

    # rossmann
    rossmann_synthetic_child_train_path = glob(str(SYNTHETIC_TRAIN_PATH / args.model / 'rossmann' / f'child_samples_num_samples=*.csv'))[0]
    rossmann_synthetic_parent_train_path = glob(str(SYNTHETIC_TRAIN_PATH / args.model / 'rossmann' / f'parent_samples_num_samples=*.csv'))[0]
    rossmann_synthetic_child_train_df = pd.read_csv(rossmann_synthetic_child_train_path)
    rossmann_synthetic_parent_train_df = pd.read_csv(rossmann_synthetic_parent_train_path)
    # join synthetic child and parent train dataframes on 'Store' column
    rossmann_synthetic_train_df = pd.merge(rossmann_synthetic_child_train_df, rossmann_synthetic_parent_train_df, on='Store')

    rossmann_private_child_train_path = glob(str(ORIGINAL_DATA_PATH / 'rossmann' / f'rossmann_train.csv'))[0]
    rossmann_private_parent_train_path = glob(str(ORIGINAL_DATA_PATH / 'rossmann' / f'rossmann_parent.csv'))[0]
    rossmann_private_child_train_df = pd.read_csv(rossmann_private_child_train_path)
    rossmann_private_parent_train_df = pd.read_csv(rossmann_private_parent_train_path)
    # join private child and parent train dataframes on 'Store' column
    rossmann_private_train_df = pd.merge(rossmann_private_child_train_df, rossmann_private_parent_train_df, on='Store')


    # child data
    rossmann_synthetic_child_train_df.drop(columns=["Store", "Date"], inplace=True)
    rossmann_synthetic_child_train_df.reset_index(drop=True, inplace=True)
    rossmann_private_child_train_df.drop(columns=["Store", "Date"], inplace=True)
    rossmann_private_child_train_df.reset_index(drop=True, inplace=True) # reset index the length of synthetic train data

    # parent data
    rossmann_private_parent_train_df.drop(columns=["Store", "Promo2", "PromoInterval", "Promo2SinceWeek", "Promo2SinceYear"], inplace=True)
    # drop na values
    rossmann_private_parent_train_df.dropna(inplace=True)
    rossmann_private_parent_train_df.reset_index(drop=True, inplace=True)
    rossmann_synthetic_parent_train_df.drop(columns=["Store", "Promo2", "PromoInterval", "Promo2SinceWeek", "Promo2SinceYear", "Unnamed: 0"], inplace=True)
    rossmann_synthetic_parent_train_df.reset_index(drop=True, inplace=True)

    # merged data
    rossmann_synthetic_train_df = rossmann_synthetic_train_df.drop(columns=["Store", "Date", "Promo2", "PromoInterval", 
                                                                            "Promo2SinceWeek", "Promo2SinceYear", "Unnamed: 0"])
    rossmann_private_train_df = rossmann_private_train_df.drop(columns=["Store", "Date", "Promo2", "PromoInterval", 
                                                                        "Promo2SinceWeek", "Promo2SinceYear"])

    # drop na values
    rossmann_private_train_df.dropna(inplace=True)
    rossmann_private_train_df.reset_index(drop=True, inplace=True)

    print(f"Evaluating rossmann parent")
    emf.calculate_ld_score(rossmann_private_parent_train_df, rossmann_synthetic_parent_train_df, 
                    categorical_cols=["StoreType", "Assortment"]
                    );
    
    print(f"Evaluating rossmann child")
    emf.calculate_ld_score(rossmann_private_child_train_df, rossmann_synthetic_child_train_df, 
                    categorical_cols=["Open", "Promo", "StateHoliday", "SchoolHoliday"]
                    );

    print(f"Evaluating rossmann merged")
    emf.calculate_ld_score(rossmann_private_train_df, rossmann_synthetic_train_df, 
                    categorical_cols=["Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment"]
                    );

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['realtabformer', 'dp-stoa', 'original'])
    args = parser.parse_args()
    main(args)