# Evaluation Metric Functions
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

def evaluate_mle(synthetic_train_data: pd.DataFrame, private_test_data: pd.DataFrame, categorical_cols: list = [], 
                 target_cols_categorical: list = [], target_cols_numerical: list = []) -> float:
    """General machine learning efficacy (MLE) score calculation for any dataset

    Args:
        synthetic_train_data (pd.DataFrame): Synthetic training data
        private_test_data (pd.DataFrame): Private testing data
        categorical_cols (list, optional): List of categorical columns. Defaults to [].
        target_cols_categorical (list, optional): List of categorical target columns. Defaults to [].
        target_cols_numerical (list, optional): List of numerical target columns. Defaults to [].
    Returns:
        float: MLE score
    """
    # Prepare features
    synthetic_data = synthetic_train_data.copy()
    private_data = private_test_data.copy()
    
    f1_scores = []
    r2_scores = []
    
    # Classification tasks for categorical columns
    if target_cols_categorical:
        for target_col in target_cols_categorical:
            # Prepare features
            remaining_categorical_cols = categorical_cols.copy()
            if target_col in categorical_cols:
                remaining_categorical_cols.remove(target_col)

            train_features = synthetic_data.drop(columns=[target_col])
            test_features = private_data.drop(columns=[target_col])
            
            # One-hot encode remaining categorical features if there are any
            if remaining_categorical_cols:
                train_features = pd.get_dummies(train_features, columns=remaining_categorical_cols)
                test_features = pd.get_dummies(test_features, columns=remaining_categorical_cols)
                
                # Ensure train and test have same columns
                missing_cols = set(train_features.columns) - set(test_features.columns)
                for col in missing_cols:
                    test_features[col] = 0
                test_features = test_features[train_features.columns]
            
            # Encode target
            le = LabelEncoder()
            y_train = le.fit_transform(synthetic_data[target_col])
            
            # Drop test rows with unseen labels
            seen_labels_mask = private_data[target_col].isin(le.classes_)
            test_features = test_features[seen_labels_mask]
            y_test = le.transform(private_data[target_col][seen_labels_mask])
            
            # Train classifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(train_features, y_train)
            y_pred = model.predict(test_features)
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    
    if target_cols_numerical:
        for target_col in target_cols_numerical:    
            # One-hot encode categorical features
            if categorical_cols:
                train_features = pd.get_dummies(synthetic_data, columns=categorical_cols)
                test_features = pd.get_dummies(private_data, columns=categorical_cols)
                # Ensure train and test have same columns
                missing_cols = set(train_features.columns) - set(test_features.columns)
                for col in missing_cols:
                    test_features[col] = 0
                test_features = test_features[train_features.columns]
            else:
                train_features = synthetic_data
                test_features = private_data
            
            y_train = synthetic_data[target_col].values
            y_test = private_data[target_col].values
            
            # Train regressor
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(train_features, y_train)
            y_pred = model.predict(test_features)
            r2_scores.append(r2_score(y_test, y_pred))
    
    # Calculate final scores
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    avg_r2 = np.mean(r2_scores) if r2_scores else 0
    
    print(f"Average MLE (F1) Score ({len(f1_scores)} tasks): {avg_f1:.4f}") if f1_scores else None
    print(f"Average MLE (R²) Score ({len(r2_scores)} tasks): {avg_r2:.4f}") if r2_scores else None

    return avg_f1, avg_r2


def calculate_dm_score(private_x: pd.DataFrame, synthetic_x: pd.DataFrame, categorical_cols: list = []) -> float:
    """Calculate the discriminative measure score between private and synthetic data i.e., how much synthetic data is similar to private data

    Args:
        private_x (pd.DataFrame): Private data
        synthetic_x (pd.DataFrame): Synthetic data
        categorical_cols (list, optional): List of categorical column names. Defaults to [].

    Returns:
        float: Discriminative measure score
    """    
    # Prepare data
    private_x['is_synthetic'] = 0
    synthetic_x['is_synthetic'] = 1
    
    # One-hot encode categorical features
    if categorical_cols:
        private_features = pd.get_dummies(private_x, columns=categorical_cols)
        synthetic_features = pd.get_dummies(synthetic_x, columns=categorical_cols)
        # Ensure both datasets have same columns
        missing_cols = set(synthetic_features.columns) - set(private_features.columns)
        for col in missing_cols:
            private_features[col] = 0
        private_features = private_features[synthetic_features.columns]
    else:
        private_features = private_x
        synthetic_features = synthetic_x
        
    combined_data = pd.concat([private_features, synthetic_features])
    
    X = combined_data.drop('is_synthetic', axis=1)
    y = combined_data['is_synthetic']
    
    # Split and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"DM Score: {accuracy:.4f}")
    return accuracy
