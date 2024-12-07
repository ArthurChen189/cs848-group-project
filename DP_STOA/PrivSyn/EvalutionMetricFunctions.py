# Evaluation Metric Functions
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import  LabelEncoder
import numpy as np

def evaluate_mle(data, categorical_cols=None):
    """
    General MLE score calculation for any dataset
    """
    if categorical_cols is None:
        categorical_cols = []
    
    # Prepare features
    X = data.copy()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    f1_scores = []
    r2_scores = []
    
    # Classification tasks for categorical columns
    for target_col in categorical_cols:
        # Prepare features
        features = X.drop(columns=[target_col])
        
        # One-hot encode remaining categorical features
        cat_cols = [col for col in categorical_cols if col != target_col]
        if cat_cols:
            features = pd.get_dummies(features, columns=cat_cols)
        
        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(X[target_col])
        
        # Split data
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train classifier
        model = RandomForestClassifier(
            n_estimators=100,  
            max_depth=None,    
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    
    # Regression tasks for numerical columns
    for target_col in numerical_cols:
        # Prepare features
        features = X.drop(columns=[target_col])
        
        # One-hot encode categorical features
        if categorical_cols:
            features = pd.get_dummies(features, columns=categorical_cols)
        
        y = X[target_col].values
        
        # Split data
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train regressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
    
    # Calculate final scores
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    avg_r2 = np.mean(r2_scores) if r2_scores else 0
    mle_score = (avg_f1 + avg_r2) / 2
    
    print(f"Average F1 Score ({len(f1_scores)} tasks): {avg_f1:.4f}")
    print(f"Average R² Score ({len(r2_scores)} tasks): {avg_r2:.4f}")
    print(f"MLE Score: {mle_score:.4f}")
    
    return mle_score


def calculate_dm_score(private_x, synthetic_x):
    # Prepare data
    private_x['is_synthetic'] = 0
    synthetic_x['is_synthetic'] = 1
    combined_data = pd.concat([private_x, synthetic_x])
    
    X = combined_data.drop('is_synthetic', axis=1)
    y = combined_data['is_synthetic']
    
    # Split and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    return abs(0.5 - accuracy)
