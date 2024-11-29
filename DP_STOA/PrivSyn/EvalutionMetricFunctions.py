# Evaluation Metric Functions
import pandas as pd
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  LabelEncoder, MinMaxScaler
from sklearn.svm import SVC

def evaluate_synthetic_data(private_data, synthetic_data, target_col, categorical_cols=None, is_categorical_target=False):
    """
    Evaluate synthetic data quality 
    
    Parameters:
    - private_data: original dataset
    - synthetic_data: generated synthetic dataset
    - target_col: name of the target column
    - categorical_cols: list of categorical column names
    - is_categorical_target: boolean indicating if target is categorical
    
    prints the F1 score, R² score, MSE score, and DM score
    """
    # Prepare data
    private_x = private_data.drop(columns=[target_col])
    private_y = private_data[target_col]
    synthetic_x = synthetic_data.drop(columns=[target_col])
    synthetic_y = synthetic_data[target_col]
    
    # Handle categorical target
    if is_categorical_target:
        le = LabelEncoder()
        private_y = le.fit_transform(private_y)
        synthetic_y = le.transform(synthetic_y)
    
    # One-hot encode categorical features
    if categorical_cols:
        private_x = pd.get_dummies(private_x, columns=categorical_cols)
        synthetic_x = pd.get_dummies(synthetic_x, columns=categorical_cols)
        
        # Align columns
        all_columns = set(private_x.columns) | set(synthetic_x.columns)
        for col in all_columns:
            if col not in private_x.columns:
                private_x[col] = 0
            if col not in synthetic_x.columns:
                synthetic_x[col] = 0
        private_x = private_x[sorted(all_columns)]
        synthetic_x = synthetic_x[sorted(all_columns)]
    
    # Convert numerical columns to float
    numerical_cols = private_x.select_dtypes(include=['int64', 'float64']).columns
    private_x[numerical_cols] = private_x[numerical_cols].astype(float)
    synthetic_x[numerical_cols] = synthetic_x[numerical_cols].astype(float)
    
    # Evaluate classification
    if is_categorical_target:
        private_f1, synthetic_f1 = evaluate_classification(private_x, private_y, synthetic_x, synthetic_y)
        print(f"F1 Score - Private: {private_f1:.4f}, Synthetic: {synthetic_f1:.4f}")
    
    # Evaluate regression
    else:
        metrics = evaluate_regression(private_x, private_y, synthetic_x, synthetic_y)
        print(f"R² Score - Private: {metrics['private_r2']:.4f}, Synthetic: {metrics['synthetic_r2']:.4f}")
        print(f"MSE - Private: {metrics['private_mse']:.4f}, Synthetic: {metrics['synthetic_mse']:.4f}")
    
    # Discriminator Measure
    dm_score = calculate_dm_score(private_x, synthetic_x)
    print(f"DM Score: {dm_score:.4f} (closer to 0 is better)")

def evaluate_classification(private_x, private_y, synthetic_x, synthetic_y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(private_x, private_y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    private_model = SVC()
    private_model.fit(X_train, y_train)
    private_f1 = f1_score(y_test, private_model.predict(X_test), average='macro')
    
    synthetic_model = SVC()
    synthetic_model.fit(synthetic_x, synthetic_y)
    synthetic_f1 = f1_score(y_test, synthetic_model.predict(X_test), average='macro')
    
    return private_f1, synthetic_f1

def evaluate_regression(private_x, private_y, synthetic_x, synthetic_y):
    # Scale features to [0,1] range
    x_scaler = MinMaxScaler()
    private_x_scaled = x_scaler.fit_transform(private_x)
    synthetic_x_scaled = x_scaler.transform(synthetic_x)
    
    # Scale target variable to [0,1] range
    y_scaler = MinMaxScaler()
    private_y_scaled = y_scaler.fit_transform(private_y.values.reshape(-1, 1)).ravel()
    synthetic_y_scaled = y_scaler.transform(synthetic_y.values.reshape(-1, 1)).ravel()
    
    # Split private data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        private_x, private_y, 
        test_size=0.2, random_state=42
    )
    
    # Train private model
    private_model = LinearRegression()
    private_model.fit(X_train, y_train)
    private_pred = private_model.predict(X_test)
    
    # Train synthetic model using all synthetic data
    synthetic_model = LinearRegression()
    synthetic_model.fit(synthetic_x, synthetic_y)
    synthetic_pred = synthetic_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'private_r2': r2_score(y_test, private_pred),
        'synthetic_r2': r2_score(y_test, synthetic_pred),
        'private_mse': mean_squared_error(y_test, private_pred),
        'synthetic_mse': mean_squared_error(y_test, synthetic_pred)
    }
    
    return metrics

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
