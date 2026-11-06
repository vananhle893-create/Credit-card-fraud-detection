import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def preprocess_dataset(data_path, models_dir=None):
    """
    Loads dataset, splits into Train/Test with stratification, scales Time and Amount,
    and returns scaled datasets along with the scalers.
    """
    print("[INFO] Starting data preprocessing...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load dataset
    df = pd.read_csv(data_path)
    
    # 2. Separate X and y
    X = df.drop(columns=['Class'])
    y = df['Class']

    # 3. Stratified Split (80% Train, 20% Test)
    # Using stratify=y is critical because Class is highly imbalanced
    print("[INFO] Splitting dataset into train and test sets (80/20 stratified split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Train set shape: {X_train.shape[0]} samples, {y_train.sum()} positive (fraud) cases")
    print(f"[INFO] Test set shape: {X_test.shape[0]} samples, {y_test.sum()} positive (fraud) cases")

    # 4. Scaling Features
    # 'Amount' and 'Time' have very different scales and outliers.
    # RobustScaler is less sensitive to outliers, which is perfect for transaction amounts.
    print("[INFO] Scaling 'Amount' and 'Time' features using RobustScaler...")
    amount_scaler = RobustScaler()
    time_scaler = RobustScaler()

    # Fit and transform training set (avoiding data leakage!)
    X_train_scaled = X_train.copy()
    X_train_scaled['Amount'] = amount_scaler.fit_transform(X_train[['Amount']])
    X_train_scaled['Time'] = time_scaler.fit_transform(X_train[['Time']])

    # Transform test set (using scalers fit on train set only!)
    X_test_scaled = X_test.copy()
    X_test_scaled['Amount'] = amount_scaler.transform(X_test[['Amount']])
    X_test_scaled['Time'] = time_scaler.transform(X_test[['Time']])

    # Save scalers for prediction pipeline
    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
        scalers = {
            'amount_scaler': amount_scaler,
            'time_scaler': time_scaler
        }
        scalers_path = os.path.join(models_dir, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"[INFO] Saved scalers to {scalers_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "creditcard.csv")
    MODELS_DIR = "models"
    preprocess_dataset(DATA_PATH, MODELS_DIR)
