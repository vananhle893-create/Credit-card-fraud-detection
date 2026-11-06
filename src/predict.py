import os
import sys
import io

# Cấu hình encoding utf-8 cho console trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pickle
import pandas as pd
import numpy as np

def load_prediction_assets(models_dir="models"):
    """
    Loads the trained best model and scaling assets.
    """
    scalers_path = os.path.join(models_dir, 'scalers.pkl')
    model_path = os.path.join(models_dir, 'best_fraud_detector.pkl')
    
    if not os.path.exists(scalers_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing required model assets in {models_dir}. Please run the training script first."
        )
        
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return model, scalers

def predict_transactions(df, models_dir="models"):
    """
    Predicts a DataFrame of transactions.
    Expects columns: ['Time', 'V1', 'V2', ..., 'V28', 'Amount']
    """
    model, scalers = load_prediction_assets(models_dir)
    
    # 1. Scale 'Amount' and 'Time' using the saved scalers
    df_processed = df.copy()
    df_processed['Amount'] = scalers['amount_scaler'].transform(df[['Amount']])
    df_processed['Time'] = scalers['time_scaler'].transform(df[['Time']])
    
    # Ensure correct column ordering matching the training set
    feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_processed = df_processed[feature_cols]
    
    # 2. Predict classes and probabilities
    predictions = model.predict(df_processed)
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_processed)[:, 1]
    else:
        # Fallback for models without predict_proba (decision function)
        decision_scores = model.decision_function(df_processed)
        # Apply sigmoid to convert to pseudo-probabilities
        probabilities = 1 / (1 + np.exp(-decision_scores))
        
    results = []
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "GIẢ MẠO (FRAUD)" if pred == 1 else "HỢP LỆ (VALID)"
        results.append({
            'transaction_index': idx,
            'prediction': int(pred),
            'status': status,
            'fraud_probability': float(prob),
            'amount': float(df.iloc[idx]['Amount']),
            'time': float(df.iloc[idx]['Time'])
        })
        
    return results

if __name__ == "__main__":
    print("[INFO] Running a prediction test with mock transactions...")
    
    # Let's create two mock transactions:
    # Transaction 1: Normal transaction (Time=100, low Amount, normal V values)
    # Transaction 2: Suspect transaction (High amount, highly negative V17/V14/V12 which are negatively correlated with Class)
    
    # Load headers V1 to V28
    v_cols = {f'V{i}': 0.0 for i in range(1, 29)}
    
    # Legitimate Transaction
    tx_legit = {'Time': 120.0, 'Amount': 45.5}
    tx_legit.update(v_cols) # normal V values
    
    # Fraudulent Transaction (outlier-like V values based on top correlation features V17, V14, V12)
    tx_fraud = {'Time': 54000.0, 'Amount': 999.9}
    tx_fraud.update(v_cols)
    tx_fraud['V17'] = -8.5  # Very negative V17 is strongly associated with fraud
    tx_fraud['V14'] = -9.0  # Very negative V14 is strongly associated with fraud
    tx_fraud['V12'] = -7.2  # Very negative V12 is strongly associated with fraud
    tx_fraud['V11'] = 6.0   # Positive V11 is associated with fraud
    tx_fraud['V4'] = 5.5    # Positive V4 is associated with fraud
    
    test_df = pd.DataFrame([tx_legit, tx_fraud])
    
    try:
        results = predict_transactions(test_df)
        print("\n--- KẾT QUẢ DỰ ĐOÁN GIAO DỊCH THỬ NGHIỆM ---")
        for res in results:
            print(f"Giao dịch #{res['transaction_index'] + 1}:")
            print(f"  - Số tiền: ${res['amount']:.2f}")
            print(f"  - Thời gian: {res['time']}s")
            print(f"  - Trạng thái dự đoán: {res['status']}")
            print(f"  - Xác suất giả mạo: {res['fraud_probability'] * 100:.4f}%")
            print("-" * 45)
    except Exception as e:
        print(f"[ERROR] Could not complete test prediction: {e}")
        print("[TIP] Make sure the models have been successfully trained first!")
