import os
import sys
import time
import io

# Cấu hình encoding utf-8 cho console trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def main():
    print("=" * 70)
    print("  QUY TRÌNH TỰ ĐỘNG PHÂN TÍCH & DỰ ĐOÁN GIAO DỊCH THẺ TÍN DỤNG GIẢ MẠO  ")
    print("=" * 70)
    start_time = time.time()
    
    # 1. Download Dataset
    print("\n--- BƯỚC 1: TẢI BỘ DỮ LIỆU TỪ HUGGING FACE ---")
    from src.download_data import download_dataset
    DATA_URL = "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv"
    DATA_PATH = os.path.join("data", "creditcard.csv")
    
    success = download_dataset(DATA_URL, DATA_PATH)
    if not success:
        print("[CRITICAL] Không thể tải bộ dữ liệu. Kết thúc quy trình.")
        sys.exit(1)
        
    # 2. Exploratory Data Analysis
    print("\n--- BƯỚC 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) ---")
    from src.eda import run_eda
    PLOTS_DIR = "plots"
    success = run_eda(DATA_PATH, PLOTS_DIR)
    if not success:
        print("[WARNING] Quá trình phân tích dữ liệu gặp lỗi, vẫn tiếp tục tiền xử lý...")
        
    # 3. Preprocessing and Splitting
    print("\n--- BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU & CHUẨN HÓA ---")
    from src.preprocess import preprocess_dataset
    MODELS_DIR = "models"
    try:
        X_train, X_test, y_train, y_test = preprocess_dataset(DATA_PATH, MODELS_DIR)
    except Exception as e:
        print(f"[CRITICAL] Lỗi tiền xử lý dữ liệu: {e}")
        sys.exit(1)
        
    # 4. Training and Evaluation
    print("\n--- BƯỚC 4: HUẤN LUYỆN & SO SÁNH CÁC MÔ HÌNH ---")
    from src.train import train_and_evaluate_all
    try:
        best_model_name, metrics_df = train_and_evaluate_all(
            X_train, X_test, y_train, y_test, MODELS_DIR, PLOTS_DIR
        )
        print(f"\n[INFO] Huấn luyện hoàn tất! Mô hình tốt nhất: {best_model_name}")
    except Exception as e:
        print(f"[CRITICAL] Lỗi trong quá trình huấn luyện: {e}")
        sys.exit(1)
        
    # 5. Pipeline Test Prediction
    print("\n--- BƯỚC 5: CHẠY THỬ NGHIỆM DỰ ĐOÁN GIAO DỊCH GIẢ ĐỊNH ---")
    import pandas as pd
    from src.predict import predict_transactions
    
    # Create two transactions: one normal and one with suspicious PCA features
    v_cols = {f'V{i}': 0.0 for i in range(1, 29)}
    
    tx_legit = {'Time': 100.0, 'Amount': 20.0}
    tx_legit.update(v_cols)
    
    tx_fraud = {'Time': 100000.0, 'Amount': 1500.0}
    tx_fraud.update(v_cols)
    tx_fraud['V17'] = -9.5
    tx_fraud['V14'] = -10.2
    tx_fraud['V12'] = -8.0
    tx_fraud['V11'] = 7.0
    tx_fraud['V4'] = 6.0
    
    test_df = pd.DataFrame([tx_legit, tx_fraud])
    
    try:
        results = predict_transactions(test_df, MODELS_DIR)
        print("\n=== KẾT QUẢ KIỂM THỬ PIPELINE DỰ ĐOÁN VỚI MẪU MỚI ===")
        for res in results:
            print(f"Giao dịch #{res['transaction_index'] + 1}:")
            print(f"  - Số tiền: ${res['amount']:.2f}")
            print(f"  - Thời gian: {res['time']}s")
            print(f"  - Trạng thái dự đoán: {res['status']}")
            print(f"  - Xác suất giả mạo: {res['fraud_probability'] * 100:.4f}%")
            print("-" * 55)
    except Exception as e:
        print(f"[WARNING] Lỗi chạy thử nghiệm dự đoán: {e}")
        
    duration = time.time() - start_time
    print("=" * 70)
    print(f"  QUY TRÌNH HOÀN THÀNH THÀNH CÔNG TRONG {duration:.2f} GIÂY!  ")
    print("=" * 70)

if __name__ == "__main__":
    main()
