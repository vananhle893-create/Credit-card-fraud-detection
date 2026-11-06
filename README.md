# Dự đoán giao dịch thẻ tín dụng giả mạo (Credit Card Fraud Detection)

Dự án học máy phát hiện giao dịch thẻ tín dụng giả mạo trên dữ liệu mất cân bằng cực đoan
(492 giao dịch giả mạo / 284,807 giao dịch = 0.173%). 

## Quy trình
1. Tải dữ liệu `data/creditcard.csv` (`src/download_data.py`)
2. Phân tích khám phá dữ liệu – EDA (`src/eda.py`) → `plots/`
3. Chia train/test 80/20, phân tầng, `random_state=42` (`src/preprocess.py`)
4. Chuẩn hóa `Time`, `Amount` bằng RobustScaler (chỉ fit trên train) → `models/scalers.pkl`
5. Huấn luyện 4 mô hình: Logistic Regression, Random Forest, XGBoost, SMOTE + Random Forest (`src/train.py`)
6. Đánh giá: Precision, Recall, F1, ROC-AUC, AUPRC, ma trận nhầm lẫn
7. Chọn mô hình theo AUPRC → `models/best_fraud_detector.pkl`
8. Triển khai ứng dụng Streamlit + Docker (`app.py`, `src/predict.py`)

## Kết quả trên tập test (56,962 giao dịch, 98 giả mạo)
Chạy lại trên Windows 11, Python 3.15.0, scikit-learn 1.9.1, xgboost 3.4.1, imbalanced-learn 0.14.2,
pandas 3.1.0, streamlit 1.66.1. Kết quả, nhất là của Random Forest, có thể khác nếu chạy ở môi trường khác.

| Mô hình | Precision | Recall | F1 | AUPRC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 6.09% | 91.84% | 0.1142 | 0.7175 | 0.9720 |
| Random Forest | 71.68% | 82.65% | 0.7678 | 0.8236 | 0.9790 |
| **XGBoost** | **87.23%** | 83.67% | **0.8542** | **0.8736** | 0.9648 |
| SMOTE + Random Forest | 50.60% | 85.71% | 0.6364 | 0.8250 | 0.9791 |

## Cách chạy
```bash
pip install -r requirements.txt
python run_pipeline.py          # chạy toàn bộ quy trình, tạo models/ và plots/
streamlit run app.py            # mở ứng dụng tại http://localhost:8501
```

### Chạy bằng Docker
```bash
docker build -t fraud-detector .
docker run -p 8501:8501 fraud-detector
```
Ảnh Docker không chứa `data/creditcard.csv` (đã loại để nhẹ) nhưng giữ `models/` và `plots/`
để ứng dụng hiển thị đầy đủ ba thẻ.

## Phân tích bổ sung (tùy chọn)
`src/extra_analysis.py` chạy thêm 4 phân tích, không thay đổi quy trình chính:
kiểm tra chéo 5-fold trên tập train, chọn ngưỡng phân loại tối ưu F1 (bằng dự đoán out-of-fold, không dùng test),
so sánh khi bỏ dòng trùng lặp, và feature importance của XGBoost.
```bash
python src/extra_analysis.py            # đầy đủ, khoảng 5-15 phút
python src/extra_analysis.py --quick    # chạy thử nhanh, số liệu KHÔNG dùng để báo cáo
```
Kết quả nằm trong `results/extra_summary.md` và `plots/` (cv_results.png, threshold_tuning.png, xgb_feature_importance.png).
Kết quả của lần chạy đầy đủ (khoảng 9 phút) đã được lưu sẵn trong `results/extra_summary.md`.

## Cấu trúc thư mục
```
app.py              Ứng dụng Streamlit (3 thẻ: Kiểm tra giao dịch / EDA / Hiệu năng mô hình)
run_pipeline.py     Chạy toàn bộ quy trình + kiểm thử dự đoán 2 giao dịch giả định
src/                download_data, eda, preprocess, train, predict, extra_analysis
models/             Mô hình và bộ chuẩn hóa đã lưu (.pkl)
plots/              Biểu đồ EDA và so sánh mô hình
results/            Kết quả phân tích bổ sung (extra_summary.md)
data/               creditcard.csv (không đưa lên GitHub vì nặng 143 MB; run_pipeline.py tự tải khi chạy)
```

## Hạn chế đã biết
Chọn mô hình trên chính tập test; chỉ một lần chia dữ liệu; chưa tinh chỉnh siêu tham số và ngưỡng;
còn 1,081 dòng trùng lặp; ứng dụng demo chỉ chỉnh được 5 thuộc tính (V17, V14, V12, V11, V4).
