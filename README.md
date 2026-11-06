# DỰ ĐOÁN GIAO DỊCH THẺ TÍN DỤNG GIẢ MẠO

## GIỚI THIỆU

**Credit Card Fraud Detection** là hệ thống học máy phát hiện giao dịch thẻ tín dụng giả mạo trên bộ dữ liệu mất cân bằng cực đoan: chỉ **492 giao dịch giả mạo trong 284.807 giao dịch (0,173%)**.

Dự án thực hiện trọn vẹn một quy trình khoa học dữ liệu, từ tải dữ liệu, phân tích khám phá (EDA), tiền xử lý, huấn luyện và so sánh 4 mô hình, đến triển khai ứng dụng web bằng **Streamlit** và đóng gói bằng **Docker**.

### Tính năng chính

- Tự động tải dữ liệu, chạy toàn bộ quy trình chỉ với một lệnh (`run_pipeline.py`).
- So sánh 4 mô hình: Logistic Regression, Random Forest, XGBoost, SMOTE + Random Forest.
- Đánh giá bằng các chỉ số phù hợp với dữ liệu mất cân bằng: Precision, Recall, F1, AUPRC, ROC-AUC, ma trận nhầm lẫn.
- Ứng dụng web gồm 3 thẻ: **Kiểm tra giao dịch**, **Phân tích dữ liệu (EDA)**, **Hiệu năng mô hình**.
- Phân tích bổ sung: kiểm tra chéo 5-fold, chọn ngưỡng phân loại tối ưu, so sánh khi bỏ dòng trùng lặp, độ quan trọng đặc trưng của XGBoost.

---

## QUY TRÌNH THỰC HIỆN

1. Tải dữ liệu `data/creditcard.csv` (`src/download_data.py`).
2. Phân tích khám phá dữ liệu, xuất biểu đồ vào `plots/` (`src/eda.py`).
3. Chia train/test theo tỷ lệ 80/20, phân tầng, `random_state=42` (`src/preprocess.py`).
4. Chuẩn hóa `Time` và `Amount` bằng RobustScaler, chỉ fit trên tập train, lưu vào `models/scalers.pkl`.
5. Huấn luyện và đánh giá 4 mô hình (`src/train.py`).
6. Chọn mô hình tốt nhất theo AUPRC, lưu vào `models/best_fraud_detector.pkl`.
7. Triển khai ứng dụng Streamlit và Docker (`app.py`, `src/predict.py`).

---

## KẾT QUẢ

Đánh giá trên tập test gồm 56.962 giao dịch (98 giao dịch giả mạo):

| Mô hình | Precision | Recall | F1 | AUPRC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 6,09% | 91,84% | 0,1142 | 0,7175 | 0,9720 |
| Random Forest | 71,68% | 82,65% | 0,7678 | 0,8236 | 0,9790 |
| **XGBoost** | **87,23%** | 83,67% | **0,8542** | **0,8736** | 0,9648 |
| SMOTE + Random Forest | 50,60% | 85,71% | 0,6364 | 0,8250 | 0,9791 |

**XGBoost** được chọn làm mô hình triển khai. Kết quả của Random Forest có thể khác đôi chút nếu chạy ở môi trường khác. Kết quả phân tích bổ sung nằm trong `results/extra_summary.md`.

---

## YÊU CẦU HỆ THỐNG

- Windows 10/11, Linux hoặc macOS
- Python 3.10 trở lên (Docker dùng Python 3.11)
- Kết nối Internet ở lần chạy đầu để tải dữ liệu (khoảng 144 MB)
- Docker (nếu chạy bằng container)
- Các thư viện trong `requirements.txt`: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imbalanced-learn, requests, streamlit

---

## HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY

### Cách 1: Chạy trực tiếp

```bash
git clone <link-repo>
cd <ten-repo>
pip install -r requirements.txt

# Chạy toàn bộ quy trình: tải dữ liệu, EDA, huấn luyện, kiểm thử dự đoán
python run_pipeline.py

# Mở ứng dụng tại http://localhost:8501
streamlit run app.py
```

Repo đã kèm sẵn `models/` và `plots/`, nên có thể chạy `streamlit run app.py` ngay mà không cần chạy lại pipeline.

### Cách 2: Chạy bằng Docker

```bash
docker build -t fraud-detector .
docker run -p 8501:8501 fraud-detector
```

Sau đó truy cập `http://localhost:8501`. Ảnh Docker không chứa `data/creditcard.csv` để nhẹ hơn, nhưng vẫn giữ `models/` và `plots/` để ứng dụng hiển thị đầy đủ.

### Phân tích bổ sung (tùy chọn)

```bash
python src/extra_analysis.py            # đầy đủ, khoảng 5-15 phút
python src/extra_analysis.py --quick    # chạy thử nhanh, số liệu KHÔNG dùng để báo cáo
```

---

## CẤU TRÚC THƯ MỤC

```
.
├── app.py              Ứng dụng Streamlit (3 thẻ: Kiểm tra giao dịch / EDA / Hiệu năng mô hình)
├── run_pipeline.py     Chạy toàn bộ quy trình + kiểm thử dự đoán 2 giao dịch giả định
├── requirements.txt    Danh sách thư viện
├── Dockerfile          Đóng gói ứng dụng
├── src/
│   ├── download_data.py    Tải dữ liệu
│   ├── eda.py              Phân tích khám phá dữ liệu
│   ├── preprocess.py       Chia dữ liệu và chuẩn hóa
│   ├── train.py            Huấn luyện và đánh giá mô hình
│   ├── predict.py          Dự đoán giao dịch mới
│   └── extra_analysis.py   Phân tích bổ sung
├── models/             Mô hình và bộ chuẩn hóa đã lưu (.pkl)
├── plots/              Biểu đồ EDA và so sánh mô hình
├── results/            Kết quả phân tích bổ sung
├── data/               creditcard.csv (tự tải, không đưa lên Git)
└── docs/               Báo cáo cuối kỳ và hình minh họa
```

---

## HẠN CHẾ ĐÃ BIẾT

- Mô hình được chọn trên chính tập test và chỉ dùng một lần chia dữ liệu.
- Chưa tinh chỉnh siêu tham số và ngưỡng phân loại trong quy trình chính.
- Còn 1.081 dòng trùng lặp trong dữ liệu gốc.
- Ứng dụng demo chỉ cho chỉnh 5 thuộc tính (V17, V14, V12, V11, V4); các thuộc tính còn lại đặt bằng 0.

Chi tiết xem trong báo cáo 

---

## THÀNH VIÊN NHÓM

| Thành viên | MSSV | Mức độ hoàn thành |
|---|---|---|
| Lê Thị Vân Anh | 46.01.103.007 | 100% |
| Lê Việt Hoàng Thảo | 47.01.104.196 | 100% |

---

## GHI CHÚ
- Xem hướng dẫn chạy chi tiết tại [HUONG_DAN_DEMO.md](HUONG_DAN_DEMO.md)
- Không đưa file `.env`, API key hoặc mật khẩu lên Git.
- `data/creditcard.csv` nặng hơn 100 MB nên bị `.gitignore` chặn; dữ liệu được tải tự động khi chạy `run_pipeline.py`.
