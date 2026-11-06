# Kết quả phân tích bổ sung

Chế độ: đầy đủ

## A. Kiểm tra chéo phân tầng 5-fold trên tập huấn luyện (trung bình ± độ lệch chuẩn, ngưỡng 0.5)

| Mô hình | Precision | Recall | F1 | AUPRC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.0630 ± 0.0041 | 0.9138 ± 0.0258 | 0.1178 ± 0.0071 | 0.7477 ± 0.0559 | 0.9826 ± 0.0112 |
| Random Forest | 0.8473 ± 0.0288 | 0.8020 ± 0.0631 | 0.8221 ± 0.0224 | 0.8193 ± 0.0256 | 0.9747 ± 0.0167 |
| XGBoost | 0.8775 ± 0.0669 | 0.8046 ± 0.0460 | 0.8367 ± 0.0201 | 0.8418 ± 0.0226 | 0.9795 ± 0.0106 |
| SMOTE + Random Forest | 0.5588 ± 0.0756 | 0.8248 ± 0.0564 | 0.6615 ± 0.0437 | 0.8085 ± 0.0320 | 0.9830 ± 0.0125 |

## B. Chọn ngưỡng bằng dự đoán out-of-fold, đánh giá trên tập test

(@0.5 = ngưỡng mặc định; @chọn = ngưỡng tối ưu F1 trên dự đoán out-of-fold của tập train)

| Mô hình | Ngưỡng chọn | F1 (OOF) | Precision @0.5 | Recall @0.5 | F1 @0.5 | FP @0.5 | FN @0.5 | Precision @chọn | Recall @chọn | F1 @chọn | FP @chọn | FN @chọn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 0.8076 | 0.0609 | 0.9184 | 0.1142 | 1388 | 8 | 0.8298 | 0.7959 | 0.8125 | 16 | 20 |
| Random Forest | 0.5187 | 0.8316 | 0.7168 | 0.8265 | 0.7678 | 32 | 17 | 0.7297 | 0.8265 | 0.7751 | 30 | 17 |
| XGBoost | 0.8790 | 0.8627 | 0.8723 | 0.8367 | 0.8542 | 12 | 16 | 0.8966 | 0.7959 | 0.8432 | 9 | 20 |
| SMOTE + Random Forest | 0.8055 | 0.8179 | 0.5060 | 0.8571 | 0.6364 | 82 | 14 | 0.7778 | 0.7857 | 0.7817 | 22 | 21 |

## Đối chiếu: kết quả trên tập test của lần chạy này (ngưỡng 0.5, có dòng trùng lặp)

| Mô hình | Precision | Recall | F1 | AUPRC | ROC-AUC | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.0609 | 0.9184 | 0.1142 | 0.7175 | 0.9720 | 90 | 8 | 1388 | 55476 |
| Random Forest | 0.7168 | 0.8265 | 0.7678 | 0.8236 | 0.9790 | 81 | 17 | 32 | 56832 |
| XGBoost | 0.8723 | 0.8367 | 0.8542 | 0.8736 | 0.9648 | 82 | 16 | 12 | 56852 |
| SMOTE + Random Forest | 0.5060 | 0.8571 | 0.6364 | 0.8250 | 0.9791 | 84 | 14 | 82 | 56782 |

## D. Mức độ quan trọng của thuộc tính (XGBoost, 15 thuộc tính đầu)

| Thuộc tính | Độ quan trọng |
|---|---|
| V14 | 0.5495 |
| V4 | 0.0588 |
| V12 | 0.0463 |
| V8 | 0.0283 |
| V27 | 0.0195 |
| V20 | 0.0187 |
| Amount | 0.0183 |
| V7 | 0.0182 |
| V13 | 0.0176 |
| V28 | 0.0175 |
| V26 | 0.0161 |
| V18 | 0.0146 |
| V1 | 0.0141 |
| V10 | 0.0139 |
| V17 | 0.0131 |

## C. Sau khi bỏ 1081 dòng trùng lặp (còn 283726 dòng, 473 giao dịch giả mạo; tập test 56746 giao dịch, 95 giả mạo)

| Mô hình | Precision | Recall | F1 | AUPRC | ROC-AUC | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.0562 | 0.8737 | 0.1057 | 0.6719 | 0.9657 | 83 | 12 | 1393 | 55258 |
| Random Forest | 0.8780 | 0.7579 | 0.8136 | 0.7931 | 0.9722 | 72 | 23 | 10 | 56641 |
| XGBoost | 0.9146 | 0.7895 | 0.8475 | 0.8273 | 0.9698 | 75 | 20 | 7 | 56644 |
| SMOTE + Random Forest | 0.7170 | 0.8000 | 0.7562 | 0.7957 | 0.9769 | 76 | 19 | 30 | 56621 |
