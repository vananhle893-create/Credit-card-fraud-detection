# HƯỚNG DẪN CHẠY DỰ ÁN (QUY TRÌNH DEMO)

Dự án: **Phát hiện giao dịch thẻ tín dụng giả mạo**

## A. Chạy quy trình chính

1. Mở VS Code, chọn **File → Open Folder**, chọn thư mục `du_doan_giao_dich` (thư mục có `app.py`). Vào Extensions (Ctrl+Shift+X), cài extension **Python**.
2. Mở terminal: **Terminal → New Terminal** (Ctrl+`). Kiểm tra terminal đang ở đúng thư mục dự án.
3. Tạo môi trường ảo:
   ```
   py -m venv venv
   ```
4. Bật môi trường ảo:
   ```
   venv\Scripts\activate
   ```
   Nếu báo lỗi đỏ chặn script, gõ `Set-ExecutionPolicy -Scope Process Bypass` rồi bật lại. Thành công khi đầu dòng có chữ `(venv)`.
5. Cài thư viện:
   ```
   pip install -r requirements.txt
   ```
   Thành công khi hiện dòng `Successfully installed ...` và đầu dòng vẫn có `(venv)`.
6. Chạy toàn bộ quy trình:
   ```
   py run_pipeline.py
   ```
   Chương trình chạy vài phút (huấn luyện 4 mô hình). **Không bấm gì** cho tới khi dòng cuối lại hiện `(venv) PS D:\...>`.
7. Xem kết quả ở thanh bên trái VS Code:
   - Thư mục **plots**, mở các hình: `class_distribution.png`, `feature_distributions.png`, `correlation_matrix.png`, `model_comparison_bar.png`, `model_comparison_pr_curve.png`, `confusion_matrices.png`.
   - Thư mục **models** chứa 2 file mô hình đã lưu: `best_fraud_detector.pkl` và `scalers.pkl` (chỉ cần chỉ ra file đã được tạo, không mở xem được).

## B. Phân tích bổ sung (vẫn dùng terminal cũ)

1. Chạy thử nhanh (kiểm tra lỗi):
   ```
   python src/extra_analysis.py --quick
   ```
   Cuối cùng phải hiện dòng `[SUCCESS]`.
2. Chạy thật. **Không bấm gì** cho tới khi hiện lại `(venv) PS D:\...>`:
   ```
   python src/extra_analysis.py
   ```
   Chạy xong sẽ có thêm 3 hình trong thư mục **plots**:
   - `cv_results.png`
   - `threshold_tuning.png`
   - `xgb_feature_importance.png`
3. Mở file `results/extra_summary.md`, nhìn **dòng thứ ba**: phải ghi **"Chế độ: đầy đủ"**. Nếu thấy chữ **QUICK** nghĩa là file chưa cập nhật, tắt file trong VS Code rồi mở lại.

> Lưu ý: chạy thử nhanh (B1) sẽ ghi đè 3 hình và file kết quả bằng số liệu thử nghiệm, nên luôn chạy thật (B2) sau cùng.

## C. Chạy ứng dụng web

1. Vẫn ở terminal cũ, gõ:
   ```
   streamlit run app.py
   ```
   Trình duyệt tự mở, hoặc vào thủ công `http://localhost:8501` (hay `http://127.0.0.1:8501`).
2. Ứng dụng đã được đưa lên mạng bằng Streamlit Community Cloud: https://phat-hien-giao-dich-gia-mao.streamlit.app
