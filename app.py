import os
import sys
import io

# Cấu hình encoding utf-8 cho console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Cấu hình giao diện Streamlit (Premium design)
st.set_page_config(
    page_title="Phát Hiện Gian Lận Thẻ Tín Dụng",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy biến cho đẹp mắt
st.markdown("""
<style>
    .main-title {
        color: #4A90E2;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>💳 HỆ THỐNG PHÁT HIỆN GIAO DỊCH THẺ TÍN DỤNG GIẢ MẠO</h1>", unsafe_allow_html=True)
st.markdown("---")

# 1. Hàm Load mô hình và bộ scaler đã lưu
@st.cache_resource
def load_assets():
    models_dir = "models"
    scalers_path = os.path.join(models_dir, 'scalers.pkl')
    model_path = os.path.join(models_dir, 'best_fraud_detector.pkl')
    
    if not os.path.exists(scalers_path) or not os.path.exists(model_path):
        return None, None
        
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return model, scalers

model, scalers = load_assets()

if model is None:
    st.error("⚠️ Không tìm thấy tệp tin mô hình đã huấn luyện trong thư mục `models/`!")
    st.info("💡 Vui lòng chạy kịch bản huấn luyện bằng lệnh terminal trước: `py run_pipeline.py` để tạo và lưu mô hình.")
    st.stop()

# 2. Sidebar - Nhập dữ liệu giao dịch
st.sidebar.header("📥 THÔNG TIN GIAO DỊCH MỚI")

time_val = st.sidebar.number_input("⏰ Thời gian (Giây kể từ GD đầu tiên)", min_value=0.0, value=100.0, step=1.0)
amount_val = st.sidebar.number_input("💵 Số tiền giao dịch (USD)", min_value=0.0, value=50.0, step=0.01)

st.sidebar.subheader("🧬 Các đặc trưng ẩn danh quan trọng (PCA)")
st.sidebar.write("*(Điều chỉnh các đặc trưng có độ tương quan mạnh đến gian lận)*")

# Sliders cho các PCA features có tương quan mạnh
v17 = st.sidebar.slider("V17 (Tương quan âm cực mạnh)", -30.0, 10.0, 0.0, step=0.1)
v14 = st.sidebar.slider("V14 (Tương quan âm cực mạnh)", -30.0, 10.0, 0.0, step=0.1)
v12 = st.sidebar.slider("V12 (Tương quan âm mạnh)", -30.0, 10.0, 0.0, step=0.1)
v11 = st.sidebar.slider("V11 (Tương quan dương mạnh)", -10.0, 20.0, 0.0, step=0.1)
v4 = st.sidebar.slider("V4 (Tương quan dương mạnh)", -10.0, 20.0, 0.0, step=0.1)

# Các features khác gán mặc định bằng 0
other_vs = {}
for i in range(1, 29):
    feature_name = f'V{i}'
    if feature_name not in ['V17', 'V14', 'V12', 'V11', 'V4']:
        other_vs[feature_name] = 0.0

# Tạo Tabs giao diện chính
tab1, tab2, tab3 = st.tabs(["🔍 Kiểm Tra Giao Dịch", "📊 Biểu Đồ Trực Quan Dữ Liệu (EDA)", "🏆 Hiệu Năng Các Mô Hình"])

# TAB 1: DỰ ĐOÁN GIAO DỊCH
with tab1:
    st.write("### Nhấn nút kiểm tra ở cột bên trái để đánh giá mức độ rủi ro của giao dịch.")
    
    if st.sidebar.button("🚀 KIỂM TRA GIAO DỊCH", use_container_width=True):
        # Thiết lập dictionary dữ liệu
        tx_data = {
            'Time': time_val,
            'Amount': amount_val,
            'V17': v17,
            'V14': v14,
            'V12': v12,
            'V11': v11,
            'V4': v4
        }
        tx_data.update(other_vs)
        
        # Chuyển đổi thành DataFrame
        df_tx = pd.DataFrame([tx_data])
        
        # Tiền xử lý (chuẩn hóa robust Amount và Time)
        df_tx_scaled = df_tx.copy()
        df_tx_scaled['Amount'] = scalers['amount_scaler'].transform(df_tx[['Amount']])
        df_tx_scaled['Time'] = scalers['time_scaler'].transform(df_tx[['Time']])
        
        # Sắp xếp đúng thứ tự cột của tập train
        feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_tx_scaled = df_tx_scaled[feature_cols]
        
        # Dự đoán
        prediction = model.predict(df_tx_scaled)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df_tx_scaled)[0][1]
        else:
            decision_score = model.decision_function(df_tx_scaled)[0]
            probability = 1 / (1 + np.exp(-decision_score))
            
        # Hiển thị kết quả trực quan
        st.markdown("### Kết quả Phân tích Rủi ro:")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction == 1:
                st.error("🚨 **CẢNH BÁO: GIAO DỊCH CÓ DẤU HIỆU GIẢ MẠO (FRAUD)**")
                st.markdown(f"**Xác suất gian lận được ước tính:** <h2 style='color:#D0021B;'>{probability*100:.4f}%</h2>", unsafe_allow_html=True)
            else:
                st.success("✅ **GIAO DỊCH AN TOÀN / HỢP LỆ (VALID)**")
                st.markdown(f"**Xác suất gian lận được ước tính:** <h2 style='color:#28A745;'>{probability*100:.4f}%</h2>", unsafe_allow_html=True)
                
        with col2:
            st.metric("💵 Số tiền giao dịch", f"${amount_val:,.2f}")
            st.metric("⏰ Thời gian giao dịch", f"{time_val}s")
            
        st.write("---")
        st.markdown("#### 💡 Các yếu tố ảnh hưởng trực tiếp đến kết quả này:")
        st.markdown(f"""
        - **Số tiền giao dịch (`Amount`):** ${amount_val} -> Sau chuẩn hóa Robust: `{df_tx_scaled['Amount'][0]:.4f}`
        - **Thời gian giao dịch (`Time`):** {time_val}s -> Sau chuẩn hóa Robust: `{df_tx_scaled['Time'][0]:.4f}`
        - **Thuộc tính V17 (Tương quan âm mạnh):** `{v17}` *(Các thuộc tính tương quan âm càng nhỏ (càng âm) thì giao dịch càng có rủi ro gian lận cực cao)*
        - **Thuộc tính V14 (Tương quan âm mạnh):** `{v14}`
        - **Thuộc tính V12 (Tương quan âm mạnh):** `{v12}`
        - **Thuộc tính V11 (Tương quan dương mạnh):** `{v11}` *(Các thuộc tính tương quan dương càng lớn thì giao dịch càng có rủi ro gian lận)*
        - **Thuộc tính V4 (Tương quan dương mạnh):** `{v4}`
        """)

# TAB 2: BIỂU ĐỒ TRỰC QUAN HÓA EDA
with tab2:
    st.subheader("📊 Kết quả phân tích và trực quan hóa tập dữ liệu giao dịch gốc")
    st.write("Tập dữ liệu chứa 284,807 dòng giao dịch, trong đó lớp Giả mạo chỉ chiếm 0.173% (492 giao dịch).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if os.path.exists("plots/class_distribution.png"):
            st.image("plots/class_distribution.png", caption="Phân phối lớp cực kỳ mất cân bằng (0: Hợp lệ, 1: Giả mạo) - Biểu đồ cột Log-scale", use_container_width=True)
        else:
            st.warning("Không tìm thấy biểu đồ `plots/class_distribution.png`")
            
    with col2:
        if os.path.exists("plots/feature_distributions.png"):
            st.image("plots/feature_distributions.png", caption="Phân phối số tiền (Amount - Log scale) và Thời gian (Time) giao dịch của lớp Hợp lệ vs Giả mạo", use_container_width=True)
        else:
            st.warning("Không tìm thấy biểu đồ `plots/feature_distributions.png`")
            
    st.write("---")
    st.subheader("🔥 Ma trận tương quan các đặc trưng với nhãn Giả mạo (Class)")
    if os.path.exists("plots/correlation_matrix.png"):
        st.image("plots/correlation_matrix.png", caption="Ma trận tương quan toàn bộ các đặc trưng biến đổi PCA cùng với thuộc tính Time, Amount và Class", use_container_width=True)
    else:
        st.warning("Không tìm thấy biểu đồ `plots/correlation_matrix.png`")

# TAB 3: HIỆU NĂNG CÁC MÔ HÌNH
with tab3:
    st.subheader("🏆 So sánh hiệu năng của các mô hình học máy trên tập kiểm thử")
    st.write("Đánh giá khách quan trên tập test gốc (56,962 giao dịch) với 98 mẫu giả mạo thực tế:")
    
    # Render bảng so sánh
    st.markdown("""
    | Mô hình | Phương pháp xử lý mất cân bằng | Precision | Recall | F1-Score | AUPRC | ROC-AUC |
    | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
    | **Logistic Regression** | Balanced Class Weights | 6.09% | **91.84%** | 0.1142 | 0.7175 | 0.9720 |
    | **Random Forest** | Balanced Class Weights | 71.68% | 82.65% | 0.7678 | 0.8236 | 0.9790 |
    | **XGBoost (Winner)** | Trọng số (`scale_pos_weight`) | **87.23%** | 83.67% | **0.8542** | **0.8736** | 0.9648 |
    | **SMOTE + Random Forest**| Oversampling tập Train | 50.60% | 85.71% | 0.6364 | 0.8250 | **0.9791** |
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if os.path.exists("plots/model_comparison_pr_curve.png"):
            st.image("plots/model_comparison_pr_curve.png", caption="Đường cong Precision-Recall so sánh hiệu năng các mô hình (AUPRC càng gần 1.0 càng tốt)", use_container_width=True)
        else:
            st.warning("Không tìm thấy biểu đồ `plots/model_comparison_pr_curve.png`")
            
    with col2:
        if os.path.exists("plots/model_comparison_bar.png"):
            st.image("plots/model_comparison_bar.png", caption="Biểu đồ so sánh trực quan các chỉ số đo lường", use_container_width=True)
        else:
            st.warning("Không tìm thấy biểu đồ `plots/model_comparison_bar.png`")
            
    st.write("---")
    st.subheader("🔵 Ma trận nhầm lẫn (Confusion Matrices) của các mô hình")
    if os.path.exists("plots/confusion_matrices.png"):
        st.image("plots/confusion_matrices.png", caption="Số lượng giao dịch dự đoán Đúng và Sai của từng mô hình (Mục tiêu: Giảm thiểu False Negatives ở dòng 2 cột 1 và False Positives ở dòng 1 cột 2)", use_container_width=True)
    else:
        st.warning("Không tìm thấy biểu đồ `plots/confusion_matrices.png`")
