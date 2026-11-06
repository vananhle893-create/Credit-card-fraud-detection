import os
import sys
import io

# Cấu hình encoding utf-8 cho console trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path, plots_dir):
    """
    Performs Exploratory Data Analysis on the credit card transactions dataset.
    """
    print("[INFO] Starting Exploratory Data Analysis (EDA)...")
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}. Please download it first.")
        return False

    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"[INFO] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check missing values
    missing_values = df.isnull().sum().sum()
    print(f"[INFO] Total missing values: {missing_values}")
    
    # 2. Class Distribution Analysis
    class_counts = df['Class'].value_counts()
    valid_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    total_count = len(df)
    
    valid_pct = (valid_count / total_count) * 100
    fraud_pct = (fraud_count / total_count) * 100
    
    print("\n--- CLASS DISTRIBUTION ---")
    print(f"Valid Transactions (Class 0): {valid_count:,} ({valid_pct:.3f}%)")
    print(f"Fraudulent Transactions (Class 1): {fraud_count:,} ({fraud_pct:.3f}%)")
    print("--------------------------\n")
    
    # Plot Class Distribution (using clean harmonious colors, modern typography)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    colors = ["#4A90E2", "#D0021B"] # Modern blue & red
    ax = sns.countplot(x='Class', data=df, palette=colors)
    plt.title("Phân Phối Giao Dịch: Hợp Lệ (0) vs Giả Mạo (1)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Lớp (Class)", fontsize=12)
    plt.ylabel("Số lượng giao dịch (Log scale)", fontsize=12)
    plt.yscale('log') # Use log scale because class imbalance is extreme
    
    # Annotate counts and percentages
    for i, p in enumerate(ax.patches):
        count = class_counts[i]
        pct = (count / total_count) * 100
        ax.annotate(f'{count:,}\n({pct:.3f}%)', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points', 
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot_path1 = os.path.join(plots_dir, "class_distribution.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"[INFO] Saved class distribution plot to {plot_path1}")

    # 3. Analyze Amount and Time
    print("[INFO] Analyzing Time and Amount distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Amount distribution (Log distribution for visualization since Amount has high range)
    sns.histplot(data=df[df['Class'] == 0], x='Amount', ax=axes[0], color='#4A90E2', label='Hợp lệ', kde=True, bins=50, log_scale=True)
    sns.histplot(data=df[df['Class'] == 1], x='Amount', ax=axes[0], color='#D0021B', label='Giả mạo', kde=True, bins=50, log_scale=True)
    axes[0].set_title("Phân Phối Số Tiền Giao Dịch (Log Scale)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Số tiền (USD)", fontsize=11)
    axes[0].set_ylabel("Mật độ / Số lượng", fontsize=11)
    axes[0].legend()
    
    # Time distribution
    sns.kdeplot(data=df[df['Class'] == 0], x='Time', ax=axes[1], color='#4A90E2', label='Hợp lệ', fill=True, alpha=0.3)
    sns.kdeplot(data=df[df['Class'] == 1], x='Time', ax=axes[1], color='#D0021B', label='Giả mạo', fill=True, alpha=0.3)
    axes[1].set_title("Phân Phối Thời Gian Giao Dịch", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Thời gian (giây kể từ GD đầu tiên)", fontsize=11)
    axes[1].set_ylabel("Mật độ (Density)", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, "feature_distributions.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"[INFO] Saved feature distributions plot to {plot_path2}")

    # 4. Correlation Matrix of PCA features and Target
    print("[INFO] Computing correlation matrix...")
    plt.figure(figsize=(12, 10))
    # We sample or use full dataset because correlation calculation is fast enough
    corr = df.corr()
    
    # We plot the correlation matrix. Since V1-V28 are PCA-derived, their mutual correlation is 0.
    # We focus on the correlation of all features with the 'Class' target.
    sns.heatmap(corr, cmap="coolwarm", robust=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Ma Trận Tương Quan Toàn Bộ Các Đặc Trưng", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plot_path3 = os.path.join(plots_dir, "correlation_matrix.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"[INFO] Saved correlation matrix plot to {plot_path3}")
    
    # Identify top positively and negatively correlated features with Class
    corr_with_class = corr['Class'].drop('Class').sort_values()
    print("\n--- TOP CORRELATED FEATURES WITH CLASS ---")
    print("Top 5 Negatively Correlated Features:")
    print(corr_with_class.head(5))
    print("\nTop 5 Positively Correlated Features:")
    print(corr_with_class.tail(5))
    print("------------------------------------------\n")
    
    print("[SUCCESS] Exploratory Data Analysis completed successfully!")
    return True

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "creditcard.csv")
    PLOTS_DIR = "plots"
    run_eda(DATA_PATH, PLOTS_DIR)
