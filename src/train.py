import os
import sys
import io

# Cấu hình encoding utf-8 cho console trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a model and returns key performance metrics.
    """
    print(f"\n==================== EVALUATING {model_name} ====================")
    
    # Predict classes and probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUPRC (Average Precision): {auprc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Valid", "Fraud"]))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("========================================================\n")
    
    return {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'confusion_matrix': conf_matrix
    }

def train_and_evaluate_all(X_train, X_test, y_train, y_test, models_dir, plots_dir):
    """
    Trains multiple models (Logistic Regression, Random Forest, XGBoost, and SMOTE+RF)
    and compares their performance. Gathers the best model and saves it.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {}
    
    # ----------------------------------------------------
    # Model 1: Logistic Regression with Class Weights
    # ----------------------------------------------------
    print("[TRAIN] Training Logistic Regression (Balanced Class Weights)...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    results['Logistic Regression (Weighted)'] = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression (Weighted)')
    
    # ----------------------------------------------------
    # Model 2: Random Forest with Class Weights
    # ----------------------------------------------------
    print("[TRAIN] Training Random Forest (Balanced Class Weights)...")
    rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    results['Random Forest (Weighted)'] = evaluate_model(rf_model, X_test, y_test, 'Random Forest (Weighted)')
    
    # ----------------------------------------------------
    # Model 3: XGBoost with scale_pos_weight
    # ----------------------------------------------------
    print("[TRAIN] Training XGBoost (scale_pos_weight)...")
    # scale_pos_weight = count(negative) / count(positive)
    neg_count = len(y_train) - sum(y_train)
    pos_count = sum(y_train)
    scale_weight = neg_count / pos_count
    
    xgb_model = XGBClassifier(scale_pos_weight=scale_weight, n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    results['XGBoost (Weighted)'] = evaluate_model(xgb_model, X_test, y_test, 'XGBoost (Weighted)')

    # ----------------------------------------------------
    # Model 4: SMOTE + Random Forest Pipeline
    # ----------------------------------------------------
    print("[TRAIN] Training SMOTE + Random Forest...")
    smote = SMOTE(random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    
    # Using imblearn Pipeline so SMOTE is ONLY applied to train fold during fit, never on validation
    smote_rf_pipeline = ImbPipeline([
        ('smote', smote),
        ('classifier', rf_classifier)
    ])
    smote_rf_pipeline.fit(X_train, y_train)
    results['SMOTE + Random Forest'] = evaluate_model(smote_rf_pipeline, X_test, y_test, 'SMOTE + Random Forest')

    # ----------------------------------------------------
    # 5. Visualizing & Comparing Models
    # ----------------------------------------------------
    print("[INFO] Creating model comparison visualizations...")
    
    # Plot Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for model_name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        plt.plot(recall, precision, label=f"{model_name} (AUPRC = {res['auprc']:.4f})")
        
    plt.title("Đường cong Precision-Recall So Sánh Các Mô Hình", fontsize=14, fontweight='bold')
    plt.xlabel("Recall (Độ nhạy)", fontsize=12)
    plt.ylabel("Precision (Độ chính xác)", fontsize=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True)
    pr_curve_path = os.path.join(plots_dir, "model_comparison_pr_curve.png")
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Precision-Recall Curve comparison to {pr_curve_path}")

    # Plot Bar Chart Comparison of Key Metrics
    metrics_data = []
    for model_name, res in results.items():
        metrics_data.append({
            'Model': model_name,
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1'],
            'AUPRC': res['auprc']
        })
    df_metrics = pd.DataFrame(metrics_data)
    df_melted = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title("So Sánh Hiệu Năng Các Mô Hình", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mô hình", fontsize=12)
    plt.ylabel("Điểm số (Score)", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    comparison_path = os.path.join(plots_dir, "model_comparison_bar.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved metrics comparison bar plot to {comparison_path}")

    # Plot Confusion Matrices Side by Side
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for idx, (model_name, res) in enumerate(results.items()):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'], annot_kws={'size': 14})
        axes[idx].set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrices comparison to {cm_path}")

    # ----------------------------------------------------
    # 6. Save the Best Model
    # ----------------------------------------------------
    # We choose the best model based on AUPRC (Area Under Precision-Recall Curve),
    # which is the gold standard for highly imbalanced settings.
    best_model_name = max(results, key=lambda k: results[k]['auprc'])
    print(f"\n[BEST MODEL] Best model according to AUPRC is: {best_model_name}")
    
    best_model_obj = None
    if best_model_name == 'Logistic Regression (Weighted)':
        best_model_obj = lr_model
    elif best_model_name == 'Random Forest (Weighted)':
        best_model_obj = rf_model
    elif best_model_name == 'XGBoost (Weighted)':
        best_model_obj = xgb_model
    elif best_model_name == 'SMOTE + Random Forest':
        best_model_obj = smote_rf_pipeline

    best_model_path = os.path.join(models_dir, 'best_fraud_detector.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_obj, f)
    print(f"[SUCCESS] Saved best model ({best_model_name}) to {best_model_path}")
    
    # Ghi lại bảng so sánh rút gọn ra file markdown hoặc log
    summary_df = df_metrics.set_index('Model')
    print("\n--- BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH ---")
    print(summary_df.to_string())
    print("----------------------------------------\n")
    
    return best_model_name, summary_df

if __name__ == "__main__":
    # Standard usage test
    from preprocess import preprocess_dataset
    X_train, X_test, y_train, y_test = preprocess_dataset("data/creditcard.csv", "models")
    train_and_evaluate_all(X_train, X_test, y_train, y_test, "models", "plots")
