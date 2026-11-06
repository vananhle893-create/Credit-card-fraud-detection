"""
Phân tích bổ sung cho báo cáo (không thay đổi quy trình chính trong run_pipeline.py).

Gồm 4 phần:
  A. Kiểm tra chéo phân tầng 5-fold trên TẬP HUẤN LUYỆN (không dùng tập test)
  B. Chọn ngưỡng phân loại tối ưu F1 bằng dự đoán out-of-fold của phần A,
     rồi áp dụng 1 lần lên tập test (so sánh với ngưỡng mặc định 0.5)
  C. So sánh khi bỏ các dòng trùng lặp trong dữ liệu
  D. Mức độ quan trọng của các thuộc tính (feature importance) của XGBoost

Cách chạy (trong thư mục gốc của dự án, đã bật môi trường ảo):
    python src/extra_analysis.py            # chạy đầy đủ, khoảng 5-15 phút
    python src/extra_analysis.py --quick    # chạy thử nhanh trên mẫu nhỏ để kiểm tra lỗi

Kết quả:
    results/extra_summary.md   (gom toàn bộ bảng số, gửi file này là đủ)
    results/*.csv              (từng bảng)
    plots/cv_results.png, plots/threshold_tuning.png, plots/xgb_feature_importance.png
"""
import os
import sys
import json
import time
import argparse

# Cấu hình encoding utf-8 cho console trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SMOTE + Random Forest']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- tiện ích
def make_model(name, y_train):
    """Cấu hình giống hệt src/train.py."""
    if name == 'Logistic Regression':
        return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    if name == 'Random Forest':
        return RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=12,
                                      random_state=42, n_jobs=-1)
    if name == 'XGBoost':
        pos = float(np.sum(y_train))
        neg = float(len(y_train) - pos)
        return XGBClassifier(scale_pos_weight=neg / pos, n_estimators=150, max_depth=6,
                             learning_rate=0.1, random_state=42, n_jobs=-1)
    if name == 'SMOTE + Random Forest':
        return ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12,
                                                  random_state=42, n_jobs=-1)),
        ])
    raise ValueError(name)


def scale(X_train, others):
    """RobustScaler cho Amount và Time, chỉ fit trên X_train (tránh rò rỉ dữ liệu)."""
    tr = X_train.copy()
    outs = [x.copy() for x in others]
    for col in ['Amount', 'Time']:
        sc = RobustScaler()
        tr[col] = sc.fit_transform(X_train[[col]])
        for o, x in zip(outs, others):
            o[col] = sc.transform(x[[col]])
    return tr, outs


def metrics_at(y, p, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def best_threshold(y, p):
    """Ngưỡng cho F1 cao nhất trên dự đoán (dùng dự đoán out-of-fold, không dùng test)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i]), thr, f1


def md_table(df, floatfmt=4):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'{v:.{floatfmt}f}' if isinstance(v, (float, np.floating)) else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- chương trình chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='chạy thử nhanh trên mẫu nhỏ')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--skip-dedup', action='store_true', help='bỏ qua phần C')
    args = ap.parse_args()

    t0 = time.time()
    data_path = os.path.join(ROOT, 'data', 'creditcard.csv')
    plots_dir = os.path.join(ROOT, 'plots')
    res_dir = os.path.join(ROOT, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    if args.quick:
        log('[QUICK] Chế độ chạy thử: dùng mẫu nhỏ, số liệu KHÔNG dùng để báo cáo.')

    df = pd.read_csv(data_path)
    if args.quick:
        fraud = df[df['Class'] == 1]
        valid = df[df['Class'] == 0].sample(20000, random_state=42)
        df = pd.concat([fraud, valid]).sample(frac=1, random_state=42).reset_index(drop=True)
    log(f'[INFO] Dữ liệu: {len(df)} dòng, {int(df.Class.sum())} giao dịch giả mạo')

    X = df.drop(columns=['Class'])
    y = df['Class']
    # chia giống hệt src/preprocess.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log(f'[INFO] Train {len(X_train)} ({int(y_train.sum())} giả mạo) | Test {len(X_test)} ({int(y_test.sum())} giả mạo)')

    summary = ['# Kết quả phân tích bổ sung', '',
               f'Chế độ: {"QUICK (chạy thử, không dùng để báo cáo)" if args.quick else "đầy đủ"}', '']

    # ============ A. kiểm tra chéo 5-fold trên tập huấn luyện ============
    log(f'\n=== A. Kiểm tra chéo phân tầng {args.folds}-fold trên tập huấn luyện ===')
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = {n: np.zeros(len(X_train)) for n in NAMES}
    fold_rows = []
    Xtr_np_index = np.arange(len(X_train))
    for k, (tr_idx, va_idx) in enumerate(skf.split(Xtr_np_index, y_train), start=1):
        Xf_tr, Xf_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        yf_tr, yf_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        Xf_tr_s, (Xf_va_s,) = scale(Xf_tr, [Xf_va])
        for n in NAMES:
            m = make_model(n, yf_tr)
            m.fit(Xf_tr_s, yf_tr)
            p = m.predict_proba(Xf_va_s)[:, 1]
            oof[n][va_idx] = p
            mt = metrics_at(yf_va, p, 0.5)
            fold_rows.append(dict(fold=k, model=n,
                                  auprc=average_precision_score(yf_va, p),
                                  roc_auc=roc_auc_score(yf_va, p),
                                  precision=mt['precision'], recall=mt['recall'], f1=mt['f1']))
        log(f'  xong fold {k}/{args.folds}  ({time.time() - t0:.0f}s)')
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(res_dir, 'cv_folds.csv'), index=False)
    agg = folds_df.groupby('model')[['precision', 'recall', 'f1', 'auprc', 'roc_auc']].agg(['mean', 'std'])
    cv_rows = []
    for n in NAMES:
        row = {'Mô hình': n}
        for c, label in [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1'),
                         ('auprc', 'AUPRC'), ('roc_auc', 'ROC-AUC')]:
            row[label] = f"{agg.loc[n, (c, 'mean')]:.4f} ± {agg.loc[n, (c, 'std')]:.4f}"
        cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(res_dir, 'cv_summary.csv'), index=False)
    log(cv_df.to_string(index=False))
    summary += [f'## A. Kiểm tra chéo phân tầng {args.folds}-fold trên tập huấn luyện (trung bình ± độ lệch chuẩn, ngưỡng 0.5)', '',
                md_table(cv_df), '']

    # biểu đồ CV
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, c, title in [(axes[0], 'auprc', 'AUPRC'), (axes[1], 'f1', 'F1-Score')]:
        means = [agg.loc[n, (c, 'mean')] for n in NAMES]
        stds = [agg.loc[n, (c, 'std')] for n in NAMES]
        ax.bar(range(len(NAMES)), means, yerr=stds, capsize=6, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels([n.replace(' + ', '\n+ ').replace(' Regression', '\nRegression').replace(' Forest', '\nForest')
                            if n != 'SMOTE + Random Forest' else 'SMOTE +\nRandom Forest' for n in NAMES], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{title} qua {args.folds}-fold CV (trung bình ± độ lệch chuẩn)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cv_results.png'), dpi=200)
    plt.close()

    # ============ huấn luyện lần cuối trên toàn bộ tập train (giống train.py) ============
    log('\n=== Huấn luyện lần cuối trên toàn bộ tập train để đánh giá trên test ===')
    X_train_s, (X_test_s,) = scale(X_train, [X_test])
    final_models, test_prob = {}, {}
    for n in NAMES:
        m = make_model(n, y_train)
        m.fit(X_train_s, y_train)
        final_models[n] = m
        test_prob[n] = m.predict_proba(X_test_s)[:, 1]
        log(f'  xong {n}  ({time.time() - t0:.0f}s)')

    # ============ B. chọn ngưỡng ============
    log('\n=== B. Chọn ngưỡng tối ưu F1 bằng dự đoán out-of-fold ===')
    thr_rows = []
    curves = {}
    for n in NAMES:
        thr, f1_oof, t_arr, f1_arr = best_threshold(y_train, oof[n])
        curves[n] = (t_arr, f1_arr, thr)
        m05 = metrics_at(y_test, test_prob[n], 0.5)
        mtu = metrics_at(y_test, test_prob[n], thr)
        thr_rows.append({
            'Mô hình': n, 'Ngưỡng chọn': thr, 'F1 (OOF)': f1_oof,
            'Precision @0.5': m05['precision'], 'Recall @0.5': m05['recall'], 'F1 @0.5': m05['f1'],
            'FP @0.5': m05['fp'], 'FN @0.5': m05['fn'],
            'Precision @chọn': mtu['precision'], 'Recall @chọn': mtu['recall'], 'F1 @chọn': mtu['f1'],
            'FP @chọn': mtu['fp'], 'FN @chọn': mtu['fn'],
        })
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(os.path.join(res_dir, 'threshold_results.csv'), index=False)
    log(thr_df.to_string(index=False))
    summary += ['## B. Chọn ngưỡng bằng dự đoán out-of-fold, đánh giá trên tập test', '',
                '(@0.5 = ngưỡng mặc định; @chọn = ngưỡng tối ưu F1 trên dự đoán out-of-fold của tập train)', '',
                md_table(thr_df), '']
    # AUPRC / ROC-AUC trên test để đối chiếu với Bảng 4.2 của báo cáo
    base_rows = []
    for n in NAMES:
        mt = metrics_at(y_test, test_prob[n], 0.5)
        base_rows.append({'Mô hình': n, 'Precision': mt['precision'], 'Recall': mt['recall'], 'F1': mt['f1'],
                          'AUPRC': average_precision_score(y_test, test_prob[n]),
                          'ROC-AUC': roc_auc_score(y_test, test_prob[n]),
                          'TP': mt['tp'], 'FN': mt['fn'], 'FP': mt['fp'], 'TN': mt['tn']})
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(os.path.join(res_dir, 'test_baseline.csv'), index=False)
    summary += ['## Đối chiếu: kết quả trên tập test của lần chạy này (ngưỡng 0.5, có dòng trùng lặp)', '',
                md_table(base_df), '']

    # biểu đồ F1 theo ngưỡng cho XGBoost
    t_arr, f1_arr, thr = curves['XGBoost']
    plt.figure(figsize=(8, 5))
    plt.plot(t_arr, f1_arr, color='#55A868', lw=2, label='F1 theo ngưỡng (out-of-fold, XGBoost)')
    plt.axvline(0.5, color='gray', ls='--', label='Ngưỡng mặc định 0.5')
    plt.axvline(thr, color='#C44E52', ls='--', label=f'Ngưỡng chọn = {thr:.3f}')
    plt.xlabel('Ngưỡng phân loại')
    plt.ylabel('F1-Score')
    plt.title('Chọn ngưỡng phân loại cho XGBoost', fontsize=12, fontweight='bold')
    plt.legend(loc='lower center')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'threshold_tuning.png'), dpi=200)
    plt.close()

    # ============ D. feature importance ============
    log('\n=== D. Feature importance của XGBoost ===')
    xgb_model = final_models['XGBoost']
    imp = pd.DataFrame({'Thuộc tính': list(X_train_s.columns),
                        'Độ quan trọng': np.asarray(xgb_model.feature_importances_, dtype=float)})
    imp = imp.sort_values('Độ quan trọng', ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(res_dir, 'xgb_feature_importance.csv'), index=False)
    top = imp.head(15)
    log(top.to_string(index=False))
    summary += ['## D. Mức độ quan trọng của thuộc tính (XGBoost, 15 thuộc tính đầu)', '', md_table(top), '']
    plt.figure(figsize=(8, 6))
    plt.barh(top['Thuộc tính'][::-1], top['Độ quan trọng'][::-1], color='#4C72B0')
    plt.xlabel('Độ quan trọng (feature_importances_ của XGBoost)')
    plt.title('15 thuộc tính quan trọng nhất của XGBoost', fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'xgb_feature_importance.png'), dpi=200)
    plt.close()

    # ============ C. bỏ dòng trùng lặp ============
    if not args.skip_dedup:
        log('\n=== C. Bỏ các dòng trùng lặp rồi huấn luyện lại ===')
        dfd = df.drop_duplicates().reset_index(drop=True)
        n_removed = len(df) - len(dfd)
        log(f'  Đã bỏ {n_removed} dòng trùng lặp; còn {len(dfd)} dòng, {int(dfd.Class.sum())} giả mạo')
        Xd, yd = dfd.drop(columns=['Class']), dfd['Class']
        Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
        Xd_tr_s, (Xd_te_s,) = scale(Xd_tr, [Xd_te])
        d_rows = []
        for n in NAMES:
            m = make_model(n, yd_tr)
            m.fit(Xd_tr_s, yd_tr)
            p = m.predict_proba(Xd_te_s)[:, 1]
            mt = metrics_at(yd_te, p, 0.5)
            d_rows.append({'Mô hình': n, 'Precision': mt['precision'], 'Recall': mt['recall'], 'F1': mt['f1'],
                           'AUPRC': average_precision_score(yd_te, p), 'ROC-AUC': roc_auc_score(yd_te, p),
                           'TP': mt['tp'], 'FN': mt['fn'], 'FP': mt['fp'], 'TN': mt['tn']})
            log(f'  xong {n}  ({time.time() - t0:.0f}s)')
        d_df = pd.DataFrame(d_rows)
        d_df.to_csv(os.path.join(res_dir, 'dedup_results.csv'), index=False)
        log(d_df.to_string(index=False))
        summary += [f'## C. Sau khi bỏ {n_removed} dòng trùng lặp (còn {len(dfd)} dòng, {int(dfd.Class.sum())} giao dịch giả mạo; '
                    f'tập test {len(Xd_te)} giao dịch, {int(yd_te.sum())} giả mạo)', '', md_table(d_df), '']

    with open(os.path.join(res_dir, 'extra_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    log(f'\n[SUCCESS] Xong sau {time.time() - t0:.0f} giây.')
    log('  Gửi file results/extra_summary.md và 3 ảnh trong plots/: '
        'cv_results.png, threshold_tuning.png, xgb_feature_importance.png')


if __name__ == '__main__':
    main()
