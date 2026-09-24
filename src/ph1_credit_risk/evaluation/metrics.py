"""
Tính toán metrics đánh giá mô hình. Các metrics được chọn và lý do:

1. Accuracy: % dự đoán đúng tổng thể
   - Ưu: dễ hiểu
   - Nhược: KHÔNG đáng tin với data mất cân bằng (predict toàn 0 cũng được 78%)

2. Precision (class 1): Trong số dự đoán vỡ nợ, bao nhiêu % đúng?
   - Quan trọng khi chi phí false positive cao (từ chối nhầm khách tốt → mất doanh thu)

3. Recall (class 1): Trong số thực sự vỡ nợ, model phát hiện được bao nhiêu %?
   - Quan trọng khi chi phí false negative cao (cho vay khách xấu → mất tiền)
   - Trong tài chính, recall thường QUAN TRỌNG HƠN precision

4. F1-Score: Trung bình điều hòa của Precision và Recall
   - Cân bằng cả 2, hữu ích khi không biết ưu tiên cái nào

5. ROC-AUC: Diện tích dưới đường cong ROC
   - Đánh giá khả năng phân biệt giữa 2 class (threshold-independent)
   - METRIC CHÍNH cho bài toán này vì không phụ thuộc threshold
   - 0.5 = random, 1.0 = hoàn hảo

6. PR-AUC: Diện tích dưới đường cong Precision-Recall
   - Nhạy hơn ROC-AUC với imbalanced data
   - Tập trung vào performance trên class thiểu số (vỡ nợ)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

def compute_all_metrics(y_true, y_pred, y_proba):
    """
    Tính tất cả metrics cho 1 model.

    Args:
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán (0/1)
        y_proba: Xác suất dự đoán class 1

    Returns:
        dict: Tất cả metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }

def print_classification_report(y_true, y_pred, model_name="Model"):
    """In báo cáo phân loại chi tiết."""
    print(f"\n{'='*60}")
    print(f"  BÁO CÁO PHÂN LOẠI: {model_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Không vỡ nợ (0)", "Vỡ nợ (1)"]
    ))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Ma trận nhầm lẫn:")
    print(f"    True Negative  (TN): {tn:>6} | False Positive (FP): {fp:>6}")
    print(f"    False Negative (FN): {fn:>6} | True Positive  (TP): {tp:>6}")
    print(f"{'='*60}")

# KS, Brier, calibration, ngưỡng theo chi phí nghiệp vụ
#
# KS statistic: max |TPR - FPR| trên mọi ngưỡng = khoảng cách lớn nhất 
# giữa hai phân phối điểm (vỡ nợ / không). Chuẩn trong credit scoring;
# 0 = không phân biệt, 1 = tách hoàn toàn.
#
# Brier score: trung bình (p - y)^2. Đo chất lượng xác suất (calibration + sharpness), thấp hơn = tốt hơn. 
# AUC không thấy được model "đúng thứ tự nhưng sai độ lớn xác suất" — Brier thấy.
#
# Ngưỡng theo chi phí: FN (cho vay khách xấu) đắt hơn FP (từ chối khách tốt),
#  nên ngưỡng tối ưu thường < 0.5. Quét ngưỡng, chọn tổng chi phí thấp nhất.

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_curve


def ks_statistic(y_true, y_score):
    """KS = max_t |TPR(t) - FPR(t)|."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))

def brier_score(y_true, y_score):
    return float(brier_score_loss(y_true, y_score))

def calibration_data(y_true, y_score, n_bins=10):
    """Bảng calibration: mỗi bin (chia đều [0,1]) có xác suất dự đoán trung bình và tỉ lệ dương thực."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, n_bins - 1)
    df = pd.DataFrame({"bin": idx, "pred": y_score, "y": y_true})
    out = df.groupby("bin").agg(mean_pred=("pred", "mean"), frac_pos=("y", "mean"), count=("y", "size"))
    return out.reset_index()

def threshold_table(y_true, y_score, fn_cost=5.0, fp_cost=1.0, thresholds=None):
    """Quét ngưỡng 0.05 -> 0.95; mỗi dòng: precision/recall/F1, ma trận nhầm lẫn, chi phí
    `fn_cost*FN + fp_cost*FP`. Cột `recommended` đánh dấu ngưỡng chi phí thấp nhất (không mặc định 0.5)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(t), "flag_rate": float(pred.mean()), "precision": precision, "recall": recall,
                     "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": fn_cost * fn + fp_cost * fp})
    tbl = pd.DataFrame(rows)
    tbl["recommended"] = False
    tbl.loc[tbl["cost"].idxmin(), "recommended"] = True
    return tbl

def recommended_threshold(y_true, y_score, fn_cost=5.0, fp_cost=1.0):
    tbl = threshold_table(y_true, y_score, fn_cost, fp_cost)
    return float(tbl.loc[tbl["recommended"], "threshold"].iloc[0])

def summary_at_threshold(y_true, y_score, threshold):
    """Một dòng cho bảng so sánh: AUC, KS, Brier + precision/recall/F1 tại ngưỡng cho trước."""
    pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
    }

def expected_cost_table(y_true, y_score, loss_if_bad, gain_if_good, thresholds=None):
    """Như `threshold_table` nhưng chi phí tính **theo từng hồ sơ bằng số tiền thật**:
    cost(t) = Σ_{FN} loss_if_bad_i + Σ_{FP} gain_if_good_i.
    `loss_if_bad` ~ AMT_CREDIT × LGD (mất gốc khi cho vay khách vỡ nợ);
    `gain_if_good` ~ AMT_CREDIT × r (lãi mất đi khi từ chối nhầm khách tốt)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    loss_if_bad = np.asarray(loss_if_bad, dtype=float)
    gain_if_good = np.asarray(gain_if_good, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fn_mask = (pred == 0) & (y_true == 1)
        fp_mask = (pred == 1) & (y_true == 0)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(fp_mask.sum())
        fn = int(fn_mask.sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(t), "flag_rate": float(pred.mean()), "precision": precision, "recall": recall,
                     "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "cost": float(loss_if_bad[fn_mask].sum() + gain_if_good[fp_mask].sum())})
    tbl = pd.DataFrame(rows)
    tbl["recommended"] = False
    tbl.loc[tbl["cost"].idxmin(), "recommended"] = True
    return tbl

def youden_threshold(y_true, y_score):
    """Ngưỡng tại điểm KS (max TPR − FPR, Youden J) — thuần thống kê, không cần giả định chi phí."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i])

def threshold_for_flag_rate(y_score, flag_rate):
    """Ngưỡng sao cho đúng `flag_rate` phần hồ sơ điểm cao nhất bị chặn — dùng để tái tạo khẩu vị rủi ro
    thực tế (tỉ lệ từ chối lịch sử của tổ chức cho vay) mà không cần giả định chi phí hay LGD."""
    return float(np.quantile(np.asarray(y_score), 1.0 - flag_rate))
