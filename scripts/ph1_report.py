"""báo cáo so sánh 5 model + ensemble, và chấm holdout đúng một lần.

    python scripts/ph1_report.py

Ngưỡng (xem `scripts/ph1_evidence_threshold_basis.py`): chính = chặn đúng tỉ lệ từ chối lịch sử của Home Credit
(`threshold_basis.json`, ~21.9%) — áp cùng tỉ lệ chặn cho mọi model nên P/R/F1 so sánh được; **đối chiếu** = KS/Youden.
Đọc OOF (Task 8/9), `cv_results.csv`, model fold + ensemble. Ghi:
- `outputs/reports/model_comparison.md` — bảng 6 dòng (AUC mean±std, KS, P/R/F1 tại ngưỡng, Brier, giờ train)
  + bảng ngưỡng của model cuối + **số holdout của model cuối**
- `outputs/reports/model_comparison.csv` — cùng bảng, dạng csv (web đọc bản này)
- `outputs/reports/threshold_table.csv`, `outputs/reports/holdout_metrics.json`
- `outputs/plots/ph1_roc.png`, `ph1_calibration.png`, `ph1_score_distribution.png`, `ph1_threshold_tradeoff.png`
- `outputs/oof/holdout_preds.npz` — dự đoán holdout của từng member + ensemble (cho SHAP/stress/dashboard sau)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.config import MODELS_DIR, PLOTS_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import metrics as mt  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling import registry as rg  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402
from src.ph1_credit_risk.modeling.split import load_holdout_ids  # noqa: E402

ENSEMBLE_NAME = "ensemble"
REPORT_PATH = REPORTS_DIR / "model_comparison.md"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"  # web đọc bản này
HOLDOUT_JSON = REPORTS_DIR / "holdout_metrics.json"
BASIS_PATH = REPORTS_DIR / "threshold_basis.json"

COLORS = {"catboost": "#2a78d6", "lightgbm": "#eb6834", "xgboost": "#1baf7a",
          "logistic_regression": "#eda100", "random_forest": "#e87ba4", ENSEMBLE_NAME: "#008300"}
STYLE = dict(lw=2)


def _ax_style(ax):
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def load_oofs() -> tuple[dict[str, np.ndarray], np.ndarray]:
    y = np.load(OOF_DIR / "cv_target.npy")
    names = rg.list_models() + [ENSEMBLE_NAME]
    return {m: np.load(OOF_DIR / f"{m}_oof.npy") for m in names}, y

def comparison_table(oofs, y, cv, flag_rate) -> pd.DataFrame:
    rows = []
    for m, p in oofs.items():
        t = mt.threshold_for_flag_rate(p, flag_rate)
        s = mt.summary_at_threshold(y, p, t)
        r = cv[cv["model"] == m]
        rows.append({"model": m, "auc_mean": s["auc"] if m == ENSEMBLE_NAME else float(r["auc_mean"].iloc[0]),
                     "auc_std": float("nan") if m == ENSEMBLE_NAME else float(r["auc_std"].iloc[0]),
                     "oof_auc": s["auc"], "ks": s["ks"], "threshold": t, "precision": s["precision"],
                     "recall": s["recall"], "f1": s["f1"], "brier": s["brier"],
                     "fit_seconds": float("nan") if m == ENSEMBLE_NAME else float(r["fit_seconds"].iloc[0])})
    return pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)

def plot_roc(oofs, y, table):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m in table["model"]:
        fpr, tpr, _ = roc_curve(y, oofs[m])
        auc = table.loc[table["model"] == m, "oof_auc"].iloc[0]
        ax.plot(fpr, tpr, color=COLORS[m], label=f"{m} (AUC {auc:.4f})", **STYLE)
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — OOF 5-fold, phần CV 80%")
    ax.legend(frameon=False, fontsize=9)
    _ax_style(ax)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "ph1_roc.png", dpi=150); plt.close(fig)

def plot_calibration(oofs, y, table):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--", label="hiệu chỉnh hoàn hảo")
    for m in table["model"]:
        cal = mt.calibration_data(y, oofs[m], n_bins=10)
        brier = table.loc[table["model"] == m, "brier"].iloc[0]
        ax.plot(cal["mean_pred"], cal["frac_pos"], marker="o", ms=5, color=COLORS[m],
                label=f"{m} (Brier {brier:.4f})", **STYLE)
    ax.set(xlabel="Xác suất dự đoán (trung bình bin)", ylabel="Tỉ lệ vỡ nợ thực", title="Calibration — OOF, 10 bin")
    ax.legend(frameon=False, fontsize=9)
    _ax_style(ax)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "ph1_calibration.png", dpi=150); plt.close(fig)

def plot_score_distribution(p, y, threshold, name):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, density=True, alpha=0.55, color="#2a78d6", label="TARGET=0 (trả đúng hạn)")
    ax.hist(p[y == 1], bins=bins, density=True, alpha=0.55, color="#eb6834", label="TARGET=1 (vỡ nợ)")
    ax.axvline(threshold, color="#0b0b0b", lw=1.5, ls="--", label=f"ngưỡng khuyến nghị {threshold:.2f}")
    ax.set(xlabel=f"Điểm OOF của {name}", ylabel="Mật độ", title=f"Phân phối điểm theo lớp — {name}")
    ax.legend(frameon=False, fontsize=9)
    _ax_style(ax)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "ph1_score_distribution.png", dpi=150); plt.close(fig)

def plot_threshold_tradeoff(y, p, flag_rate, t_main, t_ks, name):
    grid = np.round(np.arange(0.02, 0.951, 0.01), 2)
    tbl = mt.threshold_table(y, p, thresholds=grid)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(tbl["flag_rate"], tbl["precision"], color="#2a78d6", label="precision", **STYLE)
    ax.plot(tbl["flag_rate"], tbl["recall"], color="#eb6834", label="recall", **STYLE)
    ax.axvline(flag_rate, color="#0b0b0b", lw=1.5, ls="--", label=f"từ chối lịch sử {flag_rate:.1%} → ngưỡng {t_main:.3f}")
    ax.axvline((p >= t_ks).mean(), color="#52514e", lw=1.2, ls=":", label=f"KS/Youden → ngưỡng {t_ks:.3f}")
    ax.set(xlabel="Tỉ lệ hồ sơ bị chặn", ylabel="Giá trị", title=f"Precision / recall theo tỉ lệ chặn — {name} (OOF)", xlim=(0, 0.6))
    ax.legend(frameon=False, fontsize=9, loc="center right")
    _ax_style(ax)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "ph1_threshold_tradeoff.png", dpi=150); plt.close(fig)

def score_holdout(ensemble, feats, cat) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ids = set(load_holdout_ids())
    df = pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED) + feats)
    ho = df[df["SK_ID_CURR"].isin(ids)]
    assert len(ho) == len(ids), (len(ho), len(ids))
    X, y = ho[feats], ho["TARGET"].to_numpy()
    preds = {}
    for m in ensemble.members:
        folds = sorted(MODELS_DIR.glob(f"ph1_{m}_fold*.joblib"))
        preds[m] = np.mean([rg.predict_proba(joblib.load(f), X) for f in folds], axis=0)
        print(f"  [holdout] {m}: trung bình {len(folds)} fold", flush=True)
    preds[ENSEMBLE_NAME] = ensemble.combine(preds)
    return preds, y, ho["SK_ID_CURR"].to_numpy()

def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    basis = json.loads(BASIS_PATH.read_text())
    flag_rate = float(basis["flag_rate"])

    oofs, y = load_oofs()
    cv = pd.read_csv(REPORTS_DIR / "cv_results.csv")
    table = comparison_table(oofs, y, cv, flag_rate)

    table.to_csv(COMPARISON_CSV, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ensemble = joblib.load(MODELS_DIR / "ph1_ensemble.joblib")
    final = ENSEMBLE_NAME
    threshold = float(table.loc[table["model"] == final, "threshold"].iloc[0])
    t_ks = mt.youden_threshold(y, oofs[final])
    ks_row = mt.summary_at_threshold(y, oofs[final], t_ks)
    grid = sorted(set(np.round(np.arange(0.05, 0.951, 0.05), 2)) | {round(threshold, 4), round(t_ks, 4)})
    thr_tbl = mt.threshold_table(y, oofs[final], thresholds=grid).drop(columns=["cost", "recommended"])
    thr_tbl["basis"] = ""
    thr_tbl.loc[np.isclose(thr_tbl["threshold"], threshold, atol=1e-4), "basis"] = "CHÍNH: từ chối lịch sử"
    thr_tbl.loc[np.isclose(thr_tbl["threshold"], t_ks, atol=1e-4), "basis"] = "đối chiếu: KS/Youden"
    thr_tbl.to_csv(REPORTS_DIR / "threshold_table.csv", index=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc(oofs, y, table); plot_calibration(oofs, y, table)
    plot_score_distribution(oofs[final], y, threshold, final)
    plot_threshold_tradeoff(y, oofs[final], flag_rate, threshold, t_ks, final)

    # holdout
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH
    cat = [c for c in json.loads(CATEGORICAL_COLUMNS_PATH.read_text()) if c in feats]
    print("[report] chấm holdout (một lần)...", flush=True)
    ho_preds, y_ho, ids_ho = score_holdout(ensemble, feats, cat)
    ho = mt.summary_at_threshold(y_ho, ho_preds[final], threshold)
    ho.update({"n": int(len(y_ho)), "pos_rate": float(y_ho.mean()), "model": f"{ensemble.method} {ensemble.members}",
               "oof_auc": float(table.loc[table["model"] == final, "oof_auc"].iloc[0]),
               "scored_at": time.strftime("%Y-%m-%d %H:%M")})
    HOLDOUT_JSON.write_text(json.dumps(ho, indent=1))
    np.savez(OOF_DIR / "holdout_preds.npz", ids=ids_ho, y=y_ho, **ho_preds)
    print(f"[report] HOLDOUT {final}: AUC {ho['auc']:.4f}  KS {ho['ks']:.4f}  Brier {ho['brier']:.4f}  "
          f"P {ho['precision']:.3f} R {ho['recall']:.3f} F1 {ho['f1']:.3f} @ {threshold:.2f}")

    # markdown
    fmt = lambda v: f"{v:.4f}"
    show = table.copy()
    show["auc_cv"] = [f"{a:.4f} ± {s:.4f}" if not np.isnan(s) else f"{a:.4f} (OOF CV lồng)" for a, s in zip(show["auc_mean"], show["auc_std"])]
    show["fit_seconds"] = show["fit_seconds"].map(lambda v: "—" if np.isnan(v) else f"{v:.0f}")
    md_table = show[["model", "auc_cv", "oof_auc", "ks", "threshold", "precision", "recall", "f1", "brier", "fit_seconds"]] \
        .to_markdown(index=False, floatfmt=".4f")
    thr_show = thr_tbl[["threshold", "flag_rate", "precision", "recall", "f1", "fp", "fn", "basis"]].to_markdown(index=False, floatfmt=".3f")
    ref = basis["refusal"]
    lines = [
        "# So sánh model — PH1 Home Credit\n",
        f"Phần CV 80% ({len(y):,} dòng, {len(feats)} feature), Stratified 5-fold, OOF.\n",
        "## Cơ sở ngưỡng\n",
        f"**Ngưỡng chính — khẩu vị rủi ro thực tế:** Home Credit đã từ chối **{ref['refusal_rate']:.1%}** trong "
        f"{ref['n_decided']:,} hồ sơ lịch sử có kết quả (`previous_application`, Refused/(Approved+Refused); cash loans "
        f"{ref['refusal_rate_by_type']['Cash loans']:.1%}, consumer {ref['refusal_rate_by_type']['Consumer loans']:.1%}). "
        "Ngưỡng của mỗi model = mức chặn đúng tỉ lệ đó trên OOF của chính nó → mọi model cùng tỉ lệ chặn, P/R/F1 so sánh "
        "được. Không cần giả định chi phí FN:FP hay LGD. Không dùng 0.5.\n",
        f"**Đối chiếu — KS/Youden** (max TPR−FPR, thuần thống kê): ngưỡng {t_ks:.3f}, chặn {(oofs[final] >= t_ks).mean():.1%}, "
        f"P {ks_row['precision']:.3f} / R {ks_row['recall']:.3f} / F1 {ks_row['f1']:.3f}.\n",
        "Phân tích phụ (chi phí theo số tiền thật với r từ dữ liệu và LGD quét 0.3–1.0) nằm ở "
        "`outputs/evidence/task10-threshold-basis/`: ngưỡng ra 0.29–0.59 tùy LGD — quá nhạy với một tham số không có trong "
        "dữ liệu nên không dùng làm ngưỡng chính.\n",
        "## Bảng so sánh (OOF)\n", md_table, "",
        f"Ensemble = `{ensemble.method}` trên {ensemble.members} (Task 9). Thời gian train là tổng 5 fold (Task 8).\n",
        "## Holdout 20% — chấm đúng một lần\n",
        f"**Đây là con số duy nhất chưa từng dùng để chọn feature, model, ngưỡng hay cách trộn.** "
        f"Holdout {ho['n']:,} dòng (tỉ lệ vỡ nợ {ho['pos_rate']:.3%}), model cuối `{ho['model']}`, "
        f"mỗi member dự đoán bằng trung bình 5 model fold; ngưỡng {threshold:.3f} (chặn {flag_rate:.1%}) chốt trên OOF.\n",
        "| | AUC | KS | Brier | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|",
        f"| OOF (CV) | {ho['oof_auc']:.4f} | {table.loc[table['model'] == final, 'ks'].iloc[0]:.4f} | "
        f"{table.loc[table['model'] == final, 'brier'].iloc[0]:.4f} | {table.loc[table['model'] == final, 'precision'].iloc[0]:.4f} | "
        f"{table.loc[table['model'] == final, 'recall'].iloc[0]:.4f} | {table.loc[table['model'] == final, 'f1'].iloc[0]:.4f} |",
        f"| **Holdout** | **{ho['auc']:.4f}** | {ho['ks']:.4f} | {ho['brier']:.4f} | {ho['precision']:.4f} | {ho['recall']:.4f} | {ho['f1']:.4f} |",
        "", f"Chênh AUC holdout − OOF = {ho['auc'] - ho['oof_auc']:+.4f}.\n",
        f"## Bảng ngưỡng — {final} (OOF)\n", thr_show, "",
        "## Nhận xét\n",
        "- 5 model đơn train với `scale_pos_weight` / `class_weight=\"balanced\"` nên xác suất bị đẩy lên (calibration "
        "curve nằm dưới đường chéo, Brier 0.16–0.20, ngưỡng cùng tỉ lệ chặn rơi ở 0.6–0.7). Xếp hạng (AUC/KS) và P/R tại "
        "cùng tỉ lệ chặn không bị ảnh hưởng.",
        "- Ensemble stacking là LR **không** weight trên logit OOF nên tự hiệu chỉnh lại: Brier 0.066, calibration gần "
        "đường chéo (bin cuối ít mẫu nên nhiễu). Đây là model duy nhất có xác suất dùng được trực tiếp cho stress-test / dashboard.",
        f"- Ở mức chặn {flag_rate:.1%}: precision ~{table.loc[table['model'] == final, 'precision'].iloc[0]:.2f}, recall "
        f"~{table.loc[table['model'] == final, 'recall'].iloc[0]:.2f} — cứ ~5 hồ sơ bị chặn có 1 vỡ nợ thật (nền 8%), "
        "bắt được ~60% số vỡ nợ.\n",
        "## Plots\n",
        "- `outputs/plots/ph1_roc.png` — ROC 6 đường (OOF)",
        "- `outputs/plots/ph1_calibration.png` — calibration curve 6 model, 10 bin",
        "- `outputs/plots/ph1_score_distribution.png` — phân phối điểm theo lớp của ensemble + ngưỡng",
        "- `outputs/plots/ph1_threshold_tradeoff.png` — precision/recall theo tỉ lệ chặn của ensemble, đánh dấu 2 ngưỡng\n",
        f"Sinh bởi `scripts/ph1_report.py` lúc {time.strftime('%Y-%m-%d %H:%M')}.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[report] ghi {REPORT_PATH}, {HOLDOUT_JSON}, plots -> {PLOTS_DIR}")

if __name__ == "__main__":
    main()
