"""SHAP cho model cây thành phần tốt nhất (CatBoost).

    python scripts/ph1_shap.py --sample 5000
Cách chạy:
- Mẫu lấy từ **phần CV** (không chạm holdout), phân tầng theo TARGET, mặc định 5,000 dòng.
- SHAP **out-of-fold**: mỗi dòng giải thích bằng đúng model fold không train trên nó; fold dựng lại
  bằng `StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)`.

Ghi:
- `outputs/reports/shap_global.csv` — xếp hạng đầy đủ theo |SHAP| trung bình + mô tả 
- `outputs/reports/shap.md` — top-20 + một hồ sơ rủi ro cao
- `outputs/plots/ph1_shap_beeswarm.png`, `ph1_shap_bar.png`, `ph1_shap_waterfall.png`
- `outputs/evidence/task11-oof-vs-fold0-shap/` — OOF SHAP so với chỉ dùng fold 0
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
import shap  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from src.config import CV_FOLDS, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, RANDOM_SEED, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.explain import shap_explain as se  # noqa: E402
from src.ph1_credit_risk.explain.glossary import describe  # noqa: E402
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling.split import cv_portion  # noqa: E402

MODEL_NAME = "catboost"
GLOBAL_CSV = REPORTS_DIR / "shap_global.csv"
REPORT_MD = REPORTS_DIR / "shap.md"
EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task11-oof-vs-fold0-shap"
TOP_N = 20

def load_cv_frame() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    cat = [c for c in json.loads(CATEGORICAL_COLUMNS_PATH.read_text()) if c in feats]
    df = pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED) + feats)
    cv_df = cv_portion(df)
    return cv_df[feats], cv_df["TARGET"], cat

def sample_rows(X: pd.DataFrame, y: pd.Series, n: int) -> np.ndarray:
    """Positional của mẫu phân tầng theo TARGET trong phần CV."""
    if n >= len(X):
        return np.arange(len(X))
    pos, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=RANDOM_SEED)
    return np.sort(pos)

def fold_of_each_row(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold = np.full(len(X), -1, dtype=int)
    for k, (_, va) in enumerate(skf.split(X, y)):
        fold[va] = k
    assert (fold >= 0).all(), "Có dòng không thuộc fold nào"
    return fold

def oof_shap_on_sample(models, X_sample: pd.DataFrame, fold_sample: np.ndarray):
    """SHAP out-of-fold cho riêng mẫu: mỗi nhóm dòng dùng model fold tương ứng."""
    values = np.full(X_sample.shape, np.nan, dtype=np.float64)
    bases = []
    for k, model in enumerate(models):
        idx = np.where(fold_sample == k)[0]
        if len(idx) == 0:
            bases.append(np.nan)
            continue
        v, b = se.shap_values(model, X_sample.iloc[idx])
        values[idx] = v
        bases.append(b)
    if np.isnan(values).any():
        raise ValueError("SHAP out-of-fold còn dòng trống")
    return values, bases

# plot

def _note(fig):
    fig.text(0.01, 0.005, se.SHAP_MODEL_NOTE, fontsize=6.5, color="#5c5b55", wrap=True)

def plot_beeswarm(values, X_sample, path):
    plt.figure(figsize=(9, 8))
    shap.summary_plot(values, X_sample, max_display=TOP_N, show=False, plot_size=None)
    plt.title(f"SHAP beeswarm — {MODEL_NAME} (out-of-fold, {len(X_sample):,} hồ sơ)", fontsize=11)
    _note(plt.gcf())
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_bar(imp: pd.DataFrame, path):
    top = imp.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2a78d6")
    ax.set_xlabel("|SHAP| trung bình (log-odds)")
    ax.set_title(f"Top {TOP_N} feature — {MODEL_NAME} (SHAP out-of-fold)", fontsize=11)
    ax.grid(True, axis="x", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_waterfall(one, X_row, path):
    expl = shap.Explanation(
        values=one["shap_values"], base_values=one["base_value"],
        data=X_row.iloc[0].to_numpy(), feature_names=list(X_row.columns),
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.title(f"Hồ sơ rủi ro cao nhất trong mẫu — điểm CatBoost {one['probability']:.1%} "
              f"(mức nền {one['base_probability']:.1%}; điểm chưa hiệu chỉnh, không phải PD)", fontsize=10)
    _note(plt.gcf())
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# báo cáo 

def write_report(imp, one, n_sample, folds_used, ev):
    lines = [
        "# SHAP — giải thích model (Task 11)",
        "",
        f"> **{se.SHAP_MODEL_NOTE}**",
        "",
        f"Mẫu: **{n_sample:,} hồ sơ** lấy phân tầng theo TARGET từ phần CV (không chạm holdout). "
        f"SHAP **out-of-fold**: mỗi hồ sơ được giải thích bằng đúng 1 trong {folds_used} model fold "
        "không train trên hồ sơ đó. Đơn vị: log-odds, cộng được về dự đoán của model.",
        f"## Top {TOP_N} feature theo |SHAP| trung bình",
        "| # | Feature | \\|SHAP\\| TB | Ý nghĩa |",
        "|---:|---|---:|---|",
    ]
    for i, r in imp.head(TOP_N).iterrows():
        lines.append(f"| {i + 1} | `{r['feature']}` | {r['mean_abs_shap']:.4f} | {r['mo_ta']} |")

    lines += [
        "", f"Xếp hạng đầy đủ {len(imp)} feature: `outputs/reports/shap_global.csv`.",
        "", "## Giải thích một hồ sơ (local)", "",
        f"Hồ sơ rủi ro cao nhất trong mẫu: **điểm CatBoost {one['probability']:.1%}**, "
        f"mức nền của model {one['base_probability']:.1%}. Mức nền cao hơn tỉ lệ vỡ nợ thật (~8%) vì "
        "CatBoost fit với `scale_pos_weight` — đây là **điểm xếp hạng, không phải PD**; PD cho quyết định "
        "lấy từ ensemble năm yếu tố đẩy mạnh nhất:",
        "",
    ]
    for i, f in enumerate(one["factors"], 1):
        lines.append(f"{i}. {f['sentence']}")

    lines += [
        "", "Biểu đồ waterfall: `outputs/plots/ph1_shap_waterfall.png`.",
        "", "## Plot", "",
        "- `outputs/plots/ph1_shap_beeswarm.png` — beeswarm top-20",
        "- `outputs/plots/ph1_shap_bar.png` — |SHAP| trung bình top-20",
        "- `outputs/plots/ph1_shap_waterfall.png` — waterfall hồ sơ ở trên",
        "- **`EXT_SOURCE_*` chi phối model** (|SHAP| lớn hơn phần còn lại một bậc) nhưng là điểm hộp đen từ "
        "nguồn ngoài: không biết nó được tính từ gì, nên không kiểm tra được bản thân nó có thiên lệch hay không.",
        "", "## Bằng chứng cho cách tính SHAP", "",
        f"SHAP out-of-fold (5 model fold) so với chỉ dùng model fold 0: trùng "
        f"**{ev['overlap_top20']}/{TOP_N}** feature ở top-20, tương quan Spearman toàn bảng "
        f"**{ev['spearman']:.4f}**, lệch |SHAP| trung bình lớn nhất **{ev['max_abs_diff']:.4f}** log-odds. "
        "Chi tiết: `outputs/evidence/task11-oof-vs-fold0-shap/`.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

def write_evidence(imp_oof, imp_fold0):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    merged = imp_oof.merge(imp_fold0, on="feature", suffixes=("_oof", "_fold0"))
    merged["diff"] = merged["mean_abs_shap_oof"] - merged["mean_abs_shap_fold0"]
    merged[["feature", "mean_abs_shap_oof", "mean_abs_shap_fold0", "diff"]].to_csv(
        EVIDENCE_DIR / "importance_oof_vs_fold0.csv", index=False)

    top_oof = set(imp_oof.head(TOP_N)["feature"])
    top_f0 = set(imp_fold0.head(TOP_N)["feature"])
    ev = {
        "overlap_top20": len(top_oof & top_f0),
        "spearman": float(merged["mean_abs_shap_oof"].corr(merged["mean_abs_shap_fold0"], method="spearman")),
        "max_abs_diff": float(merged["diff"].abs().max()),
        "only_in_oof_top20": sorted(top_oof - top_f0),
        "only_in_fold0_top20": sorted(top_f0 - top_oof),
    }
    (EVIDENCE_DIR / "result.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    (EVIDENCE_DIR / "result.md").write_text(
        f"**Số:** trùng {ev['overlap_top20']}/{TOP_N} feature ở top-20, Spearman {ev['spearman']:.4f}, "
        f"lệch lớn nhất {ev['max_abs_diff']:.4f} log-odds.",
        encoding="utf-8")
    return ev

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5000, help="số hồ sơ cho beeswarm (plan: <= 5000)")
    args = ap.parse_args()

    t0 = time.time()
    X, y, cat = load_cv_frame()
    print(f"[shap] phần CV {X.shape}, {len(cat)} categorical")

    models = [joblib.load(MODELS_DIR / f"ph1_{MODEL_NAME}_fold{k}.joblib") for k in range(CV_FOLDS)]
    fold = fold_of_each_row(X, y)
    pos = sample_rows(X, y, args.sample)
    X_s, fold_s = X.iloc[pos], fold[pos]
    print(f"[shap] mẫu {len(X_s):,} hồ sơ, tỉ lệ vỡ nợ {y.iloc[pos].mean():.4f}; "
          f"phân bố theo fold {np.bincount(fold_s, minlength=CV_FOLDS).tolist()}")

    values, bases = oof_shap_on_sample(models, X_s, fold_s)
    print(f"[shap] xong SHAP out-of-fold sau {time.time() - t0:.0f}s; "
          f"base value từng fold {[round(b, 4) for b in bases]}")

    imp = se.global_importance(values, list(X_s.columns))
    imp.to_csv(GLOBAL_CSV, index=False)

    v0, _ = se.shap_values(models[0], X_s)
    ev = write_evidence(imp, se.global_importance(v0, list(X_s.columns)))

    # hồ sơ rủi ro cao nhất trong mẫu, giải thích bằng model fold của nó
    margin = values.sum(axis=1) + np.array([bases[k] for k in fold_s])
    worst = int(np.argmax(margin))
    row = X_s.iloc[[worst]]
    one = se.explain_one(models[fold_s[worst]], row, k=5)

    plot_beeswarm(values, X_s, PLOTS_DIR / "ph1_shap_beeswarm.png")
    plot_bar(imp, PLOTS_DIR / "ph1_shap_bar.png")
    plot_waterfall(one, row, PLOTS_DIR / "ph1_shap_waterfall.png")
    write_report(imp, one, len(X_s), CV_FOLDS, ev)

    print(f"\n[shap] TOP 10 ({MODEL_NAME}, out-of-fold):")
    print(imp.head(10)[["feature", "mean_abs_shap"]].to_string(index=False))
    print(f"\n[shap] hồ sơ rủi ro cao nhất: điểm CatBoost {one['probability']:.1%} "
          f"(nền {one['base_probability']:.1%}; chưa hiệu chỉnh, không phải PD)")
    for f in one["factors"]:
        print("  -", f["sentence"])
    print(f"\n[shap] -> {REPORT_MD}, {GLOBAL_CSV}, 3 plot, {EVIDENCE_DIR}  ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
