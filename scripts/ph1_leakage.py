"""kiểm rò rỉ ở cấp feature.

    python scripts/ph1_leakage.py

Ba phép kiểm: 
(1) thời gian — cột sự kiện thực có giá trị sau ngày nộp đơn; 
(2) AUC đơn biến so với EXT_SOURCE_MEAN; 
(3) ablation LightGBM 5-fold, cùng fold, so AUC từng fold theo cặp. Chỉ dùng phần CV 80%.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import OUTPUT_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.data import loader  # noqa: E402
from src.ph1_credit_risk.evaluation import leakage as lk  # noqa: E402
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling.cv import run_cv  # noqa: E402
from src.ph1_credit_risk.modeling.split import cv_portion  # noqa: E402

EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task15-leakage"
REPORT_PATH = REPORTS_DIR / "leakage.md"
SUSPECT = "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX"

def audit_all(feats: list[str]) -> pd.DataFrame:
    parts = []
    for table, cols in lk.DATE_COLUMNS.items():
        df = loader.load_table(table.lower(), columns=list(cols))
        a = lk.temporal_audit(table, df)
        a["features_in_model"] = [", ".join(lk.features_from_column(table, c, feats)) for c in a["column"]]
        parts.append(a)
        print(f"[leakage] {table}: {len(df):,} dòng", flush=True)
    return pd.concat(parts, ignore_index=True)

def bucket_default_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Tỉ lệ vỡ nợ theo SUSPECT: NaN (chưa vay HC) / <= 0 (mọi khoản đã hết lịch) / > 0 (còn khoản đang chạy)."""
    s = df[SUSPECT]
    grp = np.select([s.isna(), s <= 0], ["không có đơn cũ", "<= 0: mọi khoản đã hết lịch"], "> 0: còn khoản đang chạy")
    return df.groupby(grp)["TARGET"].agg(n="size", default_rate="mean").reset_index(names="nhom")

def main() -> None:
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    cat = [c for c in json.loads(CATEGORICAL_COLUMNS_PATH.read_text()) if c in feats]
    df = cv_portion(pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED) + feats))
    y = df["TARGET"].to_numpy()

    audit = audit_all(feats)
    uni = lk.univariate_auc(df[feats], y, skip=cat)
    ref = float(uni.loc[uni["feature"] == "EXT_SOURCE_MEAN", "strength"].iloc[0])
    uni["above_ext_source_mean"] = (uni["strength"] > ref) & (uni["feature"] != "EXT_SOURCE_MEAN")
    buckets = bucket_default_rate(df)

    leaked = sorted({f for s in audit.loc[audit["verdict"] == "RÒ RỈ", "features_in_model"] for f in s.split(", ") if f})
    groups = {
        "base": [],
        "drop_last_due_1st_version": [f for f in feats if "DAYS_LAST_DUE_1ST_VERSION" in f],
        "drop_all_prev_days_due": [f for f in feats if f.startswith("PREV") and "DAYS_LAST_DUE" in f],
    }
    if leaked:
        groups["drop_leaked"] = leaked
    folds = {}
    for name, drop in groups.items():
        cols = [f for f in feats if f not in drop]
        res = run_cv("lightgbm", df[cols], df["TARGET"], [c for c in cat if c in cols])
        folds[name] = res.fold_scores
        print(f"[leakage] {name:>28} bỏ {len(drop)} cột  AUC {res.mean_auc:.4f}", flush=True)
    base = np.array(folds["base"])
    abl = pd.DataFrame([{"config": k, "n_dropped": len(groups[k]), "auc_mean": float(np.mean(v)),
                         "delta_mean": float(np.mean(np.array(v) - base)), "delta_std": float(np.std(np.array(v) - base)),
                         "dropped": ", ".join(groups[k])} for k, v in folds.items()])

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(EVIDENCE_DIR / "temporal_audit.csv", index=False)
    uni.to_csv(EVIDENCE_DIR / "univariate_auc.csv", index=False)
    abl.to_csv(EVIDENCE_DIR / "results.csv", index=False)
    (EVIDENCE_DIR / "config.json").write_text(json.dumps(
        {"data": "cv_portion(features_train) 80%, holdout không chạm", "model": "lightgbm registry, 5-fold, cùng seed",
         "reference_univariate": f"EXT_SOURCE_MEAN strength {ref:.4f}", "groups": groups}, indent=1, ensure_ascii=False))

    verdict = "CÓ RÒ RỈ trong feature của model" if leaked else "KHÔNG phát hiện rò rỉ thời gian trong 80 feature"
    audit_display = audit.drop(columns=["n"]).copy()
    audit_display["share_future"] = audit_display["share_future"].map(lambda v: f"{v:.6f}")
    fmts = [".6f" if c == "share_future" else ".4f" for c in audit_display.columns]
    md = [
        "# Kiểm rò rỉ feature (Task 15)\n",
        f"**Kết luận:** {verdict}.\n",
        "## 1. Kiểm thời gian — cột sự kiện thực có giá trị sau ngày nộp đơn\n",
        audit_display.to_markdown(index=False, floatfmt=fmts), "",
        f"## 2. AUC đơn biến (mốc: EXT_SOURCE_MEAN = {ref:.4f})\n",
        uni.head(15).to_markdown(index=False, floatfmt=".4f"), "",
        f"## 3. Vì sao `{SUSPECT}` mạnh — tỉ lệ vỡ nợ theo nhóm\n",
        buckets.to_markdown(index=False, floatfmt=".4f"), "",
        "## 4. Ablation — LightGBM 5-fold, cùng fold, chênh AUC theo cặp so với base\n",
        abl.drop(columns=["dropped"]).to_markdown(index=False, floatfmt=".4f"), "",
        "Giới hạn: không kiểm được thời điểm tạo `EXT_SOURCE_*` (nguồn ngoài, không có mô tả thời gian).",
    ]
    REPORT_PATH.write_text("\n".join(md) + "\n")
    (EVIDENCE_DIR / "result.md").write_text("\n".join(md) + "\n")
    print(f"[leakage] {verdict}. Ghi {REPORT_PATH}, {EVIDENCE_DIR}")

if __name__ == "__main__":
    main()
