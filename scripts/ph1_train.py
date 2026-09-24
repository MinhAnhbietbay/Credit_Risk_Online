"""
    python scripts/ph1_train.py --models all
    python scripts/ph1_train.py --models lightgbm,catboost

Ghi:
- `data/processed/holdout_ids.json`      — SK_ID_CURR của holdout
- `outputs/oof/<model>_oof.npy`          — OOF theo thứ tự `outputs/oof/cv_ids.npy` (+ `cv_target.npy`)
- `outputs/models/ph1_<model>_fold<k>.joblib`
- `outputs/reports/cv_results.csv|md`    — AUC mean ± std, thời gian từng model (số liệu chính thức)
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import CV_FOLDS, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling import registry as rg  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR, run_cv, save_cv_result  # noqa: E402
from src.ph1_credit_risk.modeling.split import HOLDOUT_IDS_PATH, make_holdout_split, save_holdout_ids  # noqa: E402

CV_RESULTS_CSV = REPORTS_DIR / "cv_results.csv"
CV_RESULTS_MD = REPORTS_DIR / "cv_results.md"

def load_cv_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    cat = [c for c in json.loads(CATEGORICAL_COLUMNS_PATH.read_text()) if c in feats]
    df = pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED) + feats)
    cv_df, ho_df = make_holdout_split(df)
    save_holdout_ids(ho_df)
    print(f"[train] features_train {df.shape} -> CV {len(cv_df)} / holdout {len(ho_df)} "
          f"(holdout ids -> {HOLDOUT_IDS_PATH}); {len(feats)} feature, {len(cat)} categorical")
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / "cv_ids.npy", cv_df["SK_ID_CURR"].to_numpy())
    np.save(OOF_DIR / "cv_target.npy", cv_df["TARGET"].to_numpy())
    return cv_df[feats], cv_df["TARGET"], cat, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="all hoặc danh sách tên cách nhau bằng dấu phẩy")
    ap.add_argument("--n-splits", type=int, default=CV_FOLDS)
    args = ap.parse_args()
    names = rg.list_models() if args.models == "all" else [n.strip() for n in args.models.split(",")]
    for n in names:
        rg.get_model(n)  # fail sớm nếu tên sai

    X, y, cat, feats = load_cv_data()
    rows = []
    for name in names:
        print(f"[train] {name}: {args.n_splits}-fold CV trên {len(X)} dòng", flush=True)
        res = run_cv(name, X, y, cat, n_splits=args.n_splits, verbose=True)
        save_cv_result(res)
        rows.append({
            "model": name, "auc_mean": res.mean_auc, "auc_std": res.std_auc,
            "oof_auc": float(roc_auc_score(y, res.oof_pred)),
            **{f"fold{k}": s for k, s in enumerate(res.fold_scores)},
            "fit_seconds": res.fit_seconds,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        })
        print(f"[train] {name}: AUC {res.mean_auc:.4f} ± {res.std_auc:.4f}  OOF {rows[-1]['oof_auc']:.4f}  "
              f"{res.fit_seconds:.0f}s", flush=True)
        table = write_results(rows, names, args, len(X), len(feats))  # ghi sau mỗi model, không mất nếu model sau lỗi
    print("\n" + table)
    print(f"\n[train] ghi {CV_RESULTS_CSV}, {CV_RESULTS_MD}")


def write_results(rows: list[dict], names: list[str], args, n_rows: int, n_feats: int) -> str:
    res_df = pd.DataFrame(rows)
    # gộp với kết quả model chạy trước đó (khi --models chỉ định một phần / chạy dở)
    if CV_RESULTS_CSV.exists():
        old = pd.read_csv(CV_RESULTS_CSV)
        res_df = pd.concat([old[~old["model"].isin(res_df["model"])], res_df])
    res_df = res_df.sort_values("auc_mean", ascending=False)
    res_df.to_csv(CV_RESULTS_CSV, index=False)
    table = res_df[["model", "auc_mean", "auc_std", "oof_auc", "fit_seconds"]].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    CV_RESULTS_MD.write_text(
        f"# CV {args.n_splits}-fold — phần CV 80% ({n_rows} dòng, {n_feats} feature)\n\n"
        f"Holdout {HOLDOUT_IDS_PATH.name} không chạm. `auc_mean ± auc_std` qua fold; `oof_auc` trên toàn OOF.\n\n"
        f"```\n{table}\n```\n\nCập nhật bởi `scripts/ph1_train.py --models {args.models}` lúc {time.strftime('%Y-%m-%d %H:%M')}.\n")
    return table

if __name__ == "__main__":
    main()
