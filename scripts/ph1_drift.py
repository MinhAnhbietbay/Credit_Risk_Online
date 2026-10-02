"""PSI theo feature cho hai cặp — phần CV vs holdout (kiểm phép tách) và toàn bộ train vs
application_test (lệch thật) — cộng PSI của điểm model.

    python scripts/ph1_drift.py
"""
from __future__ import annotations

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

from src.config import MODELS_DIR, PLOTS_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import drift as dr  # noqa: E402
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH, features_path  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402
from src.ph1_credit_risk.modeling.scoring import predict_ensemble  # noqa: E402
from src.ph1_credit_risk.modeling.split import load_holdout_ids  # noqa: E402

REPORT_MD, REPORT_CSV = REPORTS_DIR / "psi.md", REPORTS_DIR / "psi.csv"

def main() -> None:
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    cat = [c for c in json.loads(CATEGORICAL_COLUMNS_PATH.read_text()) if c in feats]
    train = pd.read_parquet(FEATURES_TRAIN_PATH, columns=["SK_ID_CURR"] + feats)
    test = pd.read_parquet(features_path("test"), columns=["SK_ID_CURR"] + feats)
    ho_ids = set(load_holdout_ids())
    cv, ho = train[~train["SK_ID_CURR"].isin(ho_ids)], train[train["SK_ID_CURR"].isin(ho_ids)]

    split_tbl = dr.psi_table(cv, ho, feats, cat).assign(pair="cv_vs_holdout")
    test_tbl = dr.psi_table(train, test, feats, cat).assign(pair="train_vs_application_test")

    # Bất biến không nhìn thấy được trong báo cáo: nếu NaN bị bỏ im lặng thì PSI chỉ thấp hơn thực tế,
    # mà "thấp" lại đọc thành "ổn định" — sai theo hướng làm người đọc yên tâm. Feature nào đổi tỉ lệ
    # NaN quá 1 điểm % thì PSI của nó buộc phải > 0.
    moved = (test[feats].isna().mean() - train[feats].isna().mean()).abs()
    changed = moved[moved > 0.01].index
    psi_of = test_tbl.set_index("feature")["psi"]
    assert (psi_of.loc[changed] > 0).all(), f"NaN không được tính như một bin: {list(changed[psi_of.loc[changed] <= 0])}"
    worst = moved.idxmax()
    print(f"[psi] NaN tính như một bin ✓ — {len(changed)}/{len(feats)} feature đổi tỉ lệ NaN >1 điểm %, "
          f"lớn nhất {worst} {train[worst].isna().mean():.3f}→{test[worst].isna().mean():.3f}", flush=True)

    ens = joblib.load(MODELS_DIR / "ph1_ensemble.joblib")
    s_train = np.r_[np.load(OOF_DIR / "ensemble_oof.npy"), np.load(OOF_DIR / "holdout_preds.npz")["ensemble"]]
    s_test = predict_ensemble(ens, test[feats])["ensemble"]
    score_psi = dr.psi(pd.Series(s_train), pd.Series(s_test))
    print(f"[psi] điểm model train vs test: {score_psi:.4f} ({dr.psi_level(score_psi)})", flush=True)

    both = pd.concat([split_tbl, test_tbl], ignore_index=True)
    both.to_csv(REPORT_CSV, index=False)

    top = test_tbl.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["psi"], color="#2a78d6")
    for x, ls in ((0.1, "--"), (0.25, ":")):
        ax.axvline(x, color="#52514e", lw=1, ls=ls)
    ax.set(xlabel="PSI (mốc tham chiếu 0.1 / 0.25)", title="15 feature lệch nhiều nhất — train vs application_test")
    fig.tight_layout(); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "ph1_psi_top.png", dpi=150); plt.close(fig)

    count = lambda t: t["level"].value_counts().reindex(["ổn định", "theo dõi", "lệch lớn"], fill_value=0).to_dict()
    md = ["# PSI theo feature (Task 19)\n",
          "Bin theo phân vị tập gốc (10 bin), NaN là một bin riêng. Mốc 0.1 / 0.25 là **quy ước ngành**, dùng để tham chiếu.\n",
          f"**Điểm model (ensemble) train vs application_test: PSI = {score_psi:.4f} ({dr.psi_level(score_psi)}).**\n",
          "Lưu ý cách chấm: điểm train = OOF (mỗi dòng do 1 model fold chấm) + holdout (trung bình 5 fold); điểm test = "
          "trung bình 5 fold. Điểm trung bình 5 fold phân tán hẹp hơn nên PSI điểm thường bị đẩy lên chứ không bị kéo "
          "xuống — con số trên là cận trên, kết luận ổn định không đổi.\n",
          f"## Phần CV vs holdout — kiểm phép tách\n\nĐếm theo mức: {count(split_tbl)}. Max PSI = {split_tbl['psi'].max():.4f}.\n",
          f"## Train vs application_test — lệch thật\n\nĐếm theo mức: {count(test_tbl)}.\n",
          test_tbl.head(20).drop(columns=["pair"]).to_markdown(index=False, floatfmt=".4f"), "",
          "Bảng đủ: `outputs/reports/psi.csv`. Plot: `outputs/plots/ph1_psi_top.png`.",
          f"\nSinh bởi `scripts/ph1_drift.py` lúc {time.strftime('%Y-%m-%d %H:%M')}."]
    REPORT_MD.write_text("\n".join(md) + "\n")
    print(f"[psi] ghi {REPORT_MD}, {REPORT_CSV}")

if __name__ == "__main__":
    main()