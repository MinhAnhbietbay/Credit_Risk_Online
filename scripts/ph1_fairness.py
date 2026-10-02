"""số đo công bằng theo giới tính và nhóm tuổi — OOF (chính) và holdout (xác nhận) — cộng phép
kiểm proxy (80 feature đoán được giới tính tốt đến đâu). Chỉ đo, không sửa model.

    python scripts/ph1_fairness.py [--n-boot 500]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import OUTPUT_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import fairness as fr  # noqa: E402
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR, run_cv  # noqa: E402

EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task18-fairness"
REPORT_MD, REPORT_CSV = REPORTS_DIR / "fairness.md", REPORTS_DIR / "fairness.csv"

def gender_labels(raw: pd.Series) -> pd.Series:
    if raw.dtype.kind in "OUS" or str(raw.dtype) == "category":
        return raw.astype(str)
    maps = json.loads((FEATURES_TRAIN_PATH.parent / "label_maps.json").read_text())["CODE_GENDER"]
    inv = {v: k for k, v in maps.items()}
    return raw.map(inv).astype(str)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=500)
    n_boot = ap.parse_args().n_boot
    thr = json.loads((REPORTS_DIR / "holdout_metrics.json").read_text())["threshold"]
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    df = pd.read_parquet(FEATURES_TRAIN_PATH, columns=["SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH"] + [
        f for f in feats if f != "DAYS_BIRTH"])
    attrs = pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"], "gender": gender_labels(df["CODE_GENDER"]),
                          "age": fr.age_band(df["DAYS_BIRTH"])})

    ho = np.load(OOF_DIR / "holdout_preds.npz")
    sets = {"oof": (np.load(OOF_DIR / "cv_ids.npy"), np.load(OOF_DIR / "cv_target.npy"), np.load(OOF_DIR / "ensemble_oof.npy")),
            "holdout": (ho["ids"], ho["y"], ho["ensemble"])}
    tables, summary = [], []
    for sname, (ids, y, p) in sets.items():
        a = fr.align_attributes(ids, attrs)
        # Bất biến không nhìn thấy được trong báo cáo: ghép lệch thứ tự dòng vẫn ra một bảng đầy đủ,
        # số trông hợp lý, chỉ là gán sai người. In ra để biết bảng dưới nói về đúng những hồ sơ này.
        assert len(a) == len(ids), f"{sname}: ghép ra {len(a)} dòng, cv_ids có {len(ids)}"
        assert (a["SK_ID_CURR"].to_numpy() == ids).all(), f"{sname}: thuộc tính lệch thứ tự so với ids"
        print(f"[fair] {sname}: ghép thuộc tính {len(a):,} dòng, khớp thứ tự ids ✓", flush=True)
        for attr in ("gender", "age"):
            t = fr.group_metrics(y, p, a[attr].to_numpy(), thr)
            t.insert(0, "attribute", attr); t.insert(0, "set", sname)
            tables.append(t)
            s = fr.disparity_summary(t)
            if sname == "oof":
                s["flag_rate_ratio_lo"], s["flag_rate_ratio_hi"] = fr.flag_rate_ratio_ci(y, p, a[attr].to_numpy(), thr, n_boot)
            summary.append({"set": sname, "attribute": attr, **s})
            print(f"[fair] {sname} {attr}: ratio {s['flag_rate_ratio']:.3f}", flush=True)
    groups, summ = pd.concat(tables, ignore_index=True), pd.DataFrame(summary)

    # proxy: 80 feature (không có CODE_GENDER) đoán giới tính tốt đến đâu — trên phần CV
    cv_ids = set(sets["oof"][0].tolist())
    sub = df[df["SK_ID_CURR"].isin(cv_ids)]
    sub = sub[gender_labels(sub["CODE_GENDER"]).isin(["M", "F"])]
    cat = [c for c in json.loads(CATEGORICAL_COLUMNS_PATH.read_text()) if c in feats]
    target = (gender_labels(sub["CODE_GENDER"]) == "M").astype(int).reset_index(drop=True)
    proxy = run_cv("lightgbm", sub[feats].reset_index(drop=True), target, cat, n_splits=3)
    proxy_tbl = pd.DataFrame([{"target": "giới tính (M=1)", "auc_mean": proxy.mean_auc, "auc_std": proxy.std_auc, "n": len(sub)}])

    groups.to_csv(REPORT_CSV, index=False)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    summ.to_csv(EVIDENCE_DIR / "results.csv", index=False)
    proxy_tbl.to_csv(EVIDENCE_DIR / "proxy.csv", index=False)
    (EVIDENCE_DIR / "config.json").write_text(json.dumps(
        {"threshold": thr, "min_n": 30, "age_bands": [b[2] for b in fr.AGE_BANDS], "n_boot": n_boot,
         "reference": "quy tắc 4/5 (EEOC, Mỹ) chỉ là mốc tham chiếu cho flag_rate_ratio, không phải luật VN"}, indent=1,
        ensure_ascii=False))
    md = ["# Công bằng theo nhóm (Task 18)\n",
          f"Ensemble cuối, ngưỡng {thr:.4f} (chặn ~21.9%). OOF phần CV là số chính, holdout để xác nhận. "
          "Nhóm n < 30: chỉ ghi n. **Báo cáo chỉ đo, chưa sửa model.**\n",
          "## Tóm tắt chênh lệch\n", summ.to_markdown(index=False, floatfmt=".4f"), "",
          "`flag_rate_ratio` = tỉ lệ chặn nhóm thấp nhất / nhóm cao nhất (mốc tham chiếu 0.8 theo quy tắc 4/5 của EEOC Mỹ). "
          "Nhóm có tỉ lệ vỡ nợ thật cao hơn thì bị chặn nhiều hơn là dự kiến — xem `fpr` và `calib_gap` để biết model có "
          "đối xử khác với người cùng mức rủi ro không.\n",
          "## Chi tiết theo nhóm\n", groups.to_markdown(index=False, floatfmt=".4f"), "",
          "## Proxy — 80 feature đoán giới tính\n", proxy_tbl.to_markdown(index=False, floatfmt=".4f"), "",
          "AUC 0.5 = không đoán được; càng gần 1 thì model càng có thể phân biệt gián tiếp theo giới tính.",
          f"\nSinh bởi `scripts/ph1_fairness.py` lúc {time.strftime('%Y-%m-%d %H:%M')}."]
    REPORT_MD.write_text("\n".join(md) + "\n")
    (EVIDENCE_DIR / "result.md").write_text("\n".join(md) + "\n")
    print(f"[fair] ghi {REPORT_MD}, {REPORT_CSV}, {EVIDENCE_DIR}")

if __name__ == "__main__":
    main()