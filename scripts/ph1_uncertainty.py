"""khoảng tin cậy 95% bootstrap cho số chính (OOF và holdout) + so ensemble với model đơn tốt nhất.
    python scripts/ph1_uncertainty.py [--n-boot 1000]
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

from src.config import REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import bootstrap as bs  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402

REPORT_MD = REPORTS_DIR / "uncertainty.md"
REPORT_JSON = REPORTS_DIR / "uncertainty.json"
BEST_SINGLE = "catboost"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    n = ap.parse_args().n_boot
    thr = json.loads((REPORTS_DIR / "holdout_metrics.json").read_text())["threshold"]
    metrics = dict(bs.DEFAULT_METRICS)
    metrics["recall@thr"] = lambda y, p: float(((p >= thr) & (y == 1)).sum() / max((y == 1).sum(), 1))
    metrics["precision@thr"] = lambda y, p: float(((p >= thr) & (y == 1)).sum() / max((p >= thr).sum(), 1))

    y_oof = np.load(OOF_DIR / "cv_target.npy")
    oof = {m: np.load(OOF_DIR / f"{m}_oof.npy") for m in ("ensemble", BEST_SINGLE)}
    ho = np.load(OOF_DIR / "holdout_preds.npz")

    out, t0 = {}, time.time()
    for name, y, p in (("oof", y_oof, oof["ensemble"]), ("holdout", ho["y"], ho["ensemble"])):
        out[name] = bs.bootstrap_ci(y, p, metrics, n_boot=n).to_dict(orient="records")
        print(f"[ci] {name} xong ({time.time() - t0:.0f}s)", flush=True)
    out["ensemble_minus_" + BEST_SINGLE] = {
        "oof": bs.paired_bootstrap_diff(y_oof, oof["ensemble"], oof[BEST_SINGLE], bs.DEFAULT_METRICS["auc"], n_boot=n),
        "holdout": bs.paired_bootstrap_diff(ho["y"], ho["ensemble"], ho[BEST_SINGLE], bs.DEFAULT_METRICS["auc"], n_boot=n),
    }
    out["config"] = {"n_boot": n, "alpha": 0.05, "resample": "phân tầng theo TARGET", "threshold": thr}
    REPORT_JSON.write_text(json.dumps(out, indent=1))

    fmt = lambda rows: pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")
    d = out["ensemble_minus_" + BEST_SINGLE]
    REPORT_MD.write_text("\n".join([
        "# Khoảng tin cậy 95% (Task 16)\n",
        f"Bootstrap {n} lần, rút phân tầng theo nhãn, khoảng phân vị. Ngưỡng {thr:.4f} (chặn ~21.9%).\n",
        "## Holdout 20%\n", fmt(out["holdout"]), "",
        "## OOF phần CV 80%\n", fmt(out["oof"]), "",
        f"## Ensemble có thật sự hơn `{BEST_SINGLE}`? (bootstrap theo cặp, hiệu AUC)\n",
        fmt([{"tập": k, **v} for k, v in d.items()]), "",
        "Nếu khoảng [lo, hi] không chứa 0 thì chênh lệch không phải do nhiễu mẫu.",
        f"\nSinh bởi `scripts/ph1_uncertainty.py` lúc {time.strftime('%Y-%m-%d %H:%M')}.",
    ]) + "\n")
    print(f"[ci] ghi {REPORT_MD}, {REPORT_JSON}")

if __name__ == "__main__":
    main()
