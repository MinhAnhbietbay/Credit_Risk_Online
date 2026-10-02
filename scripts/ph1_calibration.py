"""so trước/sau hiệu chỉnh (Platt, isotonic) cho ensemble và LightGBM đơn, trên OOF 5-fold lồng.
Holdout chỉ đọc để xác nhận: calibrator fit trên toàn bộ OOF rồi áp lên holdout.

    python scripts/ph1_calibration.py [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import bootstrap as bs  # noqa: E402
from src.ph1_credit_risk.evaluation import calibration as cal  # noqa: E402
from src.ph1_credit_risk.evaluation import metrics as mt  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402

EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task17-calibration"
REPORT_PATH = REPORTS_DIR / "calibration.md"
MODELS = ["ensemble", "lightgbm"]
METHODS = ["none", "platt", "isotonic"]
COLORS = {"none": "#52514e", "platt": "#2a78d6", "isotonic": "#eb6834"}


def _ax_style(ax):
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    n_boot = ap.parse_args().n_boot
    flag_rate = float(json.loads((REPORTS_DIR / "threshold_basis.json").read_text())["flag_rate"])
    y = np.load(OOF_DIR / "cv_target.npy")
    ho = np.load(OOF_DIR / "holdout_preds.npz")

    rows, curves, decision = [], {}, {}
    for m in MODELS:
        p = np.load(OOF_DIR / f"{m}_oof.npy")
        variants = {"none": p, **{k: cal.cross_calibrate(k, p, y) for k in ("platt", "isotonic")}}
        for k, q in variants.items():
            t = mt.threshold_for_flag_rate(q, flag_rate)
            rows.append({"model": m, "method": k, **cal.calibration_summary(y, q), "auc": float(roc_auc_score(y, q)),
                         "threshold_at_flag_rate": t, "flag_rate_actual": float((q >= t).mean())})
            curves[(m, k)] = cal.quantile_reliability(y, q)
        if m == "ensemble":
            for k in ("platt", "isotonic"):
                d = bs.paired_bootstrap_diff(y, p, variants[k], bs.DEFAULT_METRICS["brier"], n_boot=n_boot)
                ece_drop = cal.ece(y, p) - cal.ece(y, variants[k])
                decision[k] = {**d, "ece_drop": ece_drop, "adopt": bool(d["lo"] > 0 and ece_drop > 0)}
        print(f"[calib] {m} xong", flush=True)
    res = pd.DataFrame(rows)
    chosen = next((k for k in ("platt", "isotonic") if decision[k]["adopt"]), "none")

    # holdout: chỉ xác nhận, calibrator fit trên toàn bộ OOF
    p_oof = np.load(OOF_DIR / "ensemble_oof.npy")
    ho_rows = [{"method": "none", **cal.calibration_summary(ho["y"], ho["ensemble"])}]
    for k in ("platt", "isotonic"):
        q = cal.make_calibrator(k).fit(p_oof, y).predict(ho["ensemble"])
        ho_rows.append({"method": k, **cal.calibration_summary(ho["y"], q)})
    ho_tbl = pd.DataFrame(ho_rows)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, m in zip(axes, MODELS):
        lim = max(curves[(m, "none")][["mean_pred", "frac_pos"]].max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#c3c2b7", lw=1, ls="--")
        for k in METHODS:
            c = curves[(m, k)]
            ax.plot(c["mean_pred"], c["frac_pos"], marker="o", ms=4, lw=2, color=COLORS[k], label=k)
        ax.set(title=f"{m} — OOF, 10 bin bằng số lượng", xlabel="Xác suất dự đoán", ylabel="Tỉ lệ vỡ nợ thực")
        ax.legend(frameon=False)
        _ax_style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ph1_reliability.png", dpi=150)
    plt.close(fig)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(EVIDENCE_DIR / "results.csv", index=False)
    (EVIDENCE_DIR / "config.json").write_text(json.dumps(
        {"fit": "cross_calibrate 5-fold lồng trên OOF", "rule": "dùng nếu CI 95% của Brier(trước)−Brier(sau) > 0 và ECE giảm",
         "n_boot": n_boot, "decision": decision, "chosen": chosen}, indent=1))
    md = ["# Hiệu chỉnh xác suất (Task 17)\n",
          f"**Quyết định cho ensemble:** `{chosen}`.\n",
          "## OOF (calibrator fit 5-fold lồng)\n", res.to_markdown(index=False, floatfmt=".4f"), "",
          "## Ensemble — hiệu Brier trước − sau (bootstrap theo cặp)\n",
          pd.DataFrame([{"method": k, **v} for k, v in decision.items()]).to_markdown(index=False, floatfmt=".5f"), "",
          "## Holdout — chỉ xác nhận (calibrator fit trên toàn bộ OOF)\n", ho_tbl.to_markdown(index=False, floatfmt=".4f"), "",
          "Plot: `outputs/plots/ph1_reliability.png`.",
          f"\nSinh bởi `scripts/ph1_calibration.py` lúc {time.strftime('%Y-%m-%d %H:%M')}."]
    REPORT_PATH.write_text("\n".join(md) + "\n")
    (EVIDENCE_DIR / "result.md").write_text("\n".join(md) + "\n")
    print(f"[calib] chọn {chosen}. Ghi {REPORT_PATH}")


if __name__ == "__main__":
    main()
