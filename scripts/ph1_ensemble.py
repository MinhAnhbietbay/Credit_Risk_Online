"""trộn OOF của 5 model -> chọn ensemble cuối.
    python scripts/ph1_ensemble.py

Đọc `outputs/oof/<model>_oof.npy` + `cv_target.npy`. Ghi:
- `outputs/evidence/task09-ensemble-method/` — A/B các cấu hình trộn (số để chọn, không phải số final)
- `outputs/models/ph1_ensemble.joblib`      — `Ensemble` đã chọn
- `outputs/oof/ensemble_oof.npy`            — OOF của ensemble (cùng thứ tự cv_ids.npy)
- `outputs/reports/ensemble.md`             — kết quả chính thức
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import MODELS_DIR, OUTPUT_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.modeling import ensemble as ens  # noqa: E402
from src.ph1_credit_risk.modeling import registry as rg  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402

EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task09-ensemble-method"
ENSEMBLE_PATH = MODELS_DIR / "ph1_ensemble.joblib"
REPORT_PATH = REPORTS_DIR / "ensemble.md"
BOOSTING = ["catboost", "lightgbm", "xgboost"]

def load_oofs() -> tuple[dict[str, np.ndarray], np.ndarray]:
    y = np.load(OOF_DIR / "cv_target.npy")
    oofs = {m: np.load(OOF_DIR / f"{m}_oof.npy") for m in rg.list_models()}
    for m, p in oofs.items():
        assert p.shape == y.shape, m
    return oofs, y

def main() -> None:
    oofs, y = load_oofs()
    singles = {m: roc_auc_score(y, p) for m, p in oofs.items()}
    auc_w = {m: singles[m] - 0.5 for m in oofs}  # trọng số ~ phần AUC vượt ngẫu nhiên

    # evidence: các cấu hình trộn, tất cả trên OOF phần CV, holdout không chạm
    configs = {
        "rank_avg_all5_equal": lambda: ens.rank_average(oofs),
        "rank_avg_all5_auc_weighted": lambda: ens.rank_average(oofs, auc_w),
        "rank_avg_boost3_equal": lambda: ens.rank_average({m: oofs[m] for m in BOOSTING}),
        "stacking_all5": lambda: ens.stacking_oof(oofs, y)[0],
        "stacking_boost3": lambda: ens.stacking_oof({m: oofs[m] for m in BOOSTING}, y)[0],
    }
    rows = [{"config": m, "n_members": 1, "auc": a} for m, a in singles.items()]
    for name, fn in configs.items():
        rows.append({"config": name, "n_members": 3 if "boost3" in name else 5, "auc": float(roc_auc_score(y, fn()))})
        print(f"{name:>28}  AUC {rows[-1]['auc']:.4f}", flush=True)
    ev = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    best_single = max(singles, key=singles.get)
    best_cfg = ev.iloc[0]

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_csv(EVIDENCE_DIR / "results.csv", index=False)
    (EVIDENCE_DIR / "config.json").write_text(json.dumps(
        {"data": "OOF 5-fold trên phần CV 80% (Task 8), holdout không chạm", "auc_weights": auc_w,
         "stacking": "LogisticRegression(C=1) trên logit(OOF), CV lồng 5-fold"}, indent=1))
    (EVIDENCE_DIR / "result.md").write_text(
        "# Task 9 — Cách trộn ensemble\n\n"
        f"**Quyết định:** `{best_cfg.config}` (AUC OOF {best_cfg.auc:.4f}).\n\n"
        "**So sánh:** rank-average (5 model đều / 5 model trọng số AUC / 3 boosting) và stacking LR (5 / 3 boosting), "
        "tất cả chấm trên cùng OOF phần CV.\n\n"
        f"**Số liệu:** model đơn tốt nhất `{best_single}` {singles[best_single]:.4f}; cấu hình chọn hơn "
        f"{best_cfg.auc - singles[best_single]:+.4f}.\n\n```\n{ev.to_string(index=False)}\n```\n")

    # model cuối
    if best_cfg.config.startswith("rank_avg"):
        members = BOOSTING if "boost3" in best_cfg.config else list(oofs)
        w = auc_w if "auc_weighted" in best_cfg.config else {m: 1.0 for m in members}
        model = ens.Ensemble("rank_average", members, weights={m: w[m] for m in members}, oof_auc=float(best_cfg.auc))
    elif best_cfg.config.startswith("stacking"):
        members = BOOSTING if "boost3" in best_cfg.config else list(oofs)
        nested_oof, stacker = ens.stacking_oof({m: oofs[m] for m in members}, y)
        model = ens.Ensemble("stacking", members, stacker=stacker, oof_auc=float(best_cfg.auc))
    else:
        model = ens.Ensemble("single", [best_single], oof_auc=float(singles[best_single]))
    ens_oof = nested_oof if model.method == "stacking" else model.combine(oofs)
    final_auc = float(roc_auc_score(y, ens_oof))
    joblib.dump(model, ENSEMBLE_PATH)
    np.save(OOF_DIR / "ensemble_oof.npy", ens_oof)

    lines = [f"# Ensemble (Task 9)\n",
             f"Chọn theo AUC OOF trên phần CV 80% ({len(y)} dòng); holdout không chạm.\n",
             f"**Model cuối:** `{model.method}` trên {model.members}"
             + (f", trọng số {json.dumps({k: round(v, 4) for k, v in model.weights.items()})}" if model.weights else "")
             + f" — AUC OOF **{final_auc:.4f}** (model đơn tốt nhất `{best_single}` {singles[best_single]:.4f}, "
             f"chênh {final_auc - singles[best_single]:+.4f}).\n",
             "```", ev.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "```\n",
            ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\n[ensemble] {model.method} {model.members} AUC OOF {final_auc:.4f} "
          f"(best single {best_single} {singles[best_single]:.4f})")
    print(f"[ensemble] ghi {ENSEMBLE_PATH}, {OOF_DIR / 'ensemble_oof.npy'}, {REPORT_PATH}, {EVIDENCE_DIR}")

if __name__ == "__main__":
    main()
