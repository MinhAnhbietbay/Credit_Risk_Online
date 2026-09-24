"""cơ sở chọn ngưỡng
Cách chọn: **ngưỡng chính = tái tạo khẩu vị rủi ro thực tế của Home Credit** — tỉ lệ từ chối
lịch sử Refused / (Approved + Refused) trong `previous_application` (1.33M hồ sơ có kết quả) → chặn đúng phần
hồ sơ điểm cao nhất tương ứng. **Đối chiếu = ngưỡng KS/Youden** .

Phân tích phụ (không dùng làm ngưỡng chính): chi phí theo số tiền thật FP = AMT_CREDIT × r, FN = AMT_CREDIT × LGD,
với r từ khoản vay cũ và LGD quét độ nhạy (LGD không có trong dữ liệu, proxy chỉ có n≈50) — cho ai có LGD nội bộ.

Ghi `outputs/evidence/task10-threshold-basis/` + `outputs/reports/threshold_basis.json` (tham số cho ph1_report).

    python scripts/ph1_evidence_threshold_basis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import HOME_CREDIT_DIR, OUTPUT_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import metrics as mt  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402

EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task10-threshold-basis"
BASIS_PATH = REPORTS_DIR / "threshold_basis.json"
LGD_GRID = [0.3, 0.45, 0.6, 0.8, 1.0]
FINE_THRESHOLDS = np.round(np.arange(0.02, 0.951, 0.01), 2)

def historical_refusal_rate(prev: pd.DataFrame) -> dict:
    d = prev[prev["NAME_CONTRACT_STATUS"].isin(["Approved", "Refused"])]
    refused = d["NAME_CONTRACT_STATUS"] == "Refused"
    return {"n_decided": int(len(d)), "refusal_rate": float(refused.mean()),
            "refusal_rate_by_type": {k: float(v) for k, v in refused.groupby(d["NAME_CONTRACT_TYPE"]).mean().items()},
            "definition": "Refused / (Approved + Refused) trên previous_application"}

def interest_ratio(prev: pd.DataFrame) -> dict:
    a = prev[(prev["NAME_CONTRACT_STATUS"] == "Approved") & prev["NAME_CONTRACT_TYPE"].isin(["Cash loans", "Consumer loans"])]
    a = a.dropna(subset=["AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT"])
    a = a[(a["AMT_CREDIT"] > 0) & (a["CNT_PAYMENT"] > 0)]
    r = a["AMT_ANNUITY"] * a["CNT_PAYMENT"] / a["AMT_CREDIT"] - 1
    by_type = r.groupby(a["NAME_CONTRACT_TYPE"]).median()
    return {"n_prev_loans": int(len(a)), "r_median_all": float(r.median()),
            "r_by_type": {k: float(v) for k, v in by_type.items()},
            "r_for_application": {"Cash loans": float(by_type["Cash loans"]), "Revolving loans": float(r.median())},
            "definition": "r = AMT_ANNUITY*CNT_PAYMENT/AMT_CREDIT - 1, previous_application Approved (Cash+Consumer)"}

def main() -> None:
    prev = pd.read_csv(HOME_CREDIT_DIR / "previous_application.csv",
                       usecols=["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT"])
    refusal = historical_refusal_rate(prev)
    r_info = interest_ratio(prev)
    del prev
    print(f"[basis] tỉ lệ từ chối lịch sử {refusal['refusal_rate']:.3f} trên {refusal['n_decided']:,} hồ sơ; "
          f"theo loại { {k: round(v, 3) for k, v in refusal['refusal_rate_by_type'].items()} }")

    ids = np.load(OOF_DIR / "cv_ids.npy"); y = np.load(OOF_DIR / "cv_target.npy")
    oof = np.load(OOF_DIR / "ensemble_oof.npy")

    rows = []
    def add(basis, t, kind, note=""):
        s = mt.summary_at_threshold(y, oof, t)
        rows.append({"basis": basis, "kind": kind, "threshold": t, "flag_rate": float((oof >= t).mean()),
                     "precision": s["precision"], "recall": s["recall"], "f1": s["f1"], "note": note})

    t_main = mt.threshold_for_flag_rate(oof, refusal["refusal_rate"])
    add(f"từ chối lịch sử {refusal['refusal_rate']:.1%}", t_main, "CHÍNH", "khẩu vị rủi ro thực tế của Home Credit")
    for typ, rate in refusal["refusal_rate_by_type"].items():
        if typ in ("Cash loans", "Revolving loans", "Consumer loans"):
            add(f"  từ chối {typ} {rate:.1%}", mt.threshold_for_flag_rate(oof, rate), "tham khảo", "cùng công thức, theo loại")
    add("KS/Youden", mt.youden_threshold(y, oof), "ĐỐI CHIẾU", "max TPR−FPR, không giả định")

    # phụ: chi phí theo số tiền thật, LGD quét
    app = pd.read_csv(HOME_CREDIT_DIR / "application_train.csv", usecols=["SK_ID_CURR", "AMT_CREDIT", "NAME_CONTRACT_TYPE"])
    app = app.set_index("SK_ID_CURR").loc[ids]
    credit = app["AMT_CREDIT"].to_numpy()
    gain = credit * app["NAME_CONTRACT_TYPE"].map(r_info["r_for_application"]).to_numpy()
    for lgd in LGD_GRID:
        tbl = mt.expected_cost_table(y, oof, loss_if_bad=credit * lgd, gain_if_good=gain, thresholds=FINE_THRESHOLDS)
        add(f"chi phí thật, LGD={lgd:.2f}", float(tbl.loc[tbl["recommended"], "threshold"].iloc[0]), "phụ",
            f"FP=AMT_CREDIT×r (cash {r_info['r_by_type']['Cash loans']:.2f}), FN=AMT_CREDIT×LGD; LGD giả định")
    old = mt.threshold_table(y, oof, 5.0, 1.0)
    add("cũ: fn=5, fp=1", float(old.loc[old["recommended"], "threshold"].iloc[0]), "bỏ", "tỉ lệ bịa, không có cơ sở")

    res = pd.DataFrame(rows)
    print(res.drop(columns="note").to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(EVIDENCE_DIR / "results.csv", index=False)
    basis = {"method": "flag_rate", "flag_rate": refusal["refusal_rate"], "refusal": refusal, "interest_ratio": r_info,
             "lgd_proxy": {"value": 0.36, "n": 48, "definition": "1 − đã trả/phải trả của khoản vay cũ kết thúc Demand/Amortized debt",
                           "status": "chỉ để tham khảo, KHÔNG dùng làm ngưỡng chính (n quá nhỏ, proxy ≠ LGD ròng)"}}
    (EVIDENCE_DIR / "config.json").write_text(json.dumps(basis, indent=1))
    BASIS_PATH.write_text(json.dumps(basis, indent=1))
    ks = res[res["kind"] == "ĐỐI CHIẾU"].iloc[0]; main_row = res.iloc[0]
    lgd_rows = res[res["kind"] == "phụ"]
    (EVIDENCE_DIR / "result.md").write_text(
        "# Task 10 — Cơ sở chọn ngưỡng\n\n"
        f"**Quyết định:** ngưỡng chính = chặn {refusal['refusal_rate']:.1%} hồ sơ điểm cao nhất — đúng tỉ lệ Home Credit "
        f"đã từ chối trên {refusal['n_decided']:,} hồ sơ lịch sử (`previous_application`); trên OOF ensemble = "
        f"{main_row.threshold:.3f} (P {main_row.precision:.3f}, R {main_row.recall:.3f}). Đối chiếu: KS/Youden "
        f"{ks.threshold:.3f} (P {ks.precision:.3f}, R {ks.recall:.3f}).\n\n"
        "**So sánh:** tỉ lệ từ chối lịch sử (chung, theo loại) / KS / chi phí theo số tiền thật với LGD 0.3–1.0 / "
        f"**Số liệu:** chi phí thật cho ngưỡng {lgd_rows.threshold.min():.2f}–{lgd_rows.threshold.max():.2f} tùy LGD — "
        "quá nhạy với một tham số không có trong dữ liệu (proxy LGD≈0.36 chỉ n=48) nên không dùng làm ngưỡng chính; "
        "```\n" + res.drop(columns="note").to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```\n")
    print(f"[basis] ghi {EVIDENCE_DIR}, {BASIS_PATH}")

if __name__ == "__main__":
    main()
