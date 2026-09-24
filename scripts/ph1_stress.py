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

from src.config import CV_FOLDS, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR  # noqa: E402
from src.ph1_credit_risk.evaluation import metrics as mt  # noqa: E402
from src.ph1_credit_risk.features.builder import CATEGORICAL_COLUMNS_PATH  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED, SELECTED_FEATURES_PATH  # noqa: E402
from src.ph1_credit_risk.modeling import registry as rg  # noqa: E402
from src.ph1_credit_risk.modeling.cv import OOF_DIR  # noqa: E402
from src.ph1_credit_risk.modeling.split import cv_portion  # noqa: E402
from src.ph1_credit_risk.stress import macro_scenarios as ms  # noqa: E402

REPORT_MD = REPORTS_DIR / "stress_test.md"
REPORT_CSV = REPORTS_DIR / "stress_test.csv"
PLOT_PATH = PLOTS_DIR / "ph1_stress_bands.png"
BASIS_PATH = REPORTS_DIR / "threshold_basis.json"
EVIDENCE_DIR = OUTPUT_DIR / "evidence" / "task12-monotonicity"

MONOTONICITY_COLUMNS = ("CREDIT_TERM", "AMT_ANNUITY", "ANNUITY_INCOME_RATIO", *ms.BEHAVIOUR_COLUMNS)

BAND_LABELS = {"band_0_0.05": "PD < 5%", "band_0.05_0.15": "5–15%",
               "band_0.15_0.3": "15–30%", "band_0.3_plus": "≥ 30%"}
BAND_COLORS = ["#1baf7a", "#eda100", "#eb6834", "#c2383f"]

class EnsemblePortfolioModel:

    def __init__(self, members: dict[str, list], ensemble):
        self.members = members
        self.ensemble = ensemble

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = {name: np.mean([rg.predict_proba(m, X) for m in models], axis=0)
                 for name, models in self.members.items()}
        p = np.asarray(self.ensemble.combine(preds))
        return np.column_stack([1 - p, p])

def load_final_model() -> EnsemblePortfolioModel:
    members = {name: [joblib.load(MODELS_DIR / f"ph1_{name}_fold{k}.joblib") for k in range(CV_FOLDS)]
               for name in rg.list_models()}
    return EnsemblePortfolioModel(members, joblib.load(MODELS_DIR / "ph1_ensemble.joblib"))

def load_portfolio() -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = json.loads(SELECTED_FEATURES_PATH.read_text())
    df = cv_portion(pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED) + feats))
    return df[feats], df

def write_monotonicity_evidence(df: pd.DataFrame) -> list[dict]:
    rows = [ms.monotonicity_check(df, c) for c in MONOTONICITY_COLUMNS if c in df.columns]
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    pd.DataFrame([{"column": r["column"], "monotonic": r["monotonic"], "n_bins": r["n_bins"],
                   "top_bin_rate": r["top_bin_rate"], "max_bin_rate": r["max_bin_rate"],
                   "argmax_bin": r["argmax_bin"],
                   "rates": " ".join(f"{x:.4f}" for x in r["rates"])} for r in rows]
                 ).to_csv(EVIDENCE_DIR / "monotonicity.csv", index=False)

    bad = [r["column"] for r in rows if r["monotonic"] is False]
    good = [r["column"] for r in rows if r["monotonic"] is True]
    (EVIDENCE_DIR / "result.md").write_text(
        "stress-test chạy trên chi phí trả nợ (`AMT_ANNUITY`) và hành vi "
        "trả nợ (trễ hạn / dùng cạn hạn mức) — vì kênh chi phí một mình cho kết quả sai chiều.\n\n"
        "**So sánh:** tỉ lệ vỡ nợ theo thập phân vị của từng cột trên phần CV. Cột đơn điệu tăng thì đẩy "
        "nó lên cao mới thực sự là 'xấu đi' dưới mắt model.\n\n"
        f"**Số:** KHÔNG đơn điệu: {', '.join(bad) or 'không có'}. Đơn điệu: {', '.join(good) or 'không có'}. "
        "Chi tiết từng thập phân vị ở `monotonicity.csv`.\n\n"
        "`CREDIT_TERM` và `AMT_ANNUITY` — hai cột của kênh chi phí — có thập phân vị **cao "
        "nhất** lại là nhóm **ít vỡ nợ nhất**, vì đó là vay tiêu dùng POS món nhỏ kỳ hạn rất ngắn, một phân "
        "khúc khách khác hẳn. Đẩy cả danh mục vào vùng đó không mô phỏng cú sốc — nó mô phỏng 'giả vờ mọi "
        "khách đều là khách vay ngắn hạn'.\n",
        encoding="utf-8")
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-scenarios", action="store_true",
                    help="đo lại phân vị gánh nặng trên phần CV rồi ghi đè scenarios.json")
    args = ap.parse_args()

    t0 = time.time()
    X, df = load_portfolio()
    print(f"[stress] danh mục {X.shape} (phần CV, không chạm holdout)")

    mono = write_monotonicity_evidence(df)
    bad = [r["column"] for r in mono if r["monotonic"] is False]
    print(f"[stress] kiểm đơn điệu {len(mono)} cột -> KHÔNG đơn điệu: {bad or 'không có'}")

    if args.rebuild_scenarios or not ms.SCENARIOS_PATH.exists():
        cfg = ms.build_scenarios_from_data(X)
        ms.SCENARIOS_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stress] đo trên {cfg['derived_from']['n_rows']:,} hồ sơ, "
              f"phân vị gánh nặng {cfg['derived_from']['quantiles']} -> {ms.SCENARIOS_PATH}")

    scenarios = ms.load_scenarios()
    # Ngưỡng tính lại từ OOF ensemble ở đúng tỉ lệ chặn 
    flag_rate = float(json.loads(BASIS_PATH.read_text())["flag_rate"])
    oof = np.load(OOF_DIR / "ensemble_oof.npy")
    threshold = float(mt.threshold_for_flag_rate(oof, flag_rate))
    print(f"[stress] {len(scenarios)} kịch bản; ngưỡng giữ nguyên {threshold:.4f} "
          f"(= chặn {flag_rate:.1%} trên OOF, Task 10)")

    model = load_final_model()
    tab = ms.portfolio_impact(model, X, scenarios, threshold=threshold)
    tab.to_csv(REPORT_CSV, index=False)

    show = ["scenario", "channel", "annuity_multiplier", "percentile_shift",
            "pd_mean", "flag_rate", "flag_rate_delta"]
    print("\n" + tab[show].to_string(index=False))

    plot_bands(tab, PLOT_PATH)
    write_report(tab, scenarios, threshold, len(X), mono)
    print(f"\n[stress] -> {REPORT_MD}, {REPORT_CSV}, {PLOT_PATH}  ({time.time() - t0:.0f}s)")

def plot_bands(tab: pd.DataFrame, path: Path) -> None:
    cols = [c for c in BAND_LABELS if c in tab.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(tab))
    for c, color in zip(cols, BAND_COLORS):
        v = tab[c].to_numpy() * 100
        ax.bar(tab["scenario"], v, bottom=bottom, label=BAND_LABELS[c], color=color)
        bottom += v
    ax.set_ylabel("% hồ sơ trong danh mục")
    ax.set_title("Dịch chuyển nhóm rủi ro theo kịch bản — kênh chi phí vs kênh hành vi", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, ms.WARNING, fontsize=6.5, color="#5c5b55", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def write_report(tab: pd.DataFrame, scenarios, threshold: float, n_rows: int, mono: list[dict]) -> None:
    cfg = json.loads(ms.SCENARIOS_PATH.read_text())
    q = cfg["derived_from"]["quantiles"]
    lines = [
        "# Stress-test: chi phí trả nợ và hành vi trả nợ", "",
        f"> **{ms.WARNING}**", "",
        f"Gánh nặng trả nợ đo bằng `{cfg['burden_measure']}`. Phân vị đo trên "
        f"**{cfg['derived_from']['n_rows']:,} hồ sơ** của phần CV: "
        + ", ".join(f"p{float(k) * 100:.0f} = {v:.5f}" for k, v in q.items()) + ".",
        "",
        f"Kênh **hành vi** sốc các cột: {', '.join(f'`{c}`' for c in cfg['behaviour_columns'])} — "
        "mỗi hồ sơ nhích lên ngần ấy điểm phần trăm trong phân phối của chính danh mục "
        "(`quantile_shift`), vì các cột này có phân vị 25 bằng 0 nên nhân hệ số vô tác dụng.",
        "",
        "| Kịch bản | Kênh | Nghĩa | Cú sốc | Nguồn con số |",
        "|---|---|---|---:|---|",
    ]
    for s in scenarios:
        src = (f"p{s.quantile * 100:.0f} vs p{s.reference_quantile * 100:.0f} đo trên dữ liệu"
               if s.quantile is not None else "mốc, không đổi")
        shock = (f"×{s.annuity_multiplier:.4f}" if s.channel == ms.CHANNEL_COST
                 else f"+{s.percentile_shift * 100:.0f} điểm phân vị")
        lines.append(f"| `{s.name}` | {s.channel} | {s.description} | {shock} | {src} |")

    lines += [
        "", "## Tác động lên danh mục", "",
        f"Model cuối (ensemble stacking), **ngưỡng giữ nguyên {threshold:.4f}** ở mọi kịch bản — kịch bản làm "
        "dịch chuyển điểm số, không dịch chuyển chính sách. Danh mục: "
        f"{n_rows:,} hồ sơ (phần CV, không chạm holdout).", "",
        "| Kịch bản | Kênh | Cú sốc | PD TB | PD trung vị | % bị chặn | Chênh so với cơ sở |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in tab.iterrows():
        shock = (f"×{r['annuity_multiplier']:.4f}" if r["channel"] == ms.CHANNEL_COST
                 else f"+{r['percentile_shift'] * 100:.0f}đ phân vị")
        lines.append(
            f"| `{r['scenario']}` | {r['channel']} | {shock} | {r['pd_mean']:.4f} | "
            f"{r['pd_median']:.4f} | {r['flag_rate']:.1%} | {r['flag_rate_delta']:+.1%} |")

    lines += ["", "## Phân bố nhóm rủi ro", "",
              "| Kịch bản | " + " | ".join(BAND_LABELS.values()) + " |",
              "|---|" + "---:|" * len(BAND_LABELS)]
    for _, r in tab.iterrows():
        cells = " | ".join(f"{r[c]:.1%}" for c in BAND_LABELS if c in tab.index.names or c in r.index)
        lines.append(f"| `{r['scenario']}` | {cells} |")

    lines += [
        "", f"Biểu đồ: `{PLOT_PATH.relative_to(PLOT_PATH.parents[2])}`.", "",
        "## Giới hạn", "",
        "- **Phân tích độ nhạy, không phải dự báo.** \"nếu gánh nặng trả nợ tăng lên mức X thì model "
        "chấm lại ra sao\", không trả lời \"khi lãi suất tăng 2% thì bao nhiêu người vỡ nợ\".",
        "- **Chỉ 3 cột đổi:** `AMT_ANNUITY`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`. Mọi cột lịch sử "
        "(bureau/previous/installments) giữ nguyên",
        "- **Không tách được lãi suất khỏi kỳ hạn:** `CREDIT_TERM` cao có thể do vay đắt hoặc phải trả nhanh. "
        "Nên đây là cú sốc **gánh nặng trả nợ**, không phải cú sốc lãi suất thuần tuý.",
        "- **Hành vi khách không nằm trong model:** vay thêm chỗ khác, cơ cấu nợ, bỏ nợ đều không được mô hình hoá.",
        "- **Kênh hành vi có tính vòng quanh nhẹ:** trễ hạn ở khoản vay *trước* là dự báo mạnh cho vỡ nợ ở "
        "khoản vay *hiện tại*, nên giả định \"khách bắt đầu trả trễ\" rồi kết luận \"họ dễ vỡ nợ\" là một "
        "lập luận gần với đồng nghĩa. Nó trả lời \"nếu chất lượng trả nợ của danh mục tụt xuống mức của nhóm "
        "xấu hơn thì điểm số dịch chuyển ra sao\", không trả lời \"cú sốc vĩ mô làm bao nhiêu người trả trễ\".",
        "- **Kênh hành vi sốc chưa trọn vẹn:** các cột trễ hạn bị đẩy xấu đi, nhưng cột trả đúng hạn/trả đủ "
        "(`INST_PAYMENT_PERC_MEAN`, `INST_DBD_*`) giữ nguyên, nên hồ sơ sau cú sốc hơi thiếu nhất quán nội tại.",
        "- **`CC_UTILIZATION_*` chỉ phủ ~25% danh mục** (75% khách không có thẻ tín dụng), nên phần đóng góp "
        "của cột này vào cú sốc nhỏ hơn vẻ ngoài.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
