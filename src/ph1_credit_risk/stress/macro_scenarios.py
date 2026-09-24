"""Stress-test: đẩy gánh nặng trả nợ của cả danh mục lên rồi chấm điểm lại.

**ĐÂY LÀ PHÂN TÍCH ĐỘ NHẠY, KHÔNG PHẢI DỰ BÁO. KHÔNG CÓ QUAN HỆ NHÂN QUẢ NÀO ĐƯỢC KIỂM CHỨNG Ở ĐÂY**. 
Nó trả lời "nếu gánh nặng trả nợ của mọi hồ sơ tăng lên mức X thì model chấm lại ra sao", chứ
không trả lời "khi lãi suất tăng 2% thì bao nhiêu người vỡ nợ". Hành vi thật của khách khi chi phí tăng
(vay thêm chỗ khác, cơ cấu nợ, bỏ nợ) không nằm trong model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Scenario", "SCENARIOS_PATH", "BURDEN_NUMERATOR", "BURDEN_DENOMINATOR", "DERIVED_FROM_ANNUITY",
    "burden_quantile_multiplier", "apply_scenario", "portfolio_impact", "load_scenarios", "WARNING",
    "CHANNEL_COST", "CHANNEL_BEHAVIOUR", "BEHAVIOUR_COLUMNS", "quantile_shift", "monotonicity_check",
]

CHANNEL_COST = "chi_phi"          # chi phí trả nợ tăng: nhân hệ số lên AMT_ANNUITY
CHANNEL_BEHAVIOUR = "hanh_vi"     # hành vi trả nợ xấu đi: dịch phân vị các cột trễ hạn / dùng hạn mức

SCENARIOS_PATH = Path(__file__).with_name("scenarios.json")

# Gánh nặng trả nợ = tiền trả mỗi kỳ / mỗi đồng vay (chính là feature CREDIT_TERM)
BURDEN_NUMERATOR = "AMT_ANNUITY"
BURDEN_DENOMINATOR = "AMT_CREDIT"

# Cột dẫn xuất từ annuity -> (tử số, mẫu số) để tính lại sau cú sốc
DERIVED_FROM_ANNUITY = {
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_TERM": ("AMT_ANNUITY", "AMT_CREDIT"),
}

WARNING = (
    "Phân tích độ nhạy, KHÔNG phải dự báo: hệ số kịch bản là giả định về mức gánh nặng trả nợ, "
    "không có quan hệ nhân quả nào giữa biến vĩ mô và vỡ nợ được kiểm chứng trên dữ liệu này."
)


# Cột hành vi trả nợ bị đẩy xấu đi ở kênh `hanh_vi`. Chỉ liệt kê cột **đơn điệu tăng theo tỉ lệ vỡ nợ**
BEHAVIOUR_COLUMNS = (
    "INST_LATE_RATIO",       # tỉ lệ kỳ trả góp bị trễ
    "INST_DPD_MEAN",         # số ngày trễ trung bình
    "INST_DPD_STD",
    "POS_SK_DPD_DEF_MEAN",   # quá hạn ở khoản trả góp POS
    "CC_UTILIZATION_MEAN",   # dùng cạn hạn mức thẻ
    "CC_UTILIZATION_MAX",
)

@dataclass(frozen=True)
class Scenario:
    name: str
    annuity_multiplier: float = 1.0
    description: str = ""
    quantile: float | None = None            # phân vị gánh nặng mà kịch bản đẩy tới
    reference_quantile: float | None = None  # phân vị làm mốc (thường là trung vị)
    channel: str = CHANNEL_COST
    percentile_shift: float = 0.0            # chỉ dùng cho kênh `hanh_vi`


def burden_quantile_multiplier(
    df: pd.DataFrame, upper: float, reference: float = 0.5
) -> float:
    """Hệ số nhân annuity = (phân vị `upper` của gánh nặng) / (phân vị `reference`), **đo trên `df`**"""
    burden = df[BURDEN_NUMERATOR] / df[BURDEN_DENOMINATOR]
    burden = burden.replace([np.inf, -np.inf], np.nan).dropna()
    ref = float(burden.quantile(reference))
    if ref <= 0:
        raise ValueError(f"Phân vị mốc {reference} của gánh nặng <= 0, không chia được")
    return float(burden.quantile(upper)) / ref


def quantile_shift(s: pd.Series, delta: float) -> pd.Series:
    """Đẩy mỗi giá trị lên `delta` điểm phần trăm trong phân phối của chính cột đó.

    Hồ sơ đang ở phân vị r nhận giá trị mà hồ sơ ở phân vị min(r + delta, 1) đang có. Tính chất:
    giữ nguyên thứ tự xếp hạng, không hồ sơ nào tốt lên, không vượt mức đã quan sát được, và giữ NaN.

    """
    if delta <= 0:
        return s.copy()
    ok = s.notna()
    if not ok.any():
        return s.copy()
    ranks = s[ok].rank(pct=True, method="average").clip(upper=1.0)
    shifted = np.quantile(s[ok].to_numpy(), np.minimum(ranks.to_numpy() + delta, 1.0))
    out = s.copy().astype("float64")
    out[ok] = np.maximum(shifted, s[ok].to_numpy())  # không bao giờ tốt lên
    return out

def apply_scenario(X: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Trả **bản sao** đã điều chỉnh. Không sửa `X` tại chỗ.

    Kênh `chi_phi`: nhân `AMT_ANNUITY` với hệ số rồi **tính lại** các cột dẫn xuất từ định nghĩa gốc
    của chúng (không nhân mù). Kênh `hanh_vi`: dịch phân vị các cột `BEHAVIOUR_COLUMNS`.
    Cột nào không có trong `X` thì bỏ qua
    """
    out = X.copy(deep=True)

    if scenario.channel == CHANNEL_COST:
        if BURDEN_NUMERATOR not in out.columns:
            raise KeyError(f"Thiếu cột {BURDEN_NUMERATOR}, không áp được kịch bản")
        out[BURDEN_NUMERATOR] = out[BURDEN_NUMERATOR] * scenario.annuity_multiplier
        for col, (num, den) in DERIVED_FROM_ANNUITY.items():
            if col in out.columns and num in out.columns and den in out.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[col] = (out[num] / out[den]).replace([np.inf, -np.inf], np.nan)
        return out

    if scenario.channel == CHANNEL_BEHAVIOUR:
        for col in BEHAVIOUR_COLUMNS:
            if col in out.columns:
                out[col] = quantile_shift(out[col], scenario.percentile_shift)
        return out

    raise ValueError(f"Không biết kênh '{scenario.channel}'; chỉ có {CHANNEL_COST!r}, {CHANNEL_BEHAVIOUR!r}")

def monotonicity_check(df: pd.DataFrame, col: str, target: str = "TARGET",
                       n_bins: int = 10, tol: float = 0.003) -> dict:
    """Tỉ lệ vỡ nợ theo phân vị của `col` — cột có đơn điệu tăng theo rủi ro không?

    **Đây là kiểm tra bắt buộc trước khi chọn một cột làm kênh sốc** (phát hiện của Task 12: đẩy
    `CREDIT_TERM` lên cao lại làm model chấm rủi ro THẤP đi, vì nhóm phân vị cao nhất của cột đó là
    nhóm ít vỡ nợ nhất trong dữ liệu train).

    `tol` cho phép dao động nhỏ giữa hai nhóm liền kề mà vẫn coi là đơn điệu.
    """
    s = df[col]
    try:
        bins = pd.qcut(s, n_bins, duplicates="drop")
    except ValueError:
        return {"column": col, "monotonic": None, "reason": "không chia được phân vị", "rates": []}
    rates = df.groupby(bins, observed=True)[target].mean().to_numpy()
    return {
        "column": col,
        "n_bins": int(len(rates)),
        "rates": [round(float(r), 4) for r in rates],
        "monotonic": bool(np.all(np.diff(rates) >= -tol)),
        "top_bin_rate": float(rates[-1]),
        "max_bin_rate": float(rates.max()),
        "argmax_bin": int(rates.argmax()),
    }

def portfolio_impact(
    model: Any,
    X: pd.DataFrame,
    scenarios: Sequence[Scenario],
    threshold: float,
    risk_bands: Sequence[float] = (0.05, 0.15, 0.30),
) -> pd.DataFrame:
    """Mỗi kịch bản một dòng: PD trung bình, % hồ sơ vượt ngưỡng, dịch chuyển theo nhóm rủi ro.

    `threshold` là ngưỡng chặn đã chốt ở Task 10 — **giữ nguyên qua mọi kịch bản**: kịch bản làm dịch
    chuyển điểm số, không dịch chuyển chính sách.
    """
    rows = []
    base_flag = base_pd = None
    for sc in scenarios:
        p = np.asarray(model.predict_proba(apply_scenario(X, sc)))[:, 1]
        flag = float((p >= threshold).mean())
        row = {
            "scenario": sc.name,
            "channel": sc.channel,
            "description": sc.description,
            "annuity_multiplier": sc.annuity_multiplier,
            "percentile_shift": sc.percentile_shift,
            "pd_mean": float(p.mean()),
            "pd_median": float(np.median(p)),
            "flag_rate": flag,
        }
        lo = 0.0
        for b in risk_bands:
            row[f"band_{lo:g}_{b:g}"] = float(((p >= lo) & (p < b)).mean())
            lo = b
        row[f"band_{lo:g}_plus"] = float((p >= lo).mean())

        if base_pd is None:
            base_pd, base_flag = row["pd_mean"], flag
        row["pd_mean_delta"] = row["pd_mean"] - base_pd
        row["flag_rate_delta"] = flag - base_flag
        rows.append(row)
    return pd.DataFrame(rows)


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[Scenario]:
    """Đọc `scenarios.json`. Kịch bản cơ sở luôn đứng đầu."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = ("name", "annuity_multiplier", "description", "quantile", "reference_quantile",
              "channel", "percentile_shift")
    return [Scenario(**{k: s[k] for k in fields if k in s}) for s in cfg["scenarios"]]


# Sinh scenarios.json từ dữ liệu

SCENARIO_LEVELS = [
    # (hậu tố tên, phân vị đẩy tới, nhãn mức độ)
    ("bat_loi", 0.75, "nhóm 25% nặng nhất"),
    ("rat_bat_loi", 0.90, "nhóm 10% nặng nhất"),
]
REFERENCE_QUANTILE = 0.5  # mốc so sánh: hồ sơ trung vị


def build_scenarios_from_data(df: pd.DataFrame, levels=SCENARIO_LEVELS) -> dict:
    """Đo phân vị trên `df` rồi sinh cấu hình kịch bản cho **cả hai kênh**.

    Độ mạnh của hai kênh neo vào **cùng một cặp phân vị** (p50 → p75, p50 → p90):
    - kênh `chi_phi`: hệ số nhân annuity = thương hai phân vị của gánh nặng `AMT_ANNUITY/AMT_CREDIT`;
    - kênh `hanh_vi`: dịch mỗi hồ sơ lên đúng khoảng cách phân vị đó (p75 − p50 = 25 điểm phần trăm,
      p90 − p50 = 40 điểm) trong phân phối các cột hành vi.

    Con số là **kết quả đo**, không phải tham số đầu vào (xem `docs/ph1-co-so-kich-ban-stress-test.md`).
    Người chọn chỉ chọn *lấy cặp phân vị nào*.
    """
    burden = (df[BURDEN_NUMERATOR] / df[BURDEN_DENOMINATOR]).replace([np.inf, -np.inf], np.nan).dropna()

    scenarios = [
        {"name": "co_so", "annuity_multiplier": 1.0, "channel": CHANNEL_COST,
         "description": "giữ nguyên gánh nặng trả nợ hiện tại"},
    ]
    for name, q, label in levels:
        scenarios.append({
            "name": f"chi_phi_{name}",
            "channel": CHANNEL_COST,
            "annuity_multiplier": round(burden_quantile_multiplier(df, q, REFERENCE_QUANTILE), 4),
            "description": f"chi phí trả nợ bằng {label} của chính danh mục",
            "quantile": q,
            "reference_quantile": REFERENCE_QUANTILE,
        })
    for name, q, label in levels:
        scenarios.append({
            "name": f"hanh_vi_{name}",
            "channel": CHANNEL_BEHAVIOUR,
            "annuity_multiplier": 1.0,
            "percentile_shift": round(q - REFERENCE_QUANTILE, 4),
            "description": f"hành vi trả nợ xấu đi bằng {label} của chính danh mục",
            "quantile": q,
            "reference_quantile": REFERENCE_QUANTILE,
        })

    present = [c for c in BEHAVIOUR_COLUMNS if c in df.columns]
    return {
        "warning": WARNING,
        "burden_measure": f"{BURDEN_NUMERATOR} / {BURDEN_DENOMINATOR} (= feature CREDIT_TERM)",
        "behaviour_columns": present,
        "doc": "docs/ph1-co-so-kich-ban-stress-test.md",
        "derived_from": {
            "n_rows": int(len(burden)),
            "quantiles": {str(q): round(float(burden.quantile(q)), 6)
                          for q in (REFERENCE_QUANTILE, *[lv[1] for lv in levels])},
        },
        "scenarios": scenarios,
    }
