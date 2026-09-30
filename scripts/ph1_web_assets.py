from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from src.online.feature_set import load_categorical, load_selected_features  # noqa: E402
from src.ph1_credit_risk.explain.glossary import decode_category, describe  # noqa: E402
from src.ph1_credit_risk.features.selection import FEATURES_TRAIN_PATH, PROTECTED  # noqa: E402
from src.ph1_credit_risk.modeling.split import cv_portion  # noqa: E402

MEDIANS_PATH = PROCESSED_DIR / "feature_medians.json"
FORM_PATH = PROCESSED_DIR / "web_form_features.json"
SHAP_GLOBAL_PATH = REPORTS_DIR / "shap_global.csv"
N_FORM_FEATURES = 15


def build_medians(df: pd.DataFrame, features: list[str], categorical: list[str]) -> dict:
    out = {}
    for c in features:
        s = df[c]
        if c in categorical:
            mode = s.mode(dropna=True)
            out[c] = None if mode.empty else int(mode.iloc[0])
        else:
            v = pd.to_numeric(s, errors="coerce").median()
            out[c] = None if pd.isna(v) else float(v)
    return out

def build_form_features(df: pd.DataFrame, categorical: list[str]) -> list[dict]:
    """Top feature theo |SHAP|, kèm khoảng nhập và các lựa chọn cho cột categorical."""
    ranking = pd.read_csv(SHAP_GLOBAL_PATH)["feature"].tolist()[:N_FORM_FEATURES]
    items = []
    for name in ranking:
        s = df[name]
        item = {"feature": name, "mo_ta": describe(name), "categorical": name in categorical}
        if name in categorical:
            codes = sorted(int(v) for v in s.dropna().unique())
            item["options"] = [{"code": c, "label": decode_category(name, c) or str(c)}
                               for c in codes]
        else:
            s = pd.to_numeric(s, errors="coerce")
            item.update(min=float(s.quantile(0.01)), max=float(s.quantile(0.99)),
                        median=float(s.median()), step=_step(s))
        items.append(item)
    return items

def _step(s: pd.Series) -> float:
    """Bước nhảy ô nhập: 1 cho cột nguyên/tiền, nhỏ hơn cho cột tỉ lệ."""
    span = float(s.quantile(0.99) - s.quantile(0.01))
    if span >= 1000:
        return 1000.0
    if span >= 10:
        return 1.0
    return round(max(span / 100, 1e-4), 4)

def main() -> None:
    features = load_selected_features()
    categorical = load_categorical(features)
    df = cv_portion(pd.read_parquet(FEATURES_TRAIN_PATH, columns=list(PROTECTED) + features))
    print(f"[web-assets] phần CV {df.shape}, {len(features)} feature ({len(categorical)} categorical)")

    medians = build_medians(df, features, categorical)
    MEDIANS_PATH.write_text(json.dumps(medians, ensure_ascii=False, indent=1), encoding="utf-8")

    form = build_form_features(df, categorical)
    FORM_PATH.write_text(json.dumps(form, ensure_ascii=False, indent=1), encoding="utf-8")

    n_null = sum(v is None for v in medians.values())
    print(f"[web-assets] {MEDIANS_PATH} — {len(medians)} trung vị ({n_null} cột toàn NaN -> null)")
    print(f"[web-assets] {FORM_PATH} — {len(form)} feature lên form:")
    for i in form:
        kind = "categorical" if i["categorical"] else f"{i['min']:,.4g} .. {i['max']:,.4g}"
        print(f"   {i['feature']:36s} {kind}")

if __name__ == "__main__":
    main()