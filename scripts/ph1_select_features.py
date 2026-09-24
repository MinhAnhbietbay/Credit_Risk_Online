"""CLI Task 6: lọc + xếp hạng feature, chọn top N.

    python scripts/ph1_select_features.py [--n-features 80]

Ghi:
- `data/processed/selected_features.json` — danh sách cột chọn, `ph1_train.py` (Task 8) đọc lại
- `data/processed/feature_importance.csv`  — gain LightGBM đầy đủ sau lọc, cho evidence
- `outputs/reports/feature_selection.md`    — báo cáo: số cột loại theo lý do + top N
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402

from src.ph1_credit_risk.features.selection import select_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-features", type=int, default=80)
    args = ap.parse_args()
    chosen = select_features(args.n_features)
    print(f"[selection] đã chọn {len(chosen)} cột")


if __name__ == "__main__":
    main()
