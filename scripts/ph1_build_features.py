from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ph1_credit_risk.features import builder  # noqa: E402
from src.ph1_credit_risk.features.bureau import BUREAU_AGG_PATH, build_bureau_features  # noqa: E402
from src.ph1_credit_risk.features.credit_card import CC_AGG_PATH, build_credit_card_features  # noqa: E402
from src.ph1_credit_risk.features.installments import INST_AGG_PATH, build_installments_features  # noqa: E402
from src.ph1_credit_risk.features.pos_cash import POS_AGG_PATH, build_pos_cash_features  # noqa: E402
from src.ph1_credit_risk.features.previous import PREV_AGG_PATH, PREV_ID_MAP_PATH, build_previous_features  # noqa: E402

STEPS = [
    ("bureau", build_bureau_features, [BUREAU_AGG_PATH]),
    ("previous", build_previous_features, [PREV_AGG_PATH, PREV_ID_MAP_PATH]),
    ("pos_cash", build_pos_cash_features, [POS_AGG_PATH]),
    ("installments", build_installments_features, [INST_AGG_PATH]),
    ("credit_card", build_credit_card_features, [CC_AGG_PATH]),
]

def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-existing", action="store_true", help="không dựng lại parquet aggregate đã có")
    args = parser.parse_args()

    t0 = time.time()
    for name, fn, outputs in STEPS:
        if args.skip_existing and all(p.exists() for p in outputs):
            print(f"[{name}] bỏ qua, đã có {[p.name for p in outputs]}")
            continue
        fn()
        gc.collect()
        print(f"[{name}] xong, tổng {time.time() - t0:.0f}s, RAM đỉnh {peak_rss_gb():.2f} GB")

    for split in ("train", "test"):
        df = builder.build_feature_table(split)
        size_mb = builder.features_path(split).stat().st_size / 1024**2
        all_nan = [c for c in df.columns if df[c].isna().all()]
        print(
            f"[features_{split}] {df.shape[0]:,} dòng x {df.shape[1]} cột, "
            f"parquet {size_mb:.1f} MB, cột toàn NaN: {all_nan or 'không'}"
        )
        del df
        gc.collect()

    print(f"[done] tổng {time.time() - t0:.0f}s ({(time.time() - t0) / 60:.1f} phút), RAM đỉnh {peak_rss_gb():.2f} GB")

if __name__ == "__main__":
    main()
