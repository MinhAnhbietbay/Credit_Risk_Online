"""Chạy lại toàn bộ PH1 bằng một lệnh, đúng thứ tự phụ thuộc.

    python scripts/ph1_run_all.py                 # từ đầu (dựng feature ~ lâu nhất)
    python scripts/ph1_run_all.py --from report   # dùng lại model đã train, chạy từ báo cáo trở đi
    python scripts/ph1_run_all.py --only fairness drift
    python scripts/ph1_run_all.py --dry-run
Ghi thời gian từng bước vào outputs/reports/run_log.json. 
Chạy lại `report` là chấm lại holdout bằng **cùng** model đã chốt — tái lập,
không phải chọn lại.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES: list[tuple[str, list[str]]] = [
    ("features", ["scripts/ph1_build_features.py"]),
    ("select", ["scripts/ph1_select_features.py"]),
    ("train", ["scripts/ph1_train.py", "--models", "all"]),
    ("ensemble", ["scripts/ph1_ensemble.py"]),
    ("report", ["scripts/ph1_report.py"]),
    ("shap", ["scripts/ph1_shap.py"]),
    ("stress", ["scripts/ph1_stress.py"]),
    ("leakage", ["scripts/ph1_leakage.py"]),
    ("uncertainty", ["scripts/ph1_uncertainty.py"]),
    ("calibration", ["scripts/ph1_calibration.py"]),
    ("fairness", ["scripts/ph1_fairness.py"]),
    ("drift", ["scripts/ph1_drift.py"]),
    ("web_assets", ["scripts/ph1_web_assets.py"]),
]
NAMES = [n for n, _ in STAGES]

def select_stages(start: str | None, only: list[str] | None) -> list[str]:
    for n in ([start] if start else []) + (only or []):
        if n not in NAMES:
            raise ValueError(f"bước lạ: {n!r}. Có: {NAMES}")
    if only:
        return [n for n in NAMES if n in only]
    return NAMES[NAMES.index(start):] if start else list(NAMES)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = select_stages(args.start, args.only)
    cmds = dict(STAGES)
    log = []
    for n in todo:
        cmd = [sys.executable, *cmds[n]]
        print(f"\n=== [{n}] {' '.join(cmds[n])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        log.append({"stage": n, "seconds": round(time.time() - t0, 1), "returncode": rc})
        if rc != 0:
            print(f"[run_all] bước {n} lỗi (mã {rc}), dừng.", flush=True)
            break
    if log:
        (ROOT / "outputs" / "reports" / "run_log.json").write_text(json.dumps(
            {"finished": time.strftime("%Y-%m-%d %H:%M"), "stages": log}, indent=1))
    sys.exit(max((s["returncode"] for s in log), default=0))

if __name__ == "__main__":
    main()
