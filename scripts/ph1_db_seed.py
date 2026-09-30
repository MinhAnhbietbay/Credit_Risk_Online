from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.online.engine import drop_all, get_session, init_db, ping  # noqa: E402
from src.online.models import register_offline_champion  # noqa: E402
from src.online import repository as repo  # noqa: E402
from src.online.seed import seed_database  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", type=float, default=0.6,
                    help="tỉ lệ hồ sơ đã biết kết quả; phần còn lại tiết lộ dần ở tab Train lại")
    ap.add_argument("--reset", action="store_true", help="xoá sạch mọi bảng trước khi nạp")
    args = ap.parse_args()

    ok, message = ping()
    if not ok:
        raise SystemExit(f"Không kết nối được PostgreSQL: {message}\n"
                         "Bật DB: docker compose -f infra/docker-compose.yml up -d")
    print(f"[db] {message[:60]}")

    if args.reset:
        drop_all()
        print("[db] đã xoá sạch bảng cũ")
    print(f"[db] bảng: {init_db()}")

    t0 = time.time()
    info = seed_database(labeled_fraction=args.labeled, verbose=True)
    print(f"[db] nạp {info['n_applicants']:,} hồ sơ × {info['n_feature_columns']} feature "
          f"(bộ `{info['feature_set_version']}`) trong {time.time() - t0:.0f}s")
    print(f"[db] đã biết kết quả {info['n_labeled']:,} · còn giấu nhãn {info['n_unlabeled']:,}")

    session = get_session()
    try:
        if repo.active_model(session) is None:
            mv = register_offline_champion(session, activate=True)
            print(f"[db] đăng ký champion: {mv.model_name} v{mv.version} "
                  f"(holdout AUC {(mv.valid_metrics or {}).get('auc'):.4f}, ngưỡng {mv.threshold:.4f})")
        else:
            print(f"[db] đã có model đang dùng: {repo.active_model(session).model_name}")
        print(f"[db] {repo.counts(session)}")
    finally:
        session.close()

    print("\nChạy web:  uvicorn src.api.main:app --port 8000  &&  cd frontend && npm run dev")

if __name__ == "__main__":
    main()
