from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from src.config import HOME_CREDIT_DIR, HOME_CREDIT_ZIP

EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "HomeCredit_columns_description.csv",
]


def _has_all_expected(dest_dir: Path) -> bool:
    return dest_dir.exists() and all((dest_dir / name).exists() for name in EXPECTED_FILES)


def extract_zip(zip_path: Path, dest_dir: Path, *, force: bool = False) -> Path:
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file zip: {zip_path}")

    if _has_all_expected(dest_dir) and not force:
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    entries = list(dest_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        nested.rmdir()

    if not _has_all_expected(dest_dir):
        missing = [name for name in EXPECTED_FILES if not (dest_dir / name).exists()]
        raise ValueError(f"Thiếu file sau khi giải nén: {missing}")

    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Giải nén dataset Home Credit Default Risk")
    parser.add_argument("--zip", type=Path, default=HOME_CREDIT_ZIP)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dest = extract_zip(args.zip, HOME_CREDIT_DIR, force=args.force)
    print(f"[unzip] Dữ liệu sẵn sàng tại: {dest}")
    for name in EXPECTED_FILES:
        size_mb = (dest / name).stat().st_size / 1_000_000
        print(f"  {name:<40} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()