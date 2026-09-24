import zipfile
from pathlib import Path
import pytest
from scripts.ph1_unzip_data import extract_zip, EXPECTED_FILES

def _make_zip(path: Path, nested: bool) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name in EXPECTED_FILES:
            arcname = f"home-credit-default-risk/{name}" if nested else name
            zf.writestr(arcname, "col1,col2\n1,2\n")

def test_extract_flat_zip(tmp_path):
    zip_path = tmp_path / "data.zip"
    _make_zip(zip_path, nested=False)
    dest = tmp_path / "out"
    extract_zip(zip_path, dest)
    for name in EXPECTED_FILES:
        assert (dest / name).exists()

def test_extract_nested_zip_is_flattened(tmp_path):
    zip_path = tmp_path / "data.zip"
    _make_zip(zip_path, nested=True)
    dest = tmp_path / "out"
    extract_zip(zip_path, dest)
    for name in EXPECTED_FILES:
        assert (dest / name).exists()
    assert not (dest / "home-credit-default-risk").exists()

def test_extract_is_idempotent(tmp_path):
    zip_path = tmp_path / "data.zip"
    _make_zip(zip_path, nested=False)
    dest = tmp_path / "out"
    extract_zip(zip_path, dest)
    (dest / EXPECTED_FILES[0]).write_text("")
    extract_zip(zip_path, dest, force=False)
    assert (dest / EXPECTED_FILES[0]).read_text() == ""

def test_extract_missing_zip_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_zip(tmp_path / "no.zip", tmp_path / "out")

def test_extract_incomplete_zip_raises(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(EXPECTED_FILES[0], "col1\n1\n")
    with pytest.raises(ValueError):
        extract_zip(zip_path, tmp_path / "out")