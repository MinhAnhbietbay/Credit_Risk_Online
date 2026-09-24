"""Test schema: mọi bảng khai báo đủ KEYS/DTYPES."""

import pytest

from src.ph1_credit_risk.data import schema

def test_every_table_has_keys_and_dtypes():
    for name in schema.TABLES:
        assert name in schema.KEYS, f"{name} thiếu trong KEYS"
        assert name in schema.DTYPES, f"{name} thiếu trong DTYPES"
        assert schema.KEYS[name], f"{name} có KEYS rỗng"

def test_table_path_points_to_expected_file():
    path = schema.table_path("bureau")
    assert path.name == "bureau.csv"
    assert path.parent == schema.HOME_CREDIT_DIR

def test_table_path_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        schema.table_path("khong_ton_tai")

def test_infer_dtype_by_prefix():
    assert schema.infer_dtype("SK_ID_CURR") == "int32"
    assert schema.infer_dtype("AMT_CREDIT") == "float32"
    assert schema.infer_dtype("DAYS_BIRTH") == "float32"
    assert schema.infer_dtype("CNT_CHILDREN") == "float32"
    assert schema.infer_dtype("NAME_CONTRACT_TYPE") == "category"
    assert schema.infer_dtype("CREDIT_ACTIVE") == "category"
    assert schema.infer_dtype("STATUS") == "category"
    assert schema.infer_dtype("FLAG_OWN_CAR") == "category"
    # Cột không nằm trong quy tắc nào -> để pandas tự suy luận
    assert schema.infer_dtype("FLAG_DOCUMENT_2") is None

def test_dtype_for_merges_overrides_over_inference():
    dtypes = schema.dtype_for("bureau_balance", ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])
    assert dtypes["SK_ID_BUREAU"] == "int32"
    assert dtypes["STATUS"] == "category"
    # MONTHS_BALANCE của bảng này là số nguyên, không NaN -> override thành int16
    assert dtypes["MONTHS_BALANCE"] == "int16"

def test_dtype_for_ignores_columns_not_requested():
    dtypes = schema.dtype_for("bureau", ["SK_ID_CURR"])
    assert set(dtypes) == {"SK_ID_CURR"}
