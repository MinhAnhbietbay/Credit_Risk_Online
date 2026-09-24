"""Test aggregate previous_application trên fixture 6 dòng / 2 khách."""

import numpy as np
import pandas as pd
import pytest

from src.ph1_credit_risk.features import previous as pv

@pytest.fixture
def prev():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [100, 101, 102, 103, 104, 105],
            "SK_ID_CURR": [1, 1, 1, 1, 2, 2],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Consumer loans", "Cash loans", "Cash loans",
                                   "Revolving loans", "Revolving loans"],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Canceled",
                                     "Approved", "Refused"],
            "NAME_YIELD_GROUP": ["low_normal", "high", "low_normal", "XNA", "middle", "middle"],
            "PRODUCT_COMBINATION": ["Cash", "POS mobile", "Cash", "Cash", "Card Street", "Card Street"],
            "AMT_APPLICATION": [1000.0, 2000.0, 3000.0, 500.0, 800.0, 1200.0],
            "AMT_CREDIT": [1000.0, 2500.0, 3000.0, 0.0, 800.0, 1000.0],
            "AMT_ANNUITY": [100.0, np.nan, 300.0, 50.0, 80.0, 100.0],
            "CNT_PAYMENT": [12.0, 24.0, 12.0, np.nan, 6.0, 6.0],
            "HOUR_APPR_PROCESS_START": [10.0, 11.0, 12.0, 13.0, 9.0, 15.0],
            "DAYS_DECISION": [-100.0, -200.0, -300.0, -400.0, -50.0, -60.0],
            "DAYS_FIRST_DRAWING": [365243.0, 365243.0, -290.0, 365243.0, -40.0, 365243.0],
            "DAYS_FIRST_DUE": [-70.0, 365243.0, -260.0, 365243.0, -20.0, 365243.0],
            "DAYS_LAST_DUE_1ST_VERSION": [200.0, 365243.0, 100.0, 365243.0, 150.0, 365243.0],
            "DAYS_LAST_DUE": [365243.0, 365243.0, 80.0, 365243.0, 365243.0, 365243.0],
            "DAYS_TERMINATION": [365243.0, 365243.0, 85.0, 365243.0, 365243.0, 365243.0],
        }
    )

def test_clean_days_replaces_sentinel_with_nan(prev):
    out = pv.clean_days(prev)
    for col in pv.DAYS_SENTINEL_COLS:
        assert not (out[col] == 365243).any(), col
    # DAYS_DECISION không dùng mã 365243
    assert out["DAYS_DECISION"].equals(prev["DAYS_DECISION"])
    # không sửa tại chỗ
    assert (prev["DAYS_FIRST_DRAWING"] == 365243).any()

def test_agg_previous_one_row_per_customer(prev):
    out = pv.agg_previous(prev)
    assert out.index.name == "SK_ID_CURR"
    assert sorted(out.index) == [1, 2]
    assert out["PREV_COUNT"].tolist() == [4, 2]
    assert out.columns.is_unique
    assert all(c.startswith("PREV_") for c in out.columns)

def test_agg_previous_status_counts_and_refused_ratio(prev):
    out = pv.agg_previous(prev)
    assert out.loc[1, "PREV_STATUS_APPROVED_COUNT"] == 2
    assert out.loc[1, "PREV_STATUS_REFUSED_COUNT"] == 1
    assert out.loc[1, "PREV_STATUS_CANCELED_COUNT"] == 1
    assert out.loc[1, "PREV_STATUS_UNUSED_OFFER_COUNT"] == 0
    assert out.loc[1, "PREV_REFUSED_RATIO"] == pytest.approx(0.25)
    assert out.loc[2, "PREV_REFUSED_RATIO"] == pytest.approx(0.5)

def test_agg_previous_numeric_and_ratio(prev):
    out = pv.agg_previous(prev)
    assert out.loc[1, "PREV_AMT_APPLICATION_SUM"] == pytest.approx(6500.0)
    assert out.loc[1, "PREV_AMT_CREDIT_MAX"] == pytest.approx(3000.0)
    # APP_CREDIT_RATIO khách 1: 1.0, 0.8, 1.0, NaN (chia 0) -> max 1.0, min 0.8
    assert out.loc[1, "PREV_APP_CREDIT_RATIO_MAX"] == pytest.approx(1.0)
    assert out.loc[1, "PREV_APP_CREDIT_RATIO_MIN"] == pytest.approx(0.8)
    assert np.isfinite(out.select_dtypes("number").to_numpy()[~np.isnan(out.select_dtypes("number").to_numpy())]).all()

def test_agg_previous_days_sentinel_cleaned_before_agg(prev):
    out = pv.agg_previous(prev)
    # khách 2: DAYS_FIRST_DRAWING hợp lệ chỉ có -40 -> max/min đều -40, không phải 365243
    assert out.loc[2, "PREV_DAYS_FIRST_DRAWING_MAX"] == pytest.approx(-40.0)
    # khách 1: DAYS_LAST_DUE chỉ có 1 giá trị hợp lệ 80
    assert out.loc[1, "PREV_DAYS_LAST_DUE_MEAN"] == pytest.approx(80.0)

def test_agg_previous_approved_and_refused_groups(prev):
    out = pv.agg_previous(prev)
    # Approved của khách 1: AMT_CREDIT 1000 + 3000
    assert out.loc[1, "PREV_APPROVED_AMT_CREDIT_SUM"] == pytest.approx(4000.0)
    # Refused của khách 1: chỉ 2500
    assert out.loc[1, "PREV_REFUSED_AMT_CREDIT_SUM"] == pytest.approx(2500.0)
    assert out.loc[2, "PREV_REFUSED_AMT_CREDIT_SUM"] == pytest.approx(1000.0)

def test_agg_previous_modes(prev):
    out = pv.agg_previous(prev)
    assert out.loc[1, "PREV_NAME_CONTRACT_TYPE_MODE"] == "Cash loans"
    assert out.loc[2, "PREV_NAME_CONTRACT_TYPE_MODE"] == "Revolving loans"
    assert out.loc[1, "PREV_NAME_YIELD_GROUP_MODE"] == "low_normal"
    assert out.loc[2, "PREV_PRODUCT_COMBINATION_MODE"] == "Card Street"
    for c in pv.MODE_COLS:
        assert isinstance(out[f"PREV_{c}_MODE"].dtype, pd.CategoricalDtype)

def test_prev_id_map(prev):
    dup = pd.concat([prev, prev.iloc[[0]]], ignore_index=True)
    m = pv.prev_id_map(dup)
    assert list(m.columns) == ["SK_ID_PREV", "SK_ID_CURR"]
    assert len(m) == 6
    assert m["SK_ID_PREV"].is_unique

def test_status_subsets_drop_all_nan_columns(prev):
    """Đơn bị từ chối không có DAYS_FIRST_DUE (toàn sentinel) -> không sinh cột PREV_REFUSED_ toàn NaN."""
    out = pv.agg_previous(prev)
    assert not any(c.startswith("PREV_REFUSED_DAYS_FIRST_DUE") for c in out.columns)
    assert "PREV_REFUSED_AMT_CREDIT_SUM" in out.columns
    assert "PREV_APPROVED_DAYS_FIRST_DUE_MEAN" in out.columns
    assert not out.isna().all().any()
