import numpy as np
import pandas as pd
import pytest

from src.ph1_credit_risk.features import bureau as bu

@pytest.fixture
def bb():
    """2 khoản vay: 10 có trễ hạn (1,5), 11 sạch (C/X/0)."""
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 10, 10, 11, 11, 11, 11, 11, 11],
            "MONTHS_BALANCE": [0, -1, -2, -3, 0, -1, -2, -3, -4, -5],
            "STATUS": ["0", "1", "5", "C", "C", "C", "X", "0", "0", "X"],
        }
    )

@pytest.fixture
def bureau_df():
    """4 khoản vay thuộc 2 khách: khách 1 có 2 Active + 1 Closed, khách 2 có 1 Closed."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12, 13],
            "CREDIT_ACTIVE": ["Active", "Active", "Closed", "Closed"],
            "CREDIT_TYPE": ["Consumer credit", "Credit card", "Consumer credit", "Mortgage"],
            "DAYS_CREDIT": [-100.0, -200.0, -300.0, -400.0],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 3000.0, 4000.0],
            "AMT_CREDIT_SUM_DEBT": [500.0, 0.0, 0.0, np.nan],
            "CNT_CREDIT_PROLONG": [0.0, 1.0, 0.0, 0.0],
        }
    )

def test_agg_bureau_balance_one_row_per_loan(bb):
    out = bu.agg_bureau_balance(bb)
    assert out.index.name == "SK_ID_BUREAU"
    assert sorted(out.index) == [10, 11]
    assert len(out) == 2

def test_agg_bureau_balance_counts_and_months(bb):
    out = bu.agg_bureau_balance(bb)
    assert out.loc[10, "BB_MONTHS_BALANCE_MIN"] == -3
    assert out.loc[10, "BB_MONTHS_BALANCE_MAX"] == 0
    assert out.loc[10, "BB_MONTHS_BALANCE_SIZE"] == 4
    assert out.loc[11, "BB_MONTHS_BALANCE_SIZE"] == 6
    assert out.loc[10, "BB_STATUS_1_COUNT"] == 1
    assert out.loc[10, "BB_STATUS_5_COUNT"] == 1
    assert out.loc[10, "BB_STATUS_C_COUNT"] == 1
    assert out.loc[11, "BB_STATUS_X_COUNT"] == 2
    assert out.loc[11, "BB_STATUS_1_COUNT"] == 0  # không có -> 0, không phải NaN

def test_agg_bureau_balance_dpd_features(bb):
    out = bu.agg_bureau_balance(bb)
    # khoản 10: 2/4 tháng có STATUS thuộc {1..5}
    assert out.loc[10, "BB_DPD_RATIO"] == pytest.approx(0.5)
    assert out.loc[10, "BB_MAX_DPD_LEVEL"] == 5
    # khoản 11: chưa từng trễ
    assert out.loc[11, "BB_DPD_RATIO"] == pytest.approx(0.0)
    assert out.loc[11, "BB_MAX_DPD_LEVEL"] == 0

def test_agg_bureau_one_row_per_customer_no_loss(bureau_df, bb):
    out = bu.agg_bureau(bureau_df, bu.agg_bureau_balance(bb))
    assert out.index.name == "SK_ID_CURR"
    assert sorted(out.index) == [1, 2]
    assert out["BUREAU_COUNT"].tolist() == [3, 1]

def test_agg_bureau_active_group_and_ratios(bureau_df, bb):
    out = bu.agg_bureau(bureau_df, bu.agg_bureau_balance(bb))
    assert out.loc[1, "BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT"] == 2
    assert out.loc[1, "BUREAU_CREDIT_ACTIVE_CLOSED_COUNT"] == 1
    assert out.loc[1, "BUREAU_ACTIVE_RATIO"] == pytest.approx(2 / 3)
    assert out.loc[2, "BUREAU_ACTIVE_RATIO"] == pytest.approx(0.0)
    assert out.loc[1, "BUREAU_CREDIT_TYPE_NUNIQUE"] == 2
    # Nhóm chỉ tính trên khoản Active của khách 1: 1000 + 2000
    assert out.loc[1, "BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"] == pytest.approx(3000.0)
    # Khách 2 không có khoản Active nào -> NaN không phải 0
    assert np.isnan(out.loc[2, "BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"])

def test_agg_bureau_numeric_aggs(bureau_df, bb):
    out = bu.agg_bureau(bureau_df, bu.agg_bureau_balance(bb))
    assert out.loc[1, "BUREAU_AMT_CREDIT_SUM_SUM"] == pytest.approx(6000.0)
    assert out.loc[1, "BUREAU_AMT_CREDIT_SUM_MAX"] == pytest.approx(3000.0)
    assert out.loc[1, "BUREAU_DAYS_CREDIT_MIN"] == pytest.approx(-300.0)

def test_agg_bureau_propagates_bb_columns(bureau_df, bb):
    out = bu.agg_bureau(bureau_df, bu.agg_bureau_balance(bb))
    assert out.loc[1, "BUREAU_BB_MAX_DPD_LEVEL_MAX"] == 5
    assert out.loc[1, "BUREAU_BB_MONTHS_BALANCE_SIZE_SUM"] == 10
    assert np.isnan(out.loc[2, "BUREAU_BB_MAX_DPD_LEVEL_MAX"])

def test_agg_bureau_without_bb(bureau_df):
    out = bu.agg_bureau(bureau_df, None)
    assert sorted(out.index) == [1, 2]
    assert not [c for c in out.columns if "BB_" in c]
    assert "BUREAU_AMT_CREDIT_SUM_SUM" in out.columns

def test_no_duplicate_columns(bureau_df, bb):
    out = bu.agg_bureau(bureau_df, bu.agg_bureau_balance(bb))
    assert out.columns.is_unique
    assert all(c.startswith("BUREAU_") for c in out.columns)

def test_combine_bb_partials_matches_single_pass(bb):
    one_shot = bu.agg_bureau_balance(bb)
    partials = [bu._bb_partial(bb.iloc[:3]), bu._bb_partial(bb.iloc[3:7]), bu._bb_partial(bb.iloc[7:])]
    chunked = bu._bb_finalize(bu._combine_bb_partials(partials))
    pd.testing.assert_frame_equal(chunked.sort_index(), one_shot.sort_index())
