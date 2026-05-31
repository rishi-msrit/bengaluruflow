import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import add_temporal_features, EXPECTED_NEW_TEMPORAL_COLS


@pytest.fixture
def sample_df():
    dates = pd.date_range("2022-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Area Name": ["Whitefield"] * 30,
        "Road/Intersection Name": ["MG Road"] * 30,
        "Traffic Volume": np.random.randint(100, 1000, 30),
        "Average Speed": np.random.uniform(10, 60, 30),
        "Travel Time Index": np.random.uniform(1.0, 3.0, 30),
        "Congestion Level": np.random.randint(1, 10, 30),
        "Road Capacity Utilization": np.random.uniform(0.2, 1.0, 30),
        "Incident Reports": np.random.randint(0, 5, 30),
        "Environmental Impact": np.random.uniform(0, 1, 30),
        "Public Transport Usage": np.random.randint(50, 500, 30),
        "Traffic Signal Compliance": np.random.uniform(0.5, 1.0, 30),
        "Parking Usage": np.random.randint(10, 200, 30),
        "Pedestrian and Cyclist Count": np.random.randint(10, 300, 30),
        "Weather Conditions": ["Clear"] * 15 + ["Overcast"] * 15,
        "Roadwork and Construction Activity": ["none"] * 20 + ["active"] * 10,
    })


def test_day_sin_in_range(sample_df):
    result = add_temporal_features(sample_df)
    assert result["day_sin"].between(-1, 1).all()


def test_day_cos_in_range(sample_df):
    result = add_temporal_features(sample_df)
    assert result["day_cos"].between(-1, 1).all()


def test_month_sin_in_range(sample_df):
    result = add_temporal_features(sample_df)
    assert result["month_sin"].between(-1, 1).all()


def test_month_cos_in_range(sample_df):
    result = add_temporal_features(sample_df)
    assert result["month_cos"].between(-1, 1).all()


def test_is_weekend_binary(sample_df):
    result = add_temporal_features(sample_df)
    assert set(result["is_weekend"].unique()).issubset({0, 1})


def test_is_holiday_binary(sample_df):
    result = add_temporal_features(sample_df)
    assert set(result["is_holiday"].unique()).issubset({0, 1})


def test_days_since_start_nonnegative(sample_df):
    result = add_temporal_features(sample_df)
    assert (result["days_since_start"] >= 0).all()


def test_expected_columns_added(sample_df):
    result = add_temporal_features(sample_df)
    for col in EXPECTED_NEW_TEMPORAL_COLS:
        assert col in result.columns, f"Missing: {col}"


def test_no_nulls_after_temporal_features(sample_df):
    result = add_temporal_features(sample_df)
    for col in EXPECTED_NEW_TEMPORAL_COLS:
        assert result[col].isnull().sum() == 0
