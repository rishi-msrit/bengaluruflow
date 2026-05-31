import pytest
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_data, check_no_nulls, EXPECTED_COLUMNS, DEFAULT_DATA_PATH


@pytest.fixture(scope="module")
def raw_df():
    if not DEFAULT_DATA_PATH.exists():
        pytest.skip("Dataset not found — download from Kaggle first.")
    return load_data()


def test_all_16_columns_present(raw_df):
    for col in EXPECTED_COLUMNS:
        assert col in raw_df.columns, f"Missing column: '{col}'"


def test_column_count(raw_df):
    assert len(raw_df.columns) == 16


def test_no_missing_values(raw_df):
    null_counts = raw_df.isnull().sum()
    assert null_counts.sum() == 0, f"Null values:\n{null_counts[null_counts > 0]}"


def test_date_column_is_datetime(raw_df):
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Date"])


def test_row_count_nonzero(raw_df):
    assert len(raw_df) > 0


def test_missing_column_raises_value_error(tmp_path):
    bad_df = pd.DataFrame({col: [1] for col in EXPECTED_COLUMNS[:-1]})
    bad_path = tmp_path / "bad.csv"
    bad_df.to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        load_data(bad_path)
