import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import TrafficDataset, AnomalyDataset, WINDOW_SIZE, HORIZON


def make_synthetic_df(n_days=30, n_roads=3, feature_cols=None):
    if feature_cols is None:
        feature_cols = [f"feat_{i}" for i in range(10)]

    rows = []
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    for road_id in range(n_roads):
        for date in dates:
            row = {
                "Date": date,
                "Road/Intersection Name": f"Road_{road_id}",
                "Incident Reports": np.random.randint(0, 3),
                **{col: np.random.randn() for col in feature_cols},
                "Traffic Volume": np.random.uniform(100, 1000),
                "Congestion Level": np.random.uniform(1, 10),
                "Average Speed": np.random.uniform(10, 80),
            }
            rows.append(row)

    return pd.DataFrame(rows), feature_cols


class TestTrafficDataset:
    def test_len_correct(self):
        n_days, n_roads = 20, 3
        df, feature_cols = make_synthetic_df(n_days=n_days, n_roads=n_roads)
        ds = TrafficDataset(df, feature_cols, window_size=WINDOW_SIZE, horizon=HORIZON)
        expected = n_roads * (n_days - WINDOW_SIZE - HORIZON + 1)
        assert len(ds) == expected

    def test_getitem_x_shape(self):
        df, feature_cols = make_synthetic_df()
        ds = TrafficDataset(df, feature_cols)
        x, y = ds[0]
        assert x.shape == (WINDOW_SIZE, len(feature_cols))

    def test_getitem_y_shape(self):
        df, feature_cols = make_synthetic_df()
        ds = TrafficDataset(df, feature_cols)
        x, y = ds[0]
        assert y.shape == (HORIZON, 3)

    def test_tensors_are_float32(self):
        df, feature_cols = make_synthetic_df()
        ds = TrafficDataset(df, feature_cols)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_empty_for_insufficient_data(self):
        df, feature_cols = make_synthetic_df(n_days=5, n_roads=2)
        ds = TrafficDataset(df, feature_cols, window_size=WINDOW_SIZE, horizon=HORIZON)
        assert len(ds) == 0


class TestAnomalyDataset:
    def test_getitem_shape(self):
        df, feature_cols = make_synthetic_df()
        ds = AnomalyDataset(df, feature_cols, window_size=WINDOW_SIZE)
        x = ds[0]
        assert x.shape == (WINDOW_SIZE, len(feature_cols))

    def test_metadata_length_matches_samples(self):
        df, feature_cols = make_synthetic_df()
        ds = AnomalyDataset(df, feature_cols, window_size=WINDOW_SIZE)
        assert len(ds.metadata) == len(ds.samples)
