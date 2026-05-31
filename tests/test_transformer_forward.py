import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.transformer_forecaster import TransformerForecaster

BATCH_SIZE = 8
WINDOW_SIZE = 7
NUM_FEATURES = 25
HORIZON = 3
NUM_TARGETS = 3


@pytest.fixture
def model():
    return TransformerForecaster(
        input_size=NUM_FEATURES, d_model=32, nhead=4, num_layers=2,
        dim_feedforward=64, dropout=0.0, window_size=WINDOW_SIZE,
        horizon=HORIZON, num_targets=NUM_TARGETS,
    )


def test_output_shape(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (BATCH_SIZE, HORIZON, NUM_TARGETS)


def test_output_is_float32(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.dtype == torch.float32


def test_batch_size_one(model):
    x = torch.randn(1, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, HORIZON, NUM_TARGETS)


def test_no_nan_in_output(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert not torch.isnan(y).any()


def test_different_feature_sizes():
    for n_feat in [10, 30]:
        m = TransformerForecaster(
            input_size=n_feat, d_model=32, nhead=4, num_layers=1,
            dim_feedforward=64, dropout=0.0, window_size=WINDOW_SIZE
        )
        x = torch.randn(4, WINDOW_SIZE, n_feat)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (4, HORIZON, NUM_TARGETS)
