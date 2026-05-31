import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lstm_forecaster import LSTMForecaster

BATCH_SIZE = 8
WINDOW_SIZE = 7
NUM_FEATURES = 25
HORIZON = 3
NUM_TARGETS = 3


@pytest.fixture
def model():
    return LSTMForecaster(
        input_size=NUM_FEATURES, hidden_size=64, num_layers=2,
        dropout=0.0, horizon=HORIZON, num_targets=NUM_TARGETS,
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
    for n_feat in [10, 30, 50]:
        m = LSTMForecaster(input_size=n_feat, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(4, WINDOW_SIZE, n_feat)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (4, HORIZON, NUM_TARGETS)
