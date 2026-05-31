import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.autoencoder import LSTMAutoencoder

BATCH_SIZE = 8
WINDOW_SIZE = 7
NUM_FEATURES = 25


@pytest.fixture
def model():
    return LSTMAutoencoder(
        num_features=NUM_FEATURES, hidden_size=32,
        bottleneck_dim=16, window_size=WINDOW_SIZE,
    )


def test_reconstruction_shape_matches_input(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


def test_output_is_float32(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert out.dtype == torch.float32


def test_batch_size_one(model):
    x = torch.randn(1, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, WINDOW_SIZE, NUM_FEATURES)


def test_bottleneck_shape(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        latent = model.encode(x)
    assert latent.shape == (BATCH_SIZE, 16)


def test_no_nan_in_reconstruction(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


def test_different_feature_counts():
    for n_feat in [10, 20, 40]:
        m = LSTMAutoencoder(num_features=n_feat, hidden_size=16, bottleneck_dim=8, window_size=WINDOW_SIZE)
        x = torch.randn(4, WINDOW_SIZE, n_feat)
        with torch.no_grad():
            out = m(x)
        assert out.shape == x.shape
