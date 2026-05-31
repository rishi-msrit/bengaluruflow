import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_reconstruction_errors(model, loader, device):
    """Returns per-sample MSE for all windows in the loader."""
    model.eval()
    errors = []

    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            reconstruction = model(x_batch)
            mse_per_sample = ((x_batch - reconstruction) ** 2).mean(dim=[1, 2])
            errors.extend(mse_per_sample.cpu().numpy().tolist())

    return np.array(errors, dtype=np.float32)
