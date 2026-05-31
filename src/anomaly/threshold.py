import json
from pathlib import Path
import numpy as np


def fit_threshold(errors, percentile=95):
    return float(np.percentile(errors, percentile))


def save_threshold(threshold, percentile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "threshold": threshold,
            "percentile": percentile,
        }, f, indent=2)
    print(f"[threshold] Saved -> {path}")


def load_threshold(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {path}\n"
            "Run train_autoencoder.py first."
        )
    with open(path) as f:
        return float(json.load(f)["threshold"])
