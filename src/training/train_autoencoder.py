import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from src.data.loader import load_data, check_no_nulls
from src.data.features import build_features
from src.data.dataset import AnomalyDataset
from src.models.autoencoder import LSTMAutoencoder
from src.anomaly.score import compute_reconstruction_errors
from src.anomaly.threshold import fit_threshold, save_threshold

# hyperparameters
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
BOTTLENECK_DIM = 32
BATCH_SIZE = 32
EPOCHS = 60
EARLY_STOP_PAT = 8
WINDOW_SIZE = 7
THRESHOLD_PCTILE = 95

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "autoencoder_best.pt"
THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "anomaly_threshold.json"


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_batch in loader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            total_loss += criterion(model(x_batch), x_batch).item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_autoencoder] device: {device}")

    df = load_data()
    check_no_nulls(df)
    train_df, val_df, test_df, feature_cols, scaler, _ = build_features(df)
    num_features = len(feature_cols)

    train_loader = DataLoader(AnomalyDataset(train_df, feature_cols, WINDOW_SIZE),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AnomalyDataset(val_df, feature_cols, WINDOW_SIZE),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAutoencoder(num_features, HIDDEN_SIZE, BOTTLENECK_DIM, WINDOW_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_experiment("anomaly_detector")

    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": LEARNING_RATE, "hidden_size": HIDDEN_SIZE,
            "bottleneck_dim": BOTTLENECK_DIM, "window_size": WINDOW_SIZE,
            "threshold_pctile": THRESHOLD_PCTILE,
        })

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        patience_count = 0

        print(f"\n{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12}")
        print("-" * 38)

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f}")
            mlflow.log_metric("train_mse", train_loss, step=epoch)
            mlflow.log_metric("val_mse", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "num_features": num_features,
                    "feature_cols": feature_cols,
                }, CHECKPOINT_PATH)
            else:
                patience_count += 1
                if patience_count >= EARLY_STOP_PAT:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        # compute anomaly threshold on val set
        print("\n[train_autoencoder] Computing anomaly threshold...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        val_errors = compute_reconstruction_errors(model, val_loader, device)
        threshold = fit_threshold(val_errors, percentile=THRESHOLD_PCTILE)

        print(f"Threshold (P{THRESHOLD_PCTILE}): {threshold:.6f}")
        THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_threshold(threshold, THRESHOLD_PCTILE, THRESHOLD_PATH)

        mlflow.log_metric("best_val_mse", best_val_loss)
        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.log_artifact(str(CHECKPOINT_PATH))
        mlflow.log_artifact(str(THRESHOLD_PATH))

    print("[train_autoencoder] Done.")


if __name__ == "__main__":
    main()
