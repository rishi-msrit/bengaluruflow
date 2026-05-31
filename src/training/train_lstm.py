import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from src.data.loader import load_data, check_no_nulls
from src.data.features import build_features
from src.data.dataset import TrafficDataset
from src.models.lstm_forecaster import LSTMForecaster

# hyperparameters
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 80
EARLY_STOP_PAT = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5
WINDOW_SIZE = 7
HORIZON = 3
NUM_TARGETS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "lstm_best.pt"


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            total_loss += criterion(model(x_batch), y_batch).item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_lstm] device: {device}")

    df = load_data()
    check_no_nulls(df)
    train_df, val_df, test_df, feature_cols, scaler, _ = build_features(df)
    num_features = len(feature_cols)

    train_loader = DataLoader(TrafficDataset(train_df, feature_cols, WINDOW_SIZE, HORIZON),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TrafficDataset(val_df, feature_cols, WINDOW_SIZE, HORIZON),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMForecaster(num_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON, NUM_TARGETS).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, factor=LR_FACTOR)

    mlflow.set_experiment("lstm_forecaster")

    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": LEARNING_RATE, "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "batch_size": BATCH_SIZE, "window_size": WINDOW_SIZE,
            "horizon": HORIZON, "num_features": num_features,
        })

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        patience_count = 0

        print(f"\n{'Epoch':>6} | {'Train MAE':>10} | {'Val MAE':>10} | {'LR':>10}")
        print("-" * 45)

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            lr = optimizer.param_groups[0]["lr"]

            print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {lr:>10.6f}")
            mlflow.log_metric("train_mae", train_loss, step=epoch)
            mlflow.log_metric("val_mae", val_loss, step=epoch)

            scheduler.step(val_loss)

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

        print(f"\nBest Val MAE: {best_val_loss:.4f}")
        mlflow.log_metric("best_val_mae", best_val_loss)
        mlflow.log_artifact(str(CHECKPOINT_PATH))
        mlflow.log_artifact(str(PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.joblib"))

    print("[train_lstm] Done.")


if __name__ == "__main__":
    main()
