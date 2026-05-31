"""
evaluate_all.py — Run full evaluation across all models and save comparison CSV.

Loads all trained model checkpoints, runs inference on the test set, computes
per-target metrics (MAE, RMSE, MAPE, R²), and saves:
  - artifacts/evaluation/model_comparison.csv
  - artifacts/plots/ (predicted vs actual, loss curves, feature importance,
                       residual distribution, anomaly histogram)

Also evaluates the LSTM Autoencoder (anomaly detection) and reports
ROC-AUC, Precision, Recall, F1 using Incident Reports as weak labels.

Run from project root:
    python src/evaluation/evaluate_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
import json
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data.loader import load_data, check_no_nulls
from src.data.features import build_features
from src.data.dataset import TrafficDataset, AnomalyDataset, WINDOW_SIZE, HORIZON, TARGET_NAMES
from src.models.lstm_forecaster import LSTMForecaster
from src.models.transformer_forecaster import TransformerForecaster
from src.models.autoencoder import LSTMAutoencoder
from src.evaluation.metrics import compute_all_metrics, anomaly_roc_auc
from src.anomaly.score import compute_reconstruction_errors
from src.anomaly.threshold import load_threshold

# ── Artifact paths ─────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
BASELINE_DIR   = PROJECT_ROOT / "artifacts" / "baselines"
EVAL_DIR       = PROJECT_ROOT / "artifacts" / "evaluation"
PLOTS_DIR      = PROJECT_ROOT / "artifacts" / "plots"
THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "anomaly_threshold.json"

BATCH_SIZE     = 64
NUM_TARGETS    = 3


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning Model Inference
# ─────────────────────────────────────────────────────────────────────────────

def dl_predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader and return predictions and ground truth.

    Args:
        model: Trained DL model.
        loader: DataLoader yielding (X, y) pairs.
        device: CPU or CUDA.

    Returns:
        (y_pred, y_true) — both shape (n_samples, H, num_targets).
    """
    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred    = model(x_batch)              # (batch, H, 3)
            preds.append(pred.cpu().numpy())
            truths.append(y_batch.numpy())

    return np.concatenate(preds, axis=0), np.concatenate(truths, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn Baseline Inference
# ─────────────────────────────────────────────────────────────────────────────

def sklearn_predict(
    model,
    test_dataset: TrafficDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run sklearn model inference on flattened test windows.

    Args:
        model: Trained MultiOutputRegressor.
        test_dataset: Test TrafficDataset.

    Returns:
        (y_pred, y_true) — both shape (n_samples, H, num_targets).
    """
    X_list, y_list = [], []
    for x, y in test_dataset.samples:
        X_list.append(x.flatten())
        y_list.append(y)

    X = np.array(X_list)
    y_true_flat = np.array(y_list)                         # (n, H, 3)

    y_pred_flat = model.predict(X).reshape(-1, HORIZON, NUM_TARGETS)
    return y_pred_flat, y_true_flat


# ─────────────────────────────────────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_samples: int = 60,
) -> None:
    """
    Plot predicted vs actual Traffic Volume for the first n_samples test windows.

    Args:
        y_true:     Ground truth, shape (n_samples, H, 3).
        y_pred:     Predictions, shape (n_samples, H, 3).
        model_name: Name used in plot title and filename.
        n_samples:  Number of windows to display.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Traffic Volume is index 0 among targets
    # Show step H=1 predictions (first day of horizon) across first n_samples
    tv_true = y_true[:n_samples, 0, 0]
    tv_pred = y_pred[:n_samples, 0, 0]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(tv_true, label="Actual", color="#1f77b4", linewidth=1.5)
    ax.plot(tv_pred, label="Predicted", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_title(f"{model_name} — Traffic Volume: Predicted vs Actual (first {n_samples} test samples)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Traffic Volume (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_path = PLOTS_DIR / f"pred_vs_actual_{safe_name}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[evaluate_all] Plot saved → {save_path}")


def plot_feature_importance(rf_model, feature_cols: list[str]) -> None:
    """
    Bar chart of Random Forest feature importances (top 20).

    Args:
        rf_model:     Trained MultiOutputRegressor wrapping RandomForestRegressor.
        feature_cols: Feature column names.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Average importance across all output estimators
    importances = np.mean(
        [est.feature_importances_ for est in rf_model.estimators_], axis=0
    )

    # The feature vector is T*F; take importances for the first T=7 windows
    # Each block of len(feature_cols) corresponds to one time step
    n_features = len(feature_cols)
    # Average across T time steps
    step_importances = np.zeros(n_features)
    for t in range(WINDOW_SIZE):
        start = t * n_features
        end   = start + n_features
        if end <= len(importances):
            step_importances += importances[start:end]
    step_importances /= WINDOW_SIZE

    top_n = 20
    top_idx   = np.argsort(step_importances)[::-1][:top_n]
    top_names = [feature_cols[i] for i in top_idx]
    top_vals  = step_importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_vals, y=top_names, palette="viridis", ax=ax)
    ax.set_title("Random Forest — Top 20 Feature Importances (averaged over window)")
    ax.set_xlabel("Mean importance")
    plt.tight_layout()

    save_path = PLOTS_DIR / "feature_importance_rf.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[evaluate_all] Plot saved → {save_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Residual distribution histogram for the best model on Traffic Volume.

    Args:
        y_true:     Ground truth, shape (n_samples, H, 3).
        y_pred:     Predictions, shape (n_samples, H, 3).
        model_name: Model name for title and filename.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    residuals = (y_true[:, :, 0] - y_pred[:, :, 0]).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="#2ca02c", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{model_name} — Residuals for Traffic Volume")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    save_path = PLOTS_DIR / f"residuals_{safe_name}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[evaluate_all] Plot saved → {save_path}")


def plot_anomaly_histogram(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold: float,
) -> None:
    """
    Reconstruction error histogram: normal vs flagged anomalies.

    Args:
        normal_errors:  Reconstruction errors for non-incident samples.
        anomaly_errors: Reconstruction errors for incident samples.
        threshold:      The anomaly threshold to draw as a vertical line.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(normal_errors, bins=50, color="#1f77b4", label="Normal", alpha=0.6, ax=ax)
    if len(anomaly_errors) > 0:
        sns.histplot(anomaly_errors, bins=50, color="#d62728", label="Anomalous (incident)", alpha=0.6, ax=ax)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold (P95={threshold:.4f})")
    ax.set_title("Reconstruction Error Distribution — Normal vs Anomalous")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    save_path = PLOTS_DIR / "anomaly_error_histogram.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[evaluate_all] Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run full evaluation across all models and save artifacts."""

    device = get_device()
    print(f"[evaluate_all] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[evaluate_all] Loading data...")
    df = load_data()
    check_no_nulls(df)
    train_df, val_df, test_df, feature_cols, scaler, _ = build_features(df)
    num_features = len(feature_cols)

    test_dataset = TrafficDataset(test_df, feature_cols, WINDOW_SIZE, HORIZON)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_results: list[dict] = []

    # ── LSTM Forecaster ────────────────────────────────────────────────────────
    lstm_ckpt_path = CHECKPOINT_DIR / "lstm_best.pt"
    if lstm_ckpt_path.exists():
        print("\n[evaluate_all] Evaluating LSTM Forecaster...")
        ckpt = torch.load(lstm_ckpt_path, map_location=device)
        lstm_model = LSTMForecaster(
            input_size=num_features, hidden_size=128, num_layers=2,
            dropout=0.2, horizon=HORIZON, num_targets=NUM_TARGETS
        ).to(device)
        lstm_model.load_state_dict(ckpt["model_state"])
        y_pred, y_true = dl_predict(lstm_model, test_loader, device)

        metrics = compute_all_metrics(y_true, y_pred, TARGET_NAMES, HORIZON)
        metrics["model"] = "LSTM"
        all_results.append(metrics)

        plot_pred_vs_actual(y_true, y_pred, "LSTM")
        plot_residuals(y_true, y_pred, "LSTM")
    else:
        print(f"[evaluate_all] LSTM checkpoint not found at {lstm_ckpt_path}. Skipping.")

    # ── Transformer Forecaster ────────────────────────────────────────────────
    tf_ckpt_path = CHECKPOINT_DIR / "transformer_best.pt"
    if tf_ckpt_path.exists():
        print("\n[evaluate_all] Evaluating Transformer Forecaster...")
        ckpt = torch.load(tf_ckpt_path, map_location=device)
        tf_model = TransformerForecaster(
            input_size=num_features, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=256, dropout=0.1,
            window_size=WINDOW_SIZE, horizon=HORIZON, num_targets=NUM_TARGETS
        ).to(device)
        tf_model.load_state_dict(ckpt["model_state"])
        y_pred, y_true = dl_predict(tf_model, test_loader, device)

        metrics = compute_all_metrics(y_true, y_pred, TARGET_NAMES, HORIZON)
        metrics["model"] = "Transformer"
        all_results.append(metrics)

        plot_pred_vs_actual(y_true, y_pred, "Transformer")
    else:
        print(f"[evaluate_all] Transformer checkpoint not found. Skipping.")

    # ── Baseline Models ───────────────────────────────────────────────────────
    baseline_configs = [
        ("linear_regression", "Linear Regression"),
        ("random_forest",     "Random Forest"),
        ("xgboost",           "XGBoost"),
    ]
    rf_model_loaded = None

    for filename, display_name in baseline_configs:
        model_path = BASELINE_DIR / f"{filename}.joblib"
        if model_path.exists():
            print(f"\n[evaluate_all] Evaluating {display_name}...")
            bl_model = joblib.load(model_path)
            y_pred, y_true = sklearn_predict(bl_model, test_dataset)

            metrics = compute_all_metrics(y_true, y_pred, TARGET_NAMES, HORIZON)
            metrics["model"] = display_name
            all_results.append(metrics)

            if filename == "random_forest":
                rf_model_loaded = bl_model
        else:
            print(f"[evaluate_all] {display_name} model not found. Skipping.")

    # ── Feature Importance (Random Forest) ───────────────────────────────────
    if rf_model_loaded is not None:
        plot_feature_importance(rf_model_loaded, feature_cols)

    # ── Save Model Comparison CSV ─────────────────────────────────────────────
    if all_results:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        comparison_df = pd.DataFrame(all_results)
        # Put 'model' column first
        cols = ["model"] + [c for c in comparison_df.columns if c != "model"]
        comparison_df = comparison_df[cols]
        csv_path = EVAL_DIR / "model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n[evaluate_all] Model comparison saved → {csv_path}")
        print(comparison_df.to_string(index=False))

    # ── Anomaly Detector Evaluation ───────────────────────────────────────────
    ae_ckpt_path = CHECKPOINT_DIR / "autoencoder_best.pt"
    if ae_ckpt_path.exists() and THRESHOLD_PATH.exists():
        print("\n[evaluate_all] Evaluating LSTM Autoencoder (Anomaly Detection)...")

        ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae_model = LSTMAutoencoder(
            num_features=num_features, hidden_size=64,
            bottleneck_dim=32, window_size=WINDOW_SIZE
        ).to(device)
        ae_model.load_state_dict(ckpt["model_state"])

        test_anomaly_dataset = AnomalyDataset(test_df, feature_cols, WINDOW_SIZE)
        test_anomaly_loader  = DataLoader(test_anomaly_dataset, batch_size=BATCH_SIZE, shuffle=False)

        errors    = compute_reconstruction_errors(ae_model, test_anomaly_loader, device)
        threshold = load_threshold(THRESHOLD_PATH)

        # Weak labels: incident_reports > 0 → anomalous
        labels    = np.array([1 if m["incidents"] > 0 else 0
                              for m in test_anomaly_dataset.metadata])
        pred_labels = (errors > threshold).astype(int)

        roc_auc = anomaly_roc_auc(errors, labels)
        prec    = precision_score(labels, pred_labels, zero_division=0)
        rec     = recall_score(labels, pred_labels, zero_division=0)
        f1      = f1_score(labels, pred_labels, zero_division=0)

        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")

        # Anomaly histogram
        normal_mask  = labels == 0
        anomaly_mask = labels == 1
        plot_anomaly_histogram(
            errors[normal_mask], errors[anomaly_mask], threshold
        )

        # Save anomaly results
        anomaly_results = {
            "roc_auc": roc_auc, "precision": prec,
            "recall": rec, "f1": f1,
            "threshold": threshold,
            "n_flagged": int(pred_labels.sum()),
            "n_total":   int(len(pred_labels)),
        }
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVAL_DIR / "anomaly_results.json", "w") as f:
            json.dump(anomaly_results, f, indent=2)
    else:
        print("[evaluate_all] Autoencoder checkpoint or threshold not found. Skipping anomaly eval.")

    print("\n[evaluate_all] Evaluation complete. ✓")


if __name__ == "__main__":
    main()
