# Training Guide — BengaluruFlow

## Order of Operations

Always run in this order (each step produces artifacts the next step uses):

```
1. train_lstm.py          → artifacts/checkpoints/lstm_best.pt
2. train_transformer.py   → artifacts/checkpoints/transformer_best.pt
3. train_baselines.py     → artifacts/baselines/*.joblib
4. train_autoencoder.py   → artifacts/checkpoints/autoencoder_best.pt
                          → artifacts/anomaly_threshold.json
5. evaluate_all.py        → artifacts/evaluation/model_comparison.csv
                          → artifacts/plots/*.png
```

## Individual Commands

```bash
# From project root (bengaluruflow/)

python src/training/train_lstm.py
python src/training/train_transformer.py
python src/training/train_baselines.py
python src/training/train_autoencoder.py
python src/evaluation/evaluate_all.py
```

## All at Once (Linux / macOS / Git Bash)

```bash
bash scripts/run_all.sh
```

## Expected Training Times (CPU, no GPU)

| Model | Approximate Time |
|---|---|
| LSTM Forecaster | 10–20 minutes |
| Transformer Forecaster | 10–15 minutes |
| Random Forest | 2–5 minutes |
| XGBoost | 1–3 minutes |
| Linear Regression | < 1 minute |
| LSTM Autoencoder | 5–10 minutes |

With a GPU (CUDA), all deep learning models train in under 5 minutes each.

## Early Stopping

LSTM and Transformer use early stopping with patience=10.
If validation loss stops improving, training ends early (before 80 epochs).
This is expected behaviour — not an error.

## Viewing MLflow Results

After training, launch the MLflow UI:

```bash
mlflow ui
```

Open `http://localhost:5000`. Each model has its own experiment:
- `lstm_forecaster`
- `transformer_forecaster`
- `baseline_models`
- `anomaly_detector`

## Checkpoints

Best checkpoints are saved as `.pt` files containing:
- `model_state`: PyTorch state dict
- `epoch`: epoch at which best val loss occurred
- `val_loss`: best validation loss
- `num_features`: input feature count (for reconstruction at inference)
- `feature_cols`: ordered list of feature columns used during training
