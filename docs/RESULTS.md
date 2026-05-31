# Results — BengaluruFlow

## How to Generate Results

Run the full pipeline first (see TRAINING.md), then:

```bash
python src/evaluation/evaluate_all.py
```

This saves:
- `artifacts/evaluation/model_comparison.csv` — per-model, per-target metrics
- `artifacts/plots/pred_vs_actual_lstm.png` — LSTM predicted vs actual
- `artifacts/plots/pred_vs_actual_transformer.png`
- `artifacts/plots/feature_importance_rf.png`
- `artifacts/plots/residuals_lstm.png`
- `artifacts/plots/anomaly_error_histogram.png`
- `artifacts/evaluation/anomaly_results.json`

## Metrics Explained

| Metric | Description | Direction |
|---|---|---|
| MAE | Mean Absolute Error — average absolute prediction error | Lower = better |
| RMSE | Root Mean Squared Error — penalises large errors more | Lower = better |
| MAPE | Mean Absolute Percentage Error — scale-independent, can be misleading near 0 | Lower = better |
| R² | Coefficient of determination — how much variance is explained | Higher = better |
| ROC-AUC | Anomaly detection discriminative power | Higher = better |

## Anomaly Detection Notes

Anomaly labels are **weak pseudo-labels**: `Incident Reports > 0`.
This is an imperfect proxy — not all incidents correspond to anomalous traffic
patterns, and some anomalies may not trigger an incident report.
Interpret ROC-AUC and F1 with this caveat in mind.

The 95th-percentile threshold means:
- ~5% of validation samples are flagged as anomalous
- On test data, this rate may differ

## Reproducibility

Results may vary slightly between runs due to:
- PyTorch random weight initialization (different seed each run)
- Mini-batch ordering (shuffle=True during training)

To make results exactly reproducible, set:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```
at the top of each training script before constructing the DataLoader.
