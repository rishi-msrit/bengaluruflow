#!/usr/bin/env bash
# run_all.sh — Run full BengaluruFlow training pipeline in order.
# Usage: bash scripts/run_all.sh
# Run from project root (bengaluruflow/).

set -e  # Exit immediately if any command fails

echo "============================================"
echo " BengaluruFlow — Full Training Pipeline"
echo "============================================"

echo ""
echo "[1/4] Training LSTM Forecaster..."
python src/training/train_lstm.py

echo ""
echo "[2/4] Training Transformer Forecaster..."
python src/training/train_transformer.py

echo ""
echo "[3/4] Training Baseline Models (LR, RF, XGBoost)..."
python src/training/train_baselines.py

echo ""
echo "[4/4] Training LSTM Autoencoder (Anomaly Detector)..."
python src/training/train_autoencoder.py

echo ""
echo "[5/5] Running Evaluation & Saving Comparison Table..."
python src/evaluation/evaluate_all.py

echo ""
echo "============================================"
echo " All done! Artifacts saved to artifacts/"
echo " View MLflow results: mlflow ui"
echo " Launch Streamlit app: streamlit run app.py"
echo "============================================"
