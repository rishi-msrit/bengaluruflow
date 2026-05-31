# BengaluruFlow: Multi-Target Traffic Forecasting and Anomaly Detection

BengaluruFlow applies deep learning to real Bangalore traffic data — going beyond
exploratory analysis to build and compare LSTM, Transformer, and autoencoder models
for forecasting and anomaly detection on the **Bangalore Traffic Pulse** dataset.

---

## Motivation

Most traffic dataset projects stop at visualizations and dashboards.
This project treats the same data as a **supervised time-series ML problem**,
implementing multi-output forecasters and an unsupervised anomaly detector
from scratch in PyTorch, with full baseline comparisons and experiment tracking.

---

## Dataset

**Source**: [Bangalore Traffic Pulse — Kaggle (preethamgouda)](https://www.kaggle.com/datasets/preethamgouda/banglores-traffic-pulse)  
**File**: `Banglore_traffic_Dataset.csv`  
**Size**: 8,936 rows × 16 columns, Jan 2022 – Aug 2024, no missing values

| Column | Type | Description |
|---|---|---|
| Date | datetime | Observation date |
| Area Name | categorical | Bangalore area (e.g., Whitefield, Koramangala) |
| Road/Intersection Name | categorical | Specific road or intersection |
| Traffic Volume | numeric | Vehicle count |
| Average Speed | numeric | km/h average |
| Travel Time Index | numeric | Congestion indicator |
| Congestion Level | numeric | 1–10 scale |
| Road Capacity Utilization | numeric | % of road capacity used |
| Incident Reports | numeric | Reported incidents (used as anomaly label) |
| Environmental Impact | numeric | Pollution proxy |
| Public Transport Usage | numeric | Transit ridership |
| Traffic Signal Compliance | numeric | Signal adherence rate |
| Parking Usage | numeric | Parked vehicles |
| Pedestrian and Cyclist Count | numeric | Non-motorised count |
| Weather Conditions | categorical | Clear, Overcast, Rain, etc. |
| Roadwork and Construction Activity | categorical | active / none |

---

## ML Approach

| Model | Type | Purpose |
|---|---|---|
| **LSTM Forecaster** | 2-layer LSTM | Multi-step, multi-output traffic forecasting |
| **Transformer Forecaster** | nn.TransformerEncoder | Comparison against LSTM |
| **LSTM Autoencoder** | Encoder-Decoder LSTM | Unsupervised anomaly detection |
| **Linear Regression** | sklearn baseline | Lower bound reference |
| **Random Forest** | sklearn baseline | Strong non-DL baseline |
| **XGBoost** | sklearn baseline | Gradient boosting baseline |

**Forecasting**: Given T=7 days of features → predict H=3 future days of
[Traffic Volume, Congestion Level, Average Speed]

**Anomaly Detection**: Reconstruction error > 95th percentile threshold
(from validation set) → flagged as anomalous.

---

## Installation

```bash
# 1. Clone or download this project
# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Download Banglore_traffic_Dataset.csv from Kaggle and put it at:
# data/raw/Banglore_traffic_Dataset.csv
```

---

## Training

Run each model separately, or use the convenience script:

```bash
# Train LSTM Forecaster
python src/training/train_lstm.py

# Train Transformer Forecaster
python src/training/train_transformer.py

# Train baseline models (LR, RF, XGBoost)
python src/training/train_baselines.py

# Train LSTM Autoencoder (anomaly detector)
python src/training/train_autoencoder.py

# Run evaluation and generate comparison table + plots
python src/evaluation/evaluate_all.py

# Or run everything at once (Linux/macOS):
bash scripts/run_all.sh
```

---

## Streamlit App

```bash
streamlit run app.py
```

**Pages**:
1. Dataset Overview — stats, distributions, day-of-week patterns
2. Traffic Forecaster — LSTM forecast for any road (1–3 days)
3. Anomaly Explorer — top anomalous road/date combinations
4. Model Comparison — styled comparison table and MAE bar charts

---

## MLflow Results Viewer

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser. You will see four experiments:
- `lstm_forecaster`
- `transformer_forecaster`
- `baseline_models`
- `anomaly_detector`

---

## Results (Example — actual values depend on your run)

| Model | Traffic Volume MAE | Congestion Level MAE | Avg Speed MAE |
|---|---|---|---|
| LSTM | *~0.XX* | *~0.XX* | *~0.XX* |
| Transformer | *~0.XX* | *~0.XX* | *~0.XX* |
| Linear Regression | *~0.XX* | *~0.XX* | *~0.XX* |
| Random Forest | *~0.XX* | *~0.XX* | *~0.XX* |
| XGBoost | *~0.XX* | *~0.XX* | *~0.XX* |

*Train the models to populate `artifacts/evaluation/model_comparison.csv`.*

---

## Running Tests

```bash
pytest tests/ -v
```

Model forward-pass tests (test_lstm_forward, test_transformer_forward,
test_autoencoder_forward) run without the dataset.
Loader and feature tests require the CSV file.

---

## Project Structure

```
bengaluruflow/
├── data/raw/               # Place Banglore_traffic_Dataset.csv here
├── notebooks/01_eda.ipynb  # Exploratory data analysis
├── src/
│   ├── data/               # loader.py, features.py, dataset.py
│   ├── models/             # lstm_forecaster.py, transformer_forecaster.py, autoencoder.py
│   ├── training/           # train_lstm.py, train_transformer.py, train_baselines.py, train_autoencoder.py
│   ├── evaluation/         # metrics.py, evaluate_all.py
│   └── anomaly/            # score.py, threshold.py
├── artifacts/
│   ├── checkpoints/        # Saved model weights (.pt)
│   ├── scalers/            # feature_scaler.joblib
│   ├── evaluation/         # model_comparison.csv
│   └── plots/              # All generated plots (.png)
├── tests/                  # pytest unit tests (6 files)
├── docs/                   # README, SETUP, TRAINING, RESULTS, RESUME, QA
├── app.py                  # Streamlit demo app
├── mlruns/                 # MLflow tracking directory (auto-created)
├── requirements.txt
└── scripts/run_all.sh
```

---

## Future Work

1. **Hourly granularity**: Collect or simulate hourly data for finer-grained forecasting.
2. **Spatial features**: Add road adjacency information (simplified graph features) to model traffic propagation.
3. **External data**: Incorporate festival calendars, weather APIs, and sports event schedules.
4. **Online learning**: Update models incrementally as new data arrives, rather than full retraining.
5. **Better anomaly labels**: Collaborate with traffic management authorities for verified incident labels instead of using Incident Reports as a weak proxy.
6. **Uncertainty quantification**: Add Monte Carlo Dropout or conformal prediction to produce confidence intervals on forecasts.
7. **TorchScript export**: Package models for deployment on edge devices.

---

## Honest Scope Statement

This is a **student research project**, not a production system.
Models are trained on historical data and produce indicative forecasts.
Results have not been validated against real-time traffic management decisions.
No real-time data feeds or deployment infrastructure are included.

---

*Built by a first-year CS undergraduate as a self-directed deep learning project.*  
*Stack: PyTorch 2.2 · scikit-learn 1.4 · XGBoost 2.0 · MLflow 2.12 · Streamlit 1.33*
