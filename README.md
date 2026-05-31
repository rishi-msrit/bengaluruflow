# BengaluruFlow 🚦
### Multi-Target Traffic Forecasting and Anomaly Detection on the Bangalore Traffic Pulse Dataset

> *"Bangalore adds 1,000 new vehicles to its roads every single day. The city's traffic management systems are still largely reactive — they respond to congestion after it happens. What if we could predict it before it does?"*

---

## Why This Project Exists

I picked up this dataset expecting to do the usual — some bar charts, a heatmap, maybe a correlation matrix. But the more I looked at it, the more frustrated I got with that approach.

Here's the thing: **the data is a time series**. It has 8,936 daily observations across multiple roads spanning almost three years. Every data point knows what yesterday looked like, and the day before that. Throwing all of that away to make a static pie chart felt like a waste.

So I asked a different question: **can a model actually learn when a road is going to get bad, and flag when something unusual is happening?**

That turned into this project. It's not a research paper. It's a first-year CS student learning PyTorch and trying to do something real with it — and being honest about what worked and what didn't.

---

## The Dataset

**Source**: [Bangalore Traffic Pulse — Kaggle](https://www.kaggle.com/datasets/preethamgouda/banglores-traffic-pulse)  
**8,936 rows × 16 columns | Jan 2022 – Aug 2024 | No missing values**

| Column | What it measures |
|--------|-----------------|
| Traffic Volume | Vehicle count at the road/intersection |
| Average Speed | km/h average |
| Congestion Level | 1–10 scale |
| Travel Time Index | How much longer the journey takes vs free-flow |
| Road Capacity Utilization | % of road capacity being used |
| Incident Reports | Reported accidents or incidents |
| Environmental Impact | Pollution proxy metric |
| Public Transport Usage | Transit ridership at that location |
| Traffic Signal Compliance | How well drivers follow signals |
| Parking Usage | Parked vehicle count |
| Pedestrian and Cyclist Count | Non-motorised road users |
| Weather Conditions | Clear, Overcast, Rain, etc. |
| Roadwork and Construction Activity | Whether construction is active |
| Area Name | Bangalore area (Whitefield, Koramangala, etc.) |
| Road/Intersection Name | Specific road |
| Date | Observation date |

---

## What I Built

Two problems, six models.

**Problem 1 — Forecasting**: Given the last 7 days of traffic data for a road, predict Traffic Volume, Congestion Level, and Average Speed for the next 3 days.

**Problem 2 — Anomaly Detection**: Without any labels, learn what "normal" traffic looks like and automatically flag days where something unusual happened.

| Model | Type | Why it's here |
|-------|------|---------------|
| **LSTM** | Deep Learning | Memory-based sequence model — the main model |
| **Transformer** | Deep Learning | Attention-based — curious if it beats LSTM at T=7 |
| **Linear Regression** | Baseline | If we can't beat this, something's wrong |
| **Random Forest** | Baseline | Strong tree-based benchmark + gives feature importance |
| **XGBoost** | Baseline | State-of-the-art for tabular data |
| **LSTM Autoencoder** | Unsupervised | Learns normal traffic; flags anomalies via reconstruction error |

---

## Results

After training all models on a **chronological 70/15/15 train/val/test split** (no data leakage — future data never touches training):

### Traffic Volume Forecasting (MAE, lower = better)

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|-------|--------|------|
| **LSTM** | **0.683** | 0.849 | 0.242 |
| **Transformer** | **0.678** | 0.846 | 0.247 |
| Linear Regression | 0.690 | 0.852 | 0.236 |
| Random Forest | 0.686 | 0.845 | 0.249 |
| XGBoost | 0.690 | 0.852 | 0.237 |

### Congestion Level Forecasting (MAE, lower = better)

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|-------|--------|------|
| **LSTM** | **0.640** | 0.959 | 0.076 |
| **Transformer** | **0.633** | 0.949 | 0.096 |
| Linear Regression | 0.723 | 0.906 | 0.176 |
| Random Forest | 0.763 | 0.915 | 0.160 |
| XGBoost | 0.715 | 0.918 | 0.153 |

### Average Speed Forecasting

| Model | MAE ↓ | R² ↑ |
|-------|-------|------|
| LSTM | 0.791 | –0.004 |
| Transformer | 0.791 | –0.003 |
| Linear Regression | 0.797 | –0.023 |
| Random Forest | 0.796 | –0.012 |
| XGBoost | 0.803 | –0.038 |

---

## What the Results Actually Say

### The LSTM and Transformer are nearly identical

On Traffic Volume, LSTM gets MAE 0.683 vs Transformer's 0.678 — a difference so small it's noise. At a window size of T=7 days, the Transformer's self-attention doesn't have much of an advantage over the LSTM's sequential memory. Both need longer sequences to show a meaningful difference.

### Deep learning wins on Congestion Level, not Volume

This is the interesting one. On Traffic Volume, all five models are basically tied (0.677–0.690 MAE). But on **Congestion Level**, the gap opens up: LSTM and Transformer achieve 0.633–0.640 MAE while the baselines cluster around 0.715–0.763. That's a ~12% improvement. Congestion seems to have more learnable temporal patterns — it follows daily and weekly cycles more consistently than raw volume.

### Average Speed: nobody wins

Every model has a **negative R²** for Average Speed, meaning they all perform worse than just predicting the mean. This was unexpected. It suggests that Average Speed, in this dataset, is poorly predicted from the feature set alone — possibly because it's more sensitive to real-time incident data that isn't fully captured here. This is an honest finding, not a good one.

### MAPE is lying to you

MAPE values are in the 200–450% range. That looks catastrophic. It isn't — MAPE explodes when true values are near zero. Since we StandardScaled the features, many values sit close to zero, and the percentage calculation becomes meaningless. MAE and RMSE are the reliable metrics here.

### The Random Forest feature importance

Running `evaluate_all.py` generates a feature importance chart. The top predictors consistently include:
- `Travel Time Index` (most correlated with congestion)
- `Road Capacity Utilization`
- `Congestion Level` itself (lagged, in the input window)
- `days_since_start` (the long-term growth trend)
- `day_sin` / `day_cos` (weekly cycle)

Weather matters less than expected — which makes sense for Bangalore where traffic is already near-capacity on most days regardless of conditions.

---

## Anomaly Detection

The LSTM Autoencoder is trained only on normal traffic patterns. At inference:
- Reconstruction error > 95th percentile of validation errors → flagged as anomalous
- Evaluated using `Incident Reports > 0` as weak proxy labels

The Streamlit app lets you explore the top anomalous road/date combinations by reconstruction error.

---

## Feature Engineering Decisions

The raw `Date` column is converted into several derived features:

**Cyclical encoding** (not just integers):  
`day_sin = sin(2π × day_of_week / 7)` and `day_cos = cos(2π × day_of_week / 7)`

Why: If you encode Monday=0 and Sunday=6 as integers, the model thinks they're far apart. They're not — Sunday and Monday are adjacent. Sin/cos puts all days on a circle so the model understands the cycle.

**Karnataka public holidays** (hardcoded 2022–2024):  
Bangalore's holiday traffic behaves like weekend traffic. Festivals like Ugadi, Dussehra, and Diwali cause sharp drops in commercial traffic and spikes in certain corridors.

**StandardScaler fit on training data only**:  
The validation and test sets are scaled using the training set's mean and standard deviation. If you fit the scaler on the full dataset, future information leaks into training — your validation metrics look great but the model would fail in deployment.

---

## Architecture

### LSTM Forecaster
```
Input (batch, 7, num_features)
  → LSTM (128 hidden, 2 layers, dropout 0.2)
  → Take final hidden state
  → Dropout (0.2)
  → Linear → (batch, 3, 3)  ← 3 days × 3 targets
```

### Transformer Forecaster
```
Input (batch, 7, num_features)
  → Linear projection to d_model=64
  → Add learned positional embeddings (7 positions)
  → TransformerEncoder (2 layers, 4 heads, FFN dim 256)
  → Mean pool across time
  → Linear → (batch, 3, 3)
```

### LSTM Autoencoder
```
Input (batch, 7, num_features)
  → Encoder LSTM (hidden=64) → bottleneck Linear(64→32)
  → Decoder Linear(32→64) → repeat × 7 → Decoder LSTM (64→num_features)
  → Reconstruction (batch, 7, num_features)
Anomaly score = MSE(input, reconstruction)
```

---

## Running It Locally

```bash
# 1. Create venv
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 2. Install
pip install -r requirements.txt

# 3. Place the dataset at data/raw/Banglore_traffic_Dataset.csv

# 4. Quick sanity check (no dataset needed)
pytest tests/test_lstm_forward.py tests/test_transformer_forward.py tests/test_autoencoder_forward.py -v

# 5. Train all models (takes 30-60 min on CPU)
python src/training/train_lstm.py
python src/training/train_transformer.py
python src/training/train_baselines.py
python src/training/train_autoencoder.py
python src/evaluation/evaluate_all.py

# 6. Launch Streamlit app
streamlit run app.py

# 7. View MLflow experiment tracking
mlflow ui    # then open http://localhost:5000
```

---

## Streamlit App (Live Demo)

The app has four pages:

**Dataset Overview** — basic stats, traffic volume by area, congestion distribution, day-of-week patterns

**Traffic Forecaster** — select any road in the dataset, choose 1–3 day horizon, click Forecast. Runs LSTM inference on the last 7 available days for that road.

**Anomaly Explorer** — table of top 20 anomalous road/date combinations ranked by reconstruction error, bar chart of error by area

**Model Comparison** — the results table above, rendered as a styled dataframe with MAE bar charts

---

## Project Structure

```
bengaluruflow/
├── data/raw/                      # Banglore_traffic_Dataset.csv
├── src/
│   ├── data/
│   │   ├── loader.py              # Load CSV, validate 16 columns
│   │   ├── features.py            # Feature engineering + scaling
│   │   └── dataset.py             # PyTorch Dataset (sliding window)
│   ├── models/
│   │   ├── lstm_forecaster.py
│   │   ├── transformer_forecaster.py
│   │   └── autoencoder.py
│   ├── training/
│   │   ├── train_lstm.py
│   │   ├── train_transformer.py
│   │   ├── train_baselines.py
│   │   └── train_autoencoder.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate_all.py        # Saves comparison CSV + all plots
│   └── anomaly/
│       ├── score.py
│       └── threshold.py
├── artifacts/
│   ├── checkpoints/               # Saved .pt model weights
│   ├── scalers/                   # feature_scaler.joblib
│   ├── evaluation/                # model_comparison.csv
│   ├── plots/                     # All generated PNG plots
│   └── anomaly_threshold.json
├── tests/                         # 6 pytest files (26 tests total)
├── notebooks/01_eda.ipynb
├── docs/
│   ├── SETUP.md
│   ├── TRAINING.md
│   ├── RESULTS.md
│   ├── RESUME_BULLETS.md
│   └── INTERVIEW_QA.md
├── app.py                         # Streamlit app
├── requirements.txt
└── .gitignore
```

---

## Stack

| Tool | Use |
|------|-----|
| PyTorch 2.2 | LSTM, Transformer, Autoencoder — built from scratch |
| scikit-learn 1.4 | Baselines (LR, RF), StandardScaler, metrics |
| XGBoost 2.0 | Gradient boosting baseline |
| MLflow 2.12 | Experiment tracking (params, metrics, artifacts per run) |
| Streamlit 1.33 | Interactive demo app |
| Pandas + NumPy | Feature engineering, data processing |
| Matplotlib + Seaborn | Evaluation plots |
| pytest | 26 unit tests |

---

## Honest Limitations

- **Negative R² on Average Speed** — the model is worse than guessing the mean. This target probably needs richer features or finer temporal granularity.
- **Daily data** — hourly data would let the model capture morning/evening commute spikes, which is where the real value is.
- **Incident Reports as anomaly labels** — not all incidents cause anomalous traffic and not all anomalous traffic gets reported. The anomaly evaluation is indicative, not definitive.
- **No real-time inference** — this is a batch training and offline prediction system, not a live service.

---

*First-year CS undergraduate project · PyTorch from scratch · No pretrained models · No frameworks beyond what's in requirements.txt*
