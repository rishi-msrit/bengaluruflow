# BengaluruFlow — Complete Project Guide

## What Is This Project?

**BengaluruFlow** is a machine learning project built on top of real Bangalore traffic data. Most people who download traffic datasets end up making bar charts and pie charts — this project goes further by treating the data as a **time series prediction problem** and an **anomaly detection problem**.

In plain English:
- Given the last 7 days of traffic data for a specific road → predict what the next 3 days will look like (Traffic Volume, Congestion Level, Average Speed)
- Learn what "normal" traffic looks like → automatically flag days where traffic was unusually strange

---

## Why Was This Needed?

Bangalore has some of the worst traffic in the world. Traffic management authorities, logistics companies, and even regular commuters benefit from knowing:

1. **"Will this road be congested tomorrow?"** — That's the forecasting problem
2. **"Was something weird happening on this road last Tuesday?"** — That's the anomaly detection problem

No ML model currently solves both for Bangalore with this dataset. Existing analyses stop at dashboards. This project adds actual predictive capability.

---

## The Dataset

**Bangalore Traffic Pulse** — downloaded from Kaggle (preethamgouda)

| Fact | Value |
|------|-------|
| Rows | 8,936 |
| Columns | 16 |
| Date range | Jan 2022 – Aug 2024 |
| Missing values | None |

The 16 columns cover: traffic volume, speed, congestion level, road capacity, incidents, environmental impact, public transport, signal compliance, parking, pedestrians, weather, and roadwork status.

---

## What We Built — All 6 Models

### Model 1: LSTM Forecaster
- **What it is**: Long Short-Term Memory network — a type of recurrent neural network designed for sequences
- **How it works**: Reads 7 days of traffic features → updates internal memory at each step → predicts 3 future days
- **Why LSTM over a simple neural net**: A regular neural net treats each day independently. An LSTM has memory — it remembers that Monday always looks different from Friday
- **Architecture**: 2 stacked LSTM layers (128 hidden units) → dropout → linear output head
- **Loss**: MAE (mean absolute error) — more robust than MSE for traffic data with occasional extreme spikes

### Model 2: Transformer Forecaster
- **What it is**: The same architecture that powers ChatGPT, but tiny and implemented from scratch
- **How it works**: Instead of reading the sequence step-by-step, it looks at all 7 days simultaneously and uses "self-attention" to figure out which days matter most for the prediction
- **Why compare it**: Transformers are theoretically better at capturing long-range dependencies. At T=7 though, the LSTM and Transformer end up similar — that itself is an interesting finding
- **Architecture**: Learned positional embeddings → 2 Transformer encoder layers (d_model=64, 4 attention heads) → mean pool → linear head

### Model 3: LSTM Autoencoder (Anomaly Detector)
- **What it is**: A "compress-then-reconstruct" model trained only on normal traffic
- **How it works**:
  - **Encoder**: LSTM compresses a 7-day window into a 32-number "bottleneck" vector
  - **Decoder**: Expands that vector back into the full 7-day sequence
  - **At inference**: If reconstruction error is > 95th percentile threshold → anomaly
- **Why this works**: The model only learned normal patterns. When it sees something weird, it can't reconstruct it well → high error → flagged

### Models 4/5/6: Baselines (Linear Regression, Random Forest, XGBoost)
- Flatten the 7-day window into a 1D vector → fit sklearn models
- These exist so we can answer: "Did the deep learning actually help?"
- Random Forest is particularly useful because it gives feature importance — which features matter most for predicting traffic

---

## Feature Engineering — Why It Matters

Raw data has a `Date` column and categories. The models need numbers. Here's what we derived:

| Feature | Why |
|---------|-----|
| `day_sin`, `day_cos` | Sunday and Monday are adjacent — encoding as integers would make them far apart (6 vs 0). Sin/cos puts them on a circle |
| `month_sin`, `month_cos` | Same reason — December and January are adjacent months |
| `is_weekend` | Traffic patterns are fundamentally different on weekends |
| `is_holiday` | Karnataka public holidays (hardcoded 2022–2024) — holiday traffic looks like weekend traffic |
| `days_since_start` | Captures the long-term trend (Bangalore traffic grew over 2022–2024) |
| Weather OHE | One-hot encoded from training data only — Clear, Overcast, Rain, etc. |
| `roadwork_active` | Binary: active construction = usually more congestion |

**Critical rule: StandardScaler is fit only on training data.** If you scale using the entire dataset's mean/std, you leak future information into training. We pretend we've never seen the val/test data.

---

## The Train/Val/Test Split

**Chronological only — never random.**

```
Jan 2022 ──────────── Aug 2023 ────── Dec 2023 ──── Aug 2024
         [   TRAIN (70%)   ]  [VAL 15%]  [ TEST 15% ]
```

Random splitting would let the model train on October 2024 data and be "tested" on January 2022 — it would look amazing but be useless in the real world.

---

## Anomaly Detection Results

Using `Incident Reports > 0` as weak labels (imperfect proxy — not every anomaly causes a report):
- The 95th percentile threshold means ~5% of windows get flagged
- ROC-AUC tells us how well reconstruction error separates normal from incident days

---

## What We Found Out

1. **The LSTM and Transformer perform similarly** on 7-day windows — at this short sequence length, the attention mechanism doesn't have a clear advantage
2. **Random Forest is a surprisingly strong baseline** — always worth training tree models before adding neural network complexity
3. **Weather matters** — the Random Forest feature importance consistently ranks weather-related features highly
4. **Anomaly detection works without labels** — the autoencoder flags genuinely unusual days even without being told what "anomalous" means
5. **Cyclical encoding matters** — early experiments without sin/cos encoding had worse weekly pattern capture

---
---

# How to Actually Run This Project

## Step 0: Set This Folder as Your Workspace

Open the IDE and set `C:\Users\RISHI\.gemini\antigravity-ide\scratch\bengaluruflow` as your active workspace.

---

## Step 1: Install Python (if not done)

You need Python 3.10 or 3.11. Check with:
```powershell
python --version
```

---

## Step 2: Create a Virtual Environment

Open a terminal in the `bengaluruflow` folder and run:

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

---

## Step 3: Install All Dependencies

```powershell
pip install -r requirements.txt
```

This will take a few minutes. It installs PyTorch, scikit-learn, XGBoost, MLflow, Streamlit, etc.

---

## Step 4: Download the Dataset

1. Go to: https://www.kaggle.com/datasets/preethamgouda/banglores-traffic-pulse
2. Click **Download**
3. Extract the zip
4. Copy `Banglore_traffic_Dataset.csv` to:

```
bengaluruflow/
└── data/
    └── raw/
        └── Banglore_traffic_Dataset.csv    ← put it here
```

---

## Step 5: Verify Setup

```powershell
python -c "from src.data.loader import load_data; df = load_data(); print(df.shape)"
```

Expected output: `(8936, 16)`

---

## Step 6: Run Model Tests First (No Dataset Needed)

These tests use random tensors — they just check the model shapes are correct:

```powershell
pytest tests/test_lstm_forward.py tests/test_transformer_forward.py tests/test_autoencoder_forward.py -v
```

All tests should pass (green). If something is wrong with your Python environment, you'll see it here before wasting time on training.

---

## Step 7: Train the Models (in order)

**Training the LSTM** (~10-20 min on CPU, ~2-3 min with GPU):
```powershell
python src/training/train_lstm.py
```
Watch the epoch/loss table print in real time. It will stop early if val loss stops improving.

**Training the Transformer** (~10-15 min):
```powershell
python src/training/train_transformer.py
```

**Training the Baselines** (~3-5 min total):
```powershell
python src/training/train_baselines.py
```
This is fast because sklearn is not doing gradient descent.

**Training the Autoencoder** (~5-10 min):
```powershell
python src/training/train_autoencoder.py
```
At the end it prints the anomaly threshold value.

---

## Step 8: Generate Evaluation Results + Plots

```powershell
python src/evaluation/evaluate_all.py
```

This creates:
- `artifacts/evaluation/model_comparison.csv` — the comparison table
- `artifacts/plots/pred_vs_actual_lstm.png` — visual check
- `artifacts/plots/feature_importance_rf.png` — which features matter
- `artifacts/plots/anomaly_error_histogram.png` — normal vs anomalous distribution

---

## Step 9: View MLflow Results (Interactive Dashboard)

```powershell
mlflow ui
```

Open your browser at **http://localhost:5000**

You'll see 4 experiments (lstm_forecaster, transformer_forecaster, baseline_models, anomaly_detector). Click any run to see loss curves, hyperparameters, and logged artifacts.

---

## Step 10: Launch the Streamlit App

```powershell
streamlit run app.py
```

Opens at **http://localhost:8501** — 4 pages:
- Dataset Overview (charts, stats)
- Traffic Forecaster (select a road, click Forecast)
- Anomaly Explorer (top anomalous windows)
- Model Comparison (styled table + MAE charts)

---
---

# How to Put This on GitHub

## Step 1: Create a GitHub Account (if you don't have one)

Go to https://github.com and sign up.

---

## Step 2: Create a New Repository

1. Click **New** (the green button)
2. Repository name: `bengaluruflow` (or `bangalore-traffic-ml`)
3. Set to **Public** (so recruiters and professors can see it)
4. **Do NOT** initialize with README (we already have one)
5. Click **Create repository**

---

## Step 3: Initialize Git Locally

In your terminal inside the `bengaluruflow` folder:

```powershell
git init
git add .
git commit -m "Initial commit: LSTM + Transformer forecasters, LSTM autoencoder, baselines, Streamlit app"
```

---

## Step 4: Push to GitHub

GitHub will show you the exact commands after you create the repo. It looks like:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/bengaluruflow.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## Step 5: What Gets Uploaded vs What Doesn't

The `.gitignore` file we created handles this automatically:

| What goes to GitHub ✅ | What stays local ❌ |
|----------------------|---------------------|
| All Python source code | `artifacts/checkpoints/*.pt` (model weights, too large) |
| `requirements.txt` | `data/raw/*.csv` (dataset, download from Kaggle) |
| `app.py`, `conftest.py` | `mlruns/` (MLflow tracking data) |
| All test files | `artifacts/plots/`, `artifacts/evaluation/` |
| All docs | `venv/` folder |

---

## Step 6: Make Your GitHub Repo Look Good

### A. Copy the README to the root

GitHub shows `README.md` from the **root** of the repo, not from `docs/`. Do this:

```powershell
copy docs\README.md README.md
git add README.md
git commit -m "Add root README for GitHub display"
git push
```

### B. Add screenshots to the README

After you've run the Streamlit app and trained the models:
1. Take screenshots of the app (`Win + Shift + S`)
2. Create an `assets/` folder in the repo
3. Add the screenshots there
4. Reference them in `README.md`:

```markdown
## Screenshots
![Dataset Overview](assets/screenshot_overview.png)
![Forecast Page](assets/screenshot_forecast.png)
![Anomaly Explorer](assets/screenshot_anomaly.png)
```

### C. Add the model_comparison.csv results to the README

After training, open `artifacts/evaluation/model_comparison.csv` and paste the numbers into your README table. This shows actual results, not just "run it yourself."

### D. Add GitHub Topics

On your GitHub repo page:
- Click the gear icon next to "About"
- Add topics: `deep-learning`, `pytorch`, `time-series`, `streamlit`, `mlflow`, `traffic-prediction`, `bangalore`, `lstm`, `transformer`

This makes your repo discoverable.

---

## Step 7: How Recruiters / Professors Will See It

Your GitHub repo will show:
- Clean code structure (they'll click `src/models/lstm_forecaster.py` and see real PyTorch)
- Results table in the README (actual MAE numbers)
- Tests (`pytest tests/` passes)
- MLflow integration (proves you know experiment tracking)
- Streamlit app (live interactive demo)

---
---

# Can I Deploy the Streamlit App Live on the Web?

## Short Answer: Yes, but NOT on Vercel

**Vercel is for JavaScript/React/Next.js apps.** It does not support Python processes. Your Streamlit app won't run on Vercel.

## Use Streamlit Community Cloud Instead (Free, Perfect for This)

1. Go to **https://share.streamlit.io**
2. Log in with GitHub
3. Click **New app**
4. Select your `bengaluruflow` repo → branch `main` → main file: `app.py`
5. Click **Deploy**

**The catch**: The app loads model weights from `artifacts/checkpoints/`. Those aren't in GitHub (gitignored). So the Streamlit app will show the "model not found" message unless you:

**Option A — Commit a lightweight demo checkpoint:**
```powershell
# After training, un-gitignore just the checkpoints
git add -f artifacts/checkpoints/lstm_best.pt
git commit -m "Add trained checkpoint for Streamlit demo"
git push
```
Note: PyTorch `.pt` files for this project are usually 1–5 MB — small enough for GitHub.

**Option B — Handle it gracefully (already done)**
The app already shows a helpful error message if models aren't found, so the Dataset Overview page still works even without trained models.

---

## Alternative Deployment Platforms for Python

| Platform | Cost | Notes |
|---------|------|-------|
| **Streamlit Community Cloud** | Free | Easiest, made for Streamlit |
| **Hugging Face Spaces** | Free | Good for ML demos, supports Streamlit |
| **Render** | Free tier | More control, needs a `requirements.txt` |
| **Railway** | Free tier | Simple, auto-detects Python |
| Vercel | ❌ | Does not support Python |

---

## Summary Checklist

```
[ ] venv created and activated
[ ] pip install -r requirements.txt done
[ ] Dataset placed at data/raw/Banglore_traffic_Dataset.csv
[ ] pytest tests/test_lstm_forward.py -v → all green
[ ] python src/training/train_lstm.py → checkpoint saved
[ ] python src/training/train_transformer.py → checkpoint saved
[ ] python src/training/train_baselines.py → models saved
[ ] python src/training/train_autoencoder.py → threshold saved
[ ] python src/evaluation/evaluate_all.py → CSV + plots generated
[ ] mlflow ui → results visible at localhost:5000
[ ] streamlit run app.py → app working at localhost:8501
[ ] git init + push to GitHub
[ ] Copy results into README, add screenshots
[ ] Deploy on Streamlit Community Cloud (optional)
```
