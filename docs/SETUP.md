# Setup Guide — BengaluruFlow

## Prerequisites

- Python 3.10 or 3.11 (recommended)
- pip 23+
- At least 4 GB RAM (8 GB recommended for training)
- GPU optional — CPU training takes ~10–20 minutes per model

## Step 1 — Clone / Download Project

```bash
git clone <your-repo-url> bengaluruflow
cd bengaluruflow
```

## Step 2 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

All packages are pinned to exact versions for reproducibility.

## Step 4 — Download the Dataset

1. Go to [Kaggle — Bangalore Traffic Pulse](https://www.kaggle.com/datasets/preethamgouda/banglores-traffic-pulse)
2. Download `Banglore_traffic_Dataset.csv`
3. Place it at:

```
bengaluruflow/data/raw/Banglore_traffic_Dataset.csv
```

## Step 5 — Verify Setup

```bash
python -c "
from src.data.loader import load_data
df = load_data()
print(f'Loaded {len(df)} rows, {len(df.columns)} columns. ✓')
"
```

Expected output:
```
[loader] Loaded 8,936 rows × 16 columns from Banglore_traffic_Dataset.csv
Loaded 8936 rows, 16 columns. ✓
```

## Step 6 — Run Tests (no dataset needed for model tests)

```bash
pytest tests/test_lstm_forward.py tests/test_transformer_forward.py tests/test_autoencoder_forward.py -v
```

All 3 test files should pass without the dataset.
