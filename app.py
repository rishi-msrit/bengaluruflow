"""
app.py — BengaluruFlow Streamlit Demo App.

Four pages:
  1. Dataset Overview     — stats, distributions, and key plots
  2. Traffic Forecaster   — LSTM inference for a selected road
  3. Anomaly Explorer     — top anomalous road/date combos
  4. Model Comparison     — comparison table and bar charts

Run from project root:
    streamlit run app.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── Add project root to Python path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Artifact and data paths ───────────────────────────────────────────────────
DATA_PATH         = PROJECT_ROOT / "data" / "raw" / "Banglore_traffic_Dataset.csv"
LSTM_CKPT         = PROJECT_ROOT / "artifacts" / "checkpoints" / "lstm_best.pt"
AE_CKPT           = PROJECT_ROOT / "artifacts" / "checkpoints" / "autoencoder_best.pt"
SCALER_PATH       = PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.joblib"
WEATHER_COLS_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "weather_ohe_columns.json"
THRESHOLD_PATH    = PROJECT_ROOT / "artifacts" / "anomaly_threshold.json"
COMPARISON_CSV    = PROJECT_ROOT / "artifacts" / "evaluation" / "model_comparison.csv"

WINDOW_SIZE  = 7
HORIZON      = 3
NUM_TARGETS  = 3
TARGET_NAMES = ["Traffic Volume", "Congestion Level", "Average Speed"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BengaluruFlow",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #4fc3f7; }
    h2, h3 { color: #81d4fa; }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #263248);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2e4060;
        text-align: center;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data() -> pd.DataFrame | None:
    """Load and cache the raw dataset. Returns None on failure."""
    if not DATA_PATH.exists():
        return None
    from src.data.loader import load_data
    return load_data()


@st.cache_resource
def load_lstm_model():
    """Load and cache the LSTM forecaster checkpoint. Returns None on failure."""
    if not LSTM_CKPT.exists():
        return None, None, None
    from src.models.lstm_forecaster import LSTMForecaster

    ckpt         = torch.load(LSTM_CKPT, map_location="cpu")
    num_features = ckpt["num_features"]
    feature_cols = ckpt["feature_cols"]
    model = LSTMForecaster(
        input_size  = num_features,
        hidden_size = 128,
        num_layers  = 2,
        dropout     = 0.0,   # disable dropout at inference
        horizon     = HORIZON,
        num_targets = NUM_TARGETS,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feature_cols, num_features


@st.cache_resource
def load_autoencoder():
    """Load and cache the LSTM Autoencoder. Returns None on failure."""
    if not AE_CKPT.exists():
        return None, None, None
    from src.models.autoencoder import LSTMAutoencoder

    ckpt         = torch.load(AE_CKPT, map_location="cpu")
    num_features = ckpt["num_features"]
    feature_cols = ckpt["feature_cols"]
    model = LSTMAutoencoder(
        num_features   = num_features,
        hidden_size    = 64,
        bottleneck_dim = 32,
        window_size    = WINDOW_SIZE,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feature_cols, num_features


@st.cache_resource
def load_scaler_and_weather():
    """Load scaler and weather OHE column list."""
    if not SCALER_PATH.exists() or not WEATHER_COLS_PATH.exists():
        return None, None
    scaler       = joblib.load(SCALER_PATH)
    with open(WEATHER_COLS_PATH) as f:
        weather_cols = json.load(f)
    return scaler, weather_cols


@st.cache_data
def load_comparison_csv() -> pd.DataFrame | None:
    """Load and cache the model comparison CSV."""
    if not COMPARISON_CSV.exists():
        return None
    return pd.read_csv(COMPARISON_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Bangalore_Vidhana_Soudha.jpg/640px-Bangalore_Vidhana_Soudha.jpg",
                 use_column_width=True)
st.sidebar.title("🚦 BengaluruFlow")
st.sidebar.markdown("*Traffic Forecasting & Anomaly Detection*")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio(
    "Navigate",
    ["📊 Dataset Overview", "🔮 Traffic Forecaster", "⚠️ Anomaly Explorer", "📈 Model Comparison"],
    key="page_nav",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: Bangalore Traffic Pulse  \n"
    "**Models**: LSTM · Transformer · Autoencoder  \n"
    "**Stack**: PyTorch · Scikit-learn · MLflow"
)


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_dataset_overview():
    """Render the Dataset Overview page."""
    st.title("📊 Dataset Overview")
    st.markdown("**Bangalore Traffic Pulse** — Jan 2022 to Aug 2024, 8,936 rows, 16 columns.")

    df = load_raw_data()
    if df is None:
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Please download `Banglore_traffic_Dataset.csv` from Kaggle and place it at the path above."
        )
        return

    # ── Key stats ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows",    f"{len(df):,}")
    with col2:
        st.metric("Columns",       f"{len(df.columns)}")
    with col3:
        st.metric("Date Range",    f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    with col4:
        st.metric("Unique Roads",  f"{df['Road/Intersection Name'].nunique():,}")

    st.markdown("---")

    # ── First 20 rows ─────────────────────────────────────────────────────────
    st.subheader("Sample Data (first 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)

    # ── Basic stats ────────────────────────────────────────────────────────────
    st.subheader("Descriptive Statistics")
    numeric_cols = df.select_dtypes(include="number").columns
    st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)

    st.markdown("---")

    # ── Plot 1: Traffic Volume by Area Name ────────────────────────────────────
    st.subheader("Average Traffic Volume by Area")
    area_agg = df.groupby("Area Name")["Traffic Volume"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    area_agg.plot(kind="bar", ax=ax, color=sns.color_palette("Blues_r", len(area_agg)))
    ax.set_xlabel("Area Name")
    ax.set_ylabel("Average Traffic Volume")
    ax.set_title("Average Traffic Volume by Area")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Plot 2: Congestion Level distribution ──────────────────────────────────
    st.subheader("Congestion Level Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Congestion Level"], bins=30, kde=True, color="#ff7043", ax=ax)
    ax.set_xlabel("Congestion Level")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Plot 3: Day of week vs avg Traffic Volume ──────────────────────────────
    st.subheader("Average Traffic Volume by Day of Week")
    df["day_of_week"] = df["Date"].dt.dayofweek
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["day_name"] = df["day_of_week"].map(dow_map)
    dow_agg = df.groupby("day_of_week")["Traffic Volume"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    days = [dow_map[i] for i in range(7)]
    vals = [dow_agg.get(i, 0) for i in range(7)]
    ax.bar(days, vals, color=sns.color_palette("husl", 7))
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Traffic Volume")
    ax.set_title("Traffic Volume Pattern by Day of Week")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Traffic Forecaster
# ─────────────────────────────────────────────────────────────────────────────

def page_traffic_forecaster():
    """Render the Traffic Forecaster page."""
    st.title("🔮 Traffic Forecaster")
    st.markdown(
        "Select a road to generate a **3-day traffic forecast** using the trained LSTM model."
    )
    st.info(
        "⚠️ This model was trained on historical data. Forecasts are indicative only.",
        icon="ℹ️"
    )

    df = load_raw_data()
    if df is None:
        st.error("Dataset not found. Please place the CSV in `data/raw/`.")
        return

    model, feature_cols, num_features = load_lstm_model()
    if model is None:
        st.error(
            "LSTM model checkpoint not found at `artifacts/checkpoints/lstm_best.pt`. "
            "Please run `python src/training/train_lstm.py` first."
        )
        return

    scaler, weather_cols = load_scaler_and_weather()
    if scaler is None:
        st.error("Scaler not found. Please run training first.")
        return

    # ── Selectors ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Select Area", sorted(df["Area Name"].unique()), key="area_select")
    with col2:
        road_options = sorted(df[df["Area Name"] == area]["Road/Intersection Name"].unique())
        road = st.selectbox("Select Road/Intersection", road_options, key="road_select")

    horizon_choice = st.slider("Forecast horizon (days)", min_value=1, max_value=3, value=3, key="horizon_slider")

    if st.button("🚀 Forecast", key="forecast_btn", type="primary"):
        # ── Prepare last T=7 rows for selected road ───────────────────────────
        road_df = df[df["Road/Intersection Name"] == road].sort_values("Date")

        if len(road_df) < WINDOW_SIZE:
            st.warning(f"Not enough data for '{road}' (need ≥ {WINDOW_SIZE} rows, found {len(road_df)}).")
            return

        # Apply feature engineering
        from src.data.features import add_temporal_features, encode_categoricals, get_feature_columns
        road_df = add_temporal_features(road_df)
        road_df, _ = encode_categoricals(road_df, weather_columns=weather_cols, fit=False)

        # Align feature columns
        for col in feature_cols:
            if col not in road_df.columns:
                road_df[col] = 0.0

        last_window = road_df[feature_cols].tail(WINDOW_SIZE).values.astype("float32")
        last_window_scaled = scaler.transform(last_window)  # (7, F)

        x_tensor = torch.from_numpy(last_window_scaled).unsqueeze(0)  # (1, 7, F)

        with torch.no_grad():
            pred = model(x_tensor)  # (1, 3, 3)
        pred_np = pred.squeeze(0).numpy()[:horizon_choice]  # (H_chosen, 3)

        # ── Display forecast ──────────────────────────────────────────────────
        last_date = road_df["Date"].max()
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon_choice
        )

        forecast_df = pd.DataFrame(
            pred_np,
            columns=TARGET_NAMES,
            index=future_dates.strftime("%Y-%m-%d"),
        )

        st.success(f"✅ Forecast for **{road}** ({area})")
        st.markdown("*Values are in the model's scaled space — relative changes matter most.*")
        st.dataframe(forecast_df.round(4), use_container_width=True)

        # ── Line chart ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        colors = ["#4fc3f7", "#81c784", "#ffb74d"]

        for i, (ax, target, color) in enumerate(zip(axes, TARGET_NAMES, colors)):
            ax.plot(
                future_dates.strftime("%m-%d"),
                pred_np[:, i],
                marker="o", color=color, linewidth=2
            )
            ax.set_title(target, fontsize=10)
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"3-Day Forecast — {road}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Anomaly Explorer
# ─────────────────────────────────────────────────────────────────────────────

def page_anomaly_explorer():
    """Render the Anomaly Explorer page."""
    st.title("⚠️ Anomaly Explorer")
    st.markdown(
        "Detects anomalous traffic patterns using the **LSTM Autoencoder**. "
        "High reconstruction error = unusual traffic behaviour."
    )

    ae_model, feature_cols, num_features = load_autoencoder()
    if ae_model is None:
        st.error(
            "Autoencoder checkpoint not found. "
            "Please run `python src/training/train_autoencoder.py` first."
        )
        return

    if not THRESHOLD_PATH.exists():
        st.error("Anomaly threshold file not found. Run the autoencoder training first.")
        return

    with open(THRESHOLD_PATH) as f:
        threshold_data = json.load(f)
    threshold = float(threshold_data["threshold"])

    df = load_raw_data()
    if df is None:
        st.error("Dataset not found.")
        return

    scaler, weather_cols = load_scaler_and_weather()
    if scaler is None:
        st.error("Scaler not found.")
        return

    # ── Compute reconstruction errors on full dataset ─────────────────────────
    st.info("Computing reconstruction errors across all roads…")

    from src.data.features import add_temporal_features, encode_categoricals
    df_feat = add_temporal_features(df)
    df_feat, _ = encode_categoricals(df_feat, weather_columns=weather_cols, fit=False)

    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    df_feat[feature_cols] = scaler.transform(df_feat[feature_cols])

    from src.data.dataset import AnomalyDataset
    from torch.utils.data import DataLoader
    full_anomaly_ds     = AnomalyDataset(df_feat, feature_cols, WINDOW_SIZE)
    full_anomaly_loader = DataLoader(full_anomaly_ds, batch_size=64, shuffle=False)

    from src.anomaly.score import compute_reconstruction_errors
    errors = compute_reconstruction_errors(ae_model, full_anomaly_loader, torch.device("cpu"))

    # ── Build results DataFrame ───────────────────────────────────────────────
    meta_df = pd.DataFrame(full_anomaly_ds.metadata)
    meta_df["reconstruction_error"] = errors
    meta_df["is_anomalous"] = meta_df["reconstruction_error"] > threshold
    meta_df["date_str"] = pd.to_datetime(meta_df["date"]).dt.strftime("%Y-%m-%d")

    anomalous_df = meta_df[meta_df["is_anomalous"]].sort_values(
        "reconstruction_error", ascending=False
    ).reset_index(drop=True)

    st.metric("Total windows analysed",   f"{len(meta_df):,}")
    st.metric("Anomalous windows flagged", f"{len(anomalous_df):,}",
              delta=f"{100*len(anomalous_df)/len(meta_df):.1f}% of total")
    st.metric("Threshold (P95)",          f"{threshold:.4f}")

    st.markdown("---")

    # ── Top 20 anomalous roads ─────────────────────────────────────────────────
    st.subheader("Top 20 Anomalous Road / Date Combinations")
    top20 = anomalous_df.head(20)[["road", "date_str", "reconstruction_error", "incidents"]]
    top20.columns = ["Road", "Date", "Reconstruction Error", "Incident Reports"]

    def highlight_anomalies(row):
        return ["background-color: #3d0f0f; color: #ff6b6b"] * len(row)

    st.dataframe(top20.style.apply(highlight_anomalies, axis=1), use_container_width=True)

    # ── Reconstruction error by Area ───────────────────────────────────────────
    st.subheader("Average Reconstruction Error by Area")
    df_merged = meta_df.merge(
        df[["Road/Intersection Name", "Area Name"]].drop_duplicates(),
        left_on="road", right_on="Road/Intersection Name", how="left"
    )
    area_errors = df_merged.groupby("Area Name")["reconstruction_error"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if v > threshold else "#1f77b4" for v in area_errors.values]
    area_errors.plot(kind="bar", ax=ax, color=colors)
    ax.axhline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Area")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.set_title("Reconstruction Error by Area (red = above threshold on average)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def page_model_comparison():
    """Render the Model Comparison page."""
    st.title("📈 Model Comparison")
    st.markdown("Compare MAE, RMSE, MAPE, and R² across all forecasting models on the test set.")

    comparison_df = load_comparison_csv()
    if comparison_df is None:
        st.error(
            "Model comparison results not found at `artifacts/evaluation/model_comparison.csv`. "
            "Please run `python src/evaluation/evaluate_all.py` first."
        )
        return

    # ── Styled table ───────────────────────────────────────────────────────────
    st.subheader("Full Comparison Table")
    numeric_cols = comparison_df.select_dtypes(include="number").columns
    styled = comparison_df.style.background_gradient(
        subset=numeric_cols, cmap="RdYlGn_r"
    ).format({c: "{:.4f}" for c in numeric_cols})
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # ── MAE bar charts per target ──────────────────────────────────────────────
    st.subheader("MAE Comparison by Target Variable")

    mae_cols = [c for c in comparison_df.columns if c.endswith("_mae") and "model" not in c]
    models   = comparison_df["model"].tolist()

    if mae_cols:
        n_targets = len(mae_cols)
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5), sharey=False)
        if n_targets == 1:
            axes = [axes]

        palette = sns.color_palette("tab10", len(models))

        for ax, col in zip(axes, mae_cols):
            target_label = col.replace("_mae", "").replace("_", " ").title()
            vals = comparison_df[col].tolist()
            bars = ax.barh(models, vals, color=palette)
            ax.set_title(f"MAE — {target_label}", fontsize=10)
            ax.set_xlabel("MAE")
            # Annotate bars
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_width() + 0.01 * max(vals),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8
                )
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle("Model MAE Comparison per Target Variable", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown(
        "> **Note**: Lower MAE and RMSE are better. Higher R² is better. "
        "MAPE can be misleading for low-volume roads (see INTERVIEW_QA.md)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if PAGE == "📊 Dataset Overview":
    page_dataset_overview()
elif PAGE == "🔮 Traffic Forecaster":
    page_traffic_forecaster()
elif PAGE == "⚠️ Anomaly Explorer":
    page_anomaly_explorer()
elif PAGE == "📈 Model Comparison":
    page_model_comparison()
