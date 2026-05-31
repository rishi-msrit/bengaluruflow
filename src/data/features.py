from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.joblib"
WEATHER_ENCODER_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "weather_ohe_columns.json"
AREA_ENCODER_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "area_label_encoder.joblib"
ROAD_ENCODER_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "road_label_encoder.joblib"

# Karnataka public holidays 2022-2024
KARNATAKA_HOLIDAYS = {
    "2022-01-26", "2022-02-16", "2022-03-18", "2022-04-14",
    "2022-04-15", "2022-05-03", "2022-07-10", "2022-08-09",
    "2022-08-15", "2022-10-02", "2022-10-05", "2022-10-24",
    "2022-10-26", "2022-11-01", "2022-11-08", "2022-12-25",
    "2023-01-14", "2023-01-26", "2023-03-07", "2023-03-22",
    "2023-04-04", "2023-04-07", "2023-04-14", "2023-04-22",
    "2023-05-05", "2023-06-29", "2023-07-28", "2023-08-15",
    "2023-09-07", "2023-09-19", "2023-09-28", "2023-10-02",
    "2023-10-24", "2023-11-01", "2023-11-27", "2023-12-25",
    "2024-01-26", "2024-02-19", "2024-03-08", "2024-03-25",
    "2024-03-29", "2024-04-14", "2024-04-17", "2024-04-21",
    "2024-05-23", "2024-06-17", "2024-07-17", "2024-08-15",
    "2024-09-05", "2024-09-16", "2024-10-02", "2024-10-12",
    "2024-10-31", "2024-11-01", "2024-11-15", "2024-12-25",
}
HOLIDAY_SET = {pd.Timestamp(d) for d in KARNATAKA_HOLIDAYS}

TARGET_COLUMNS = ["Traffic Volume", "Congestion Level", "Average Speed"]

EXPECTED_NEW_TEMPORAL_COLS = [
    "day_of_week", "day_sin", "day_cos",
    "month", "month_sin", "month_cos",
    "is_weekend", "is_holiday", "days_since_start",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def add_temporal_features(df):
    df = df.copy()

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month"] = df["Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["Date"].isin(HOLIDAY_SET).astype(int)

    min_date = df["Date"].min()
    df["days_since_start"] = (df["Date"] - min_date).dt.days

    return df


def encode_categoricals(df, weather_columns=None, fit=True):
    df = df.copy()

    # binary encode roadwork
    df["roadwork_active"] = (
        df["Roadwork and Construction Activity"]
        .str.strip()
        .str.lower()
        .apply(lambda x: 1 if x not in ("none", "no", "0", "false", "") else 0)
    )

    # one-hot weather
    weather_dummies = pd.get_dummies(df["Weather Conditions"], prefix="weather", dtype=int)

    if fit:
        weather_ohe_cols = list(weather_dummies.columns)
    else:
        assert weather_columns is not None
        weather_ohe_cols = weather_columns
        for col in weather_ohe_cols:
            if col not in weather_dummies.columns:
                weather_dummies[col] = 0
        weather_dummies = weather_dummies[weather_ohe_cols]

    df = pd.concat([df, weather_dummies], axis=1)

    # label encode for grouping only
    area_le = LabelEncoder()
    road_le = LabelEncoder()
    df["area_label"] = area_le.fit_transform(df["Area Name"])
    df["road_label"] = road_le.fit_transform(df["Road/Intersection Name"])

    AREA_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(area_le, AREA_ENCODER_PATH)
    joblib.dump(road_le, ROAD_ENCODER_PATH)

    return df, weather_ohe_cols


def get_feature_columns(weather_ohe_cols):
    numeric_cols = [
        "Traffic Volume", "Average Speed", "Travel Time Index",
        "Congestion Level", "Road Capacity Utilization", "Incident Reports",
        "Environmental Impact", "Public Transport Usage",
        "Traffic Signal Compliance", "Parking Usage",
        "Pedestrian and Cyclist Count",
        "day_of_week", "day_sin", "day_cos",
        "month", "month_sin", "month_cos",
        "is_weekend", "is_holiday", "days_since_start",
        "roadwork_active",
    ]
    return numeric_cols + weather_ohe_cols


def chronological_split(df):
    unique_dates = sorted(df["Date"].unique())
    n = len(unique_dates)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_cutoff = unique_dates[train_end - 1]
    val_cutoff = unique_dates[val_end - 1]

    train_df = df[df["Date"] <= train_cutoff].copy()
    val_df = df[(df["Date"] > train_cutoff) & (df["Date"] <= val_cutoff)].copy()
    test_df = df[df["Date"] > val_cutoff].copy()

    print(f"[features] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


def fit_and_apply_scaler(train_df, val_df, test_df, feature_cols):
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[features] Scaler saved -> {SCALER_PATH}")

    return train_df, val_df, test_df, scaler


def build_features(df):
    df = add_temporal_features(df)

    train_tmp, val_tmp, test_tmp = chronological_split(df)

    train_enc, weather_ohe_cols = encode_categoricals(train_tmp, fit=True)
    val_enc, _ = encode_categoricals(val_tmp, weather_columns=weather_ohe_cols, fit=False)
    test_enc, _ = encode_categoricals(test_tmp, weather_columns=weather_ohe_cols, fit=False)

    WEATHER_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEATHER_ENCODER_PATH, "w") as f:
        json.dump(weather_ohe_cols, f)

    feature_cols = get_feature_columns(weather_ohe_cols)

    for split_df in [train_enc, val_enc, test_enc]:
        for col in feature_cols:
            if col not in split_df.columns:
                split_df[col] = 0

    train_df, val_df, test_df, scaler = fit_and_apply_scaler(
        train_enc, val_enc, test_enc, feature_cols
    )

    print(f"[features] {len(feature_cols)} features total")
    return train_df, val_df, test_df, feature_cols, scaler, weather_ohe_cols
