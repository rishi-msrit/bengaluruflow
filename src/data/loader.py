from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "Date",
    "Area Name",
    "Road/Intersection Name",
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Environmental Impact",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Weather Conditions",
    "Roadwork and Construction Activity",
]

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "Banglore_traffic_Dataset.csv"
)


def load_data(path=None):
    if path is None:
        path = DEFAULT_DATA_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download 'Banglore_traffic_Dataset.csv' from Kaggle and place it at the path above."
        )

    df = pd.read_csv(path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nFound: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Road/Intersection Name"]).reset_index(drop=True)

    print(f"[loader] Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df


def check_no_nulls(df):
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, f"Null values found:\n{cols_with_nulls}"
    print("[loader] No null values. ✓")
