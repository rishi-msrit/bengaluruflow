from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WINDOW_SIZE = 7
HORIZON = 3
TARGET_NAMES = ["Traffic Volume", "Congestion Level", "Average Speed"]


class TrafficDataset(Dataset):
    """Sliding window dataset for multi-step traffic forecasting."""

    def __init__(self, df, feature_cols, window_size=WINDOW_SIZE, horizon=HORIZON):
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target_cols = TARGET_NAMES
        self.samples = []
        self._build_samples(df)

    def _build_samples(self, df):
        road_groups = df.groupby("Road/Intersection Name", sort=False)

        for _road_name, group in road_groups:
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < self.window_size + self.horizon:
                continue

            features = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_cols].values.astype(np.float32)

            total_steps = self.window_size + self.horizon
            for i in range(len(group) - total_steps + 1):
                x = features[i : i + self.window_size]
                y = targets[i + self.window_size : i + self.window_size + self.horizon]
                self.samples.append((x, y))

        print(f"[dataset] {len(self.samples):,} windows built")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class AnomalyDataset(Dataset):
    """Input-only dataset for the LSTM autoencoder (reconstruction objective)."""

    def __init__(self, df, feature_cols, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.samples = []
        self.metadata = []
        self._build_samples(df)

    def _build_samples(self, df):
        road_groups = df.groupby("Road/Intersection Name", sort=False)

        for road_name, group in road_groups:
            group = group.sort_values("Date").reset_index(drop=True)

            if len(group) < self.window_size:
                continue

            features = group[self.feature_cols].values.astype(np.float32)

            for i in range(len(group) - self.window_size + 1):
                x = features[i : i + self.window_size]
                self.samples.append(x)

                last_date = group["Date"].iloc[i + self.window_size - 1]
                incident_count = group["Incident Reports"].iloc[i + self.window_size - 1]
                self.metadata.append({
                    "road": road_name,
                    "date": last_date,
                    "incidents": incident_count,
                })

        print(f"[dataset] Anomaly dataset: {len(self.samples):,} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])
