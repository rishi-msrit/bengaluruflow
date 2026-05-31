import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import joblib
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data.loader import load_data, check_no_nulls
from src.data.features import build_features
from src.data.dataset import TrafficDataset, WINDOW_SIZE, HORIZON, TARGET_NAMES

RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
XGB_LR = 0.1
XGB_MAX_DEPTH = 6
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "artifacts" / "baselines"


def dataset_to_numpy(dataset):
    X_list, y_list = [], []
    for x, y in dataset.samples:
        X_list.append(x.flatten())
        y_list.append(y.flatten())
    return np.array(X_list), np.array(y_list)


def compute_metrics(y_true, y_pred):
    metrics = {}
    y_true_r = y_true.reshape(-1, HORIZON, len(TARGET_NAMES))
    y_pred_r = y_pred.reshape(-1, HORIZON, len(TARGET_NAMES))

    for i, name in enumerate(TARGET_NAMES):
        yt = y_true_r[:, :, i].flatten()
        yp = y_pred_r[:, :, i].flatten()
        safe = name.replace(" ", "_").lower()
        metrics[f"{safe}_mae"] = mean_absolute_error(yt, yp)
        metrics[f"{safe}_rmse"] = np.sqrt(mean_squared_error(yt, yp))
        metrics[f"{safe}_r2"] = r2_score(yt, yp)
        mask = yt != 0
        metrics[f"{safe}_mape"] = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else float("nan")

    return metrics


def train_and_log(run_name, model, X_train, y_train, X_test, y_test, params):
    print(f"\n[baselines] Training {run_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    save_name = run_name.lower().replace(" ", "_")
    model_path = BASELINE_DIR / f"{save_name}.joblib"
    joblib.dump(model, model_path)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics, model_path


def main():
    df = load_data()
    check_no_nulls(df)
    train_df, val_df, test_df, feature_cols, scaler, _ = build_features(df)

    X_train, y_train = dataset_to_numpy(TrafficDataset(train_df, feature_cols, WINDOW_SIZE, HORIZON))
    X_test, y_test = dataset_to_numpy(TrafficDataset(test_df, feature_cols, WINDOW_SIZE, HORIZON))

    print(f"[baselines] X_train: {X_train.shape} | X_test: {X_test.shape}")

    mlflow.set_experiment("baseline_models")

    with mlflow.start_run(run_name="linear_regression"):
        mlflow.log_param("model", "LinearRegression")
        metrics, path = train_and_log(
            "linear_regression",
            MultiOutputRegressor(LinearRegression()),
            X_train, y_train, X_test, y_test, {}
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(str(path))

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_param("n_estimators", RF_N_ESTIMATORS)
        metrics, path = train_and_log(
            "random_forest",
            MultiOutputRegressor(RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)),
            X_train, y_train, X_test, y_test, {}
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(str(path))

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({"n_estimators": XGB_N_ESTIMATORS, "learning_rate": XGB_LR})
        metrics, path = train_and_log(
            "xgboost",
            MultiOutputRegressor(XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS, learning_rate=XGB_LR,
                max_depth=XGB_MAX_DEPTH, random_state=RANDOM_STATE, verbosity=0
            )),
            X_train, y_train, X_test, y_test, {}
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(str(path))

    print("\n[baselines] Done.")


if __name__ == "__main__":
    main()
