import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true.flatten(), y_pred.flatten()))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())))


def mape(y_true, y_pred):
    """MAPE — misleading when true values are near zero, excluded from calc."""
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def r2(y_true, y_pred):
    return float(r2_score(y_true.flatten(), y_pred.flatten()))


def compute_all_metrics(y_true, y_pred, target_names, horizon):
    results = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, :, i].flatten()
        yp = y_pred[:, :, i].flatten()
        safe = name.replace(" ", "_").lower()
        results[f"{safe}_mae"] = mae(yt, yp)
        results[f"{safe}_rmse"] = rmse(yt, yp)
        results[f"{safe}_mape"] = mape(yt, yp)
        results[f"{safe}_r2"] = r2(yt, yp)
    return results


def anomaly_roc_auc(errors, labels):
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, errors))
