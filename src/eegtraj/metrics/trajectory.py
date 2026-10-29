"""Trajectory decoding metrics and their aggregation (protocol decision D7).

Primary: Pearson r computed per trial and per axis, then mean over trials (per axis),
then mean over axes, giving one value per subject; across subjects report mean and SD.
Secondary: R^2 and NRMSE (RMSE / range of the true trial signal), aggregated the same way.
Pooled r (over concatenated trials) is inflated for stereotyped trajectories and is only
reported as an explicitly labeled secondary metric.

Undefined values (a constant true or predicted signal for r; a constant true signal for
R^2 and NRMSE) are NaN. Aggregation skips NaNs and reports how many were skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AXES = ("x", "y", "z")
TRIAL_METRICS = ("r", "r2", "rmse", "nrmse")
CONSTANT_TOL = 1e-12  # variance below this (relative to scale) counts as a constant signal

METRIC_DEFINITIONS = {
    "r": "Pearson correlation per trial per axis; mean over trials, then over axes, then subjects",
    "r2": "1 - SS_res / SS_tot per trial per axis (SS_tot around the trial mean); same aggregation",
    "rmse": "root mean squared error per trial per axis (target units); same aggregation",
    "nrmse": "RMSE / (max - min of the true signal) per trial per axis; same aggregation",
    "pooled_r": "SECONDARY, labeled: Pearson r over all samples of all trials concatenated, per axis, "
                "then mean over axes; inflated by trial-to-trial stereotypy",
}


def _is_constant(x: np.ndarray) -> bool:
    return bool(np.var(x) <= CONSTANT_TOL * max(1.0, float(np.mean(x * x))))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r of two 1-D arrays; NaN if either is constant or shorter than 2 samples."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2 or _is_constant(y_true) or _is_constant(y_pred):
        return float("nan")
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    return float(np.clip((a @ b) / np.sqrt((a @ a) * (b @ b)), -1.0, 1.0))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; NaN if ``y_true`` is constant."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if _is_constant(y_true):
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64)) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE divided by the range of ``y_true``; NaN if ``y_true`` is constant."""
    y_true = np.asarray(y_true, dtype=np.float64)
    span = float(y_true.max() - y_true.min())
    return float("nan") if _is_constant(y_true) or span == 0 else rmse(y_true, y_pred) / span


def per_trial_metrics(y_true: np.ndarray, y_pred: np.ndarray, trial_ids: np.ndarray) -> pd.DataFrame:
    """Metrics for every (trial, axis).

    Args:
        y_true, y_pred: float arrays (n_samples, n_axes), same units.
        trial_ids: (n_samples,) trial identifier per sample (any hashable values).

    Returns:
        DataFrame with columns trial, axis, n, r, r2, rmse, nrmse.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    trial_ids = np.asarray(trial_ids)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or len(trial_ids) != len(y_true):
        raise ValueError("y_true and y_pred must be (n_samples, n_axes) and match trial_ids")
    rows = []
    axes = AXES if y_true.shape[1] == len(AXES) else tuple(str(i) for i in range(y_true.shape[1]))
    for trial in pd.unique(trial_ids):
        m = trial_ids == trial
        for a, axis in enumerate(axes):
            t, p = y_true[m, a], y_pred[m, a]
            rows.append({"trial": trial, "axis": axis, "n": int(m.sum()), "r": pearson_r(t, p),
                         "r2": r2_score(t, p), "rmse": rmse(t, p), "nrmse": nrmse(t, p)})
    return pd.DataFrame(rows)


def pooled_r(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """SECONDARY metric: Pearson r over all concatenated samples, per axis and mean over axes."""
    out = {axis: pearson_r(y_true[:, a], y_pred[:, a]) for a, axis in enumerate(AXES[:y_true.shape[1]])}
    out["mean"] = float(np.nanmean(list(out.values())))
    return out


def aggregate_subject(trial_df: pd.DataFrame) -> dict:
    """Aggregate one subject's per-trial metrics: mean over trials per axis, then mean over axes.

    Returns a dict ``{metric: {"per_axis": {axis: value}, "mean": value, "n_nan": count}}``.
    """
    out = {}
    for metric in TRIAL_METRICS:
        per_axis = trial_df.groupby("axis", sort=False)[metric].mean()  # pandas mean skips NaN
        out[metric] = {
            "per_axis": {a: float(v) for a, v in per_axis.items()},
            "mean": float(np.nanmean(per_axis.to_numpy())),
            "n_nan": int(trial_df[metric].isna().sum()),
        }
    return out


def aggregate_across_subjects(subject_results: dict[str, dict]) -> dict:
    """Mean and SD (ddof=1) across subjects of each subject-level metric (overall and per axis)."""
    out = {}
    for metric in TRIAL_METRICS:
        vals = np.array([r[metric]["mean"] for r in subject_results.values()])
        axes = list(next(iter(subject_results.values()))[metric]["per_axis"])
        per_axis = {a: np.array([r[metric]["per_axis"][a] for r in subject_results.values()]) for a in axes}
        out[metric] = {
            "mean": float(np.nanmean(vals)),
            "sd": float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else float("nan"),
            "n_subjects": int(np.sum(~np.isnan(vals))),
            "per_axis_mean": {a: float(np.nanmean(v)) for a, v in per_axis.items()},
            "per_axis_sd": {a: float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else float("nan")
                            for a, v in per_axis.items()},
        }
    return out
