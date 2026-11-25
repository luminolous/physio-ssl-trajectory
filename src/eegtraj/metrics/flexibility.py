"""Flexibility metrics (design.md "Experiment 3"; definitions decided 2026-11-20): does a decoder follow trial-to-trial
variation, or emit a rigid mean trajectory?

Inputs per subject: ``y_true`` and ``y_pred`` (N, 3) in target units (cm; the target is the wrist position re-centered at
the lift's kinematic onset, D1), ``trial_ids`` (N,) and ``t`` (N,) seconds since LED-on, on the scored window
[kinematic onset - 0.5 s, tHandStop]. Samples of one trial must be contiguous and in time order.

Computed on the scored window:
- Prediction-spread ratio: per time bin (since LED-on) and axis, SD of predictions across trials / SD of targets
  across trials; mean over bins with enough trials, then over axes. 0 = rigid, 1 = as variable as the targets.
- DTW shape error: RMSE after dynamic time warping (Euclidean 3-D cost) per trial, mean over trials; with the plain
  RMSE on the same trials, the difference is the error explained by timing alone.
- Residual share: energy of the additive correction Delta relative to the variance of the base component (warp
  template, or template_cue for the residual head) around its mean.

Trial features (``trial_features``), all times in s since LED-on, never relative to the window start: onset (first
sample whose displacement |y| exceeds 10 % of the trial's maximum), duration (onset to the time of peak height), peak
height (max of the vertical axis), reach distance (max |y|), peak speed (max 3-D speed). Target features come from the
true trajectory on the scored window. Prediction features come from the prediction on the whole stream, a fixed span
relative to the cue (LED-on - 1.3 s, the first patch end, to the end of the lift window; saved by the runner as
``full_*`` arrays), because the scored window's edges come from the true kinematics and would leak timing into the
prediction's features. A feature whose landmark falls on the first or last sample of its span is undefined (NaN) for
that trial; ``edge_counts`` reports how many trials were excluded per feature.
- Trial-feature correlation: Pearson r across trials between predicted and true value of each feature.
- Condition sensitivity: per weight condition, the mean feature of predictions and targets; the heaviest - lightest
  difference and whether its sign agrees.

Condition labels (``weight_g``) are read only for scoring, never for training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERTICAL_AXIS = 2  # z is vertical in the WAY-EEG-GAL position frame
DISPLACEMENT_FRACTION = 0.10  # onset: first sample with |y| above this fraction of the trial maximum
FEATURES = ("peak_height", "reach_distance", "duration", "peak_speed", "onset_time")
MIN_TRIALS_PER_BIN = 10
EPS = 1e-9


def _trial_slices(trial_ids: np.ndarray) -> list[tuple[int, slice]]:
    change = np.flatnonzero(np.diff(trial_ids)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, trial_ids.size]
    ids = trial_ids[starts]
    if np.unique(ids).size != ids.size:
        raise ValueError("samples of a trial are not contiguous")
    return [(int(i), slice(a, b)) for i, a, b in zip(ids, starts, ends)]


def spread_ratio(y_true: np.ndarray, y_pred: np.ndarray, t: np.ndarray, fs: float = 100.0,
                 min_trials: int = MIN_TRIALS_PER_BIN) -> dict:
    """Prediction-spread ratio per axis and its mean over axes. Bins: time since LED-on rounded to 1 / fs."""
    b = np.round(np.asarray(t) * fs).astype(np.int64)
    b -= b.min()
    cnt = np.bincount(b)
    keep = cnt >= min_trials

    def sd(x: np.ndarray) -> np.ndarray:
        m = np.bincount(b, weights=x, minlength=cnt.size) / np.maximum(cnt, 1)
        v = np.bincount(b, weights=(x - m[b]) ** 2, minlength=cnt.size) / np.maximum(cnt - 1, 1)
        return np.sqrt(v)

    out = {}
    for a, name in enumerate(("x", "y", "z")):
        st, sp = sd(y_true[:, a]), sd(y_pred[:, a])
        ok = keep & (st > EPS)
        out[name] = float(np.mean(sp[ok] / st[ok])) if ok.any() else np.nan
    out["mean"] = float(np.nanmean([out[k] for k in ("x", "y", "z")]))
    out["n_bins"] = int(keep.sum())
    return out


def trial_features(y: np.ndarray, t: np.ndarray, fs: float = 100.0,
                   fraction: float = DISPLACEMENT_FRACTION) -> tuple[dict, dict]:
    """Features of one trial's trajectory (S, 3) in the target frame, and per feature whether its landmark lies on
    the first or last sample of the span (then the value is not trusted). Times in s since LED-on (``t``)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    def edge(i: int) -> bool:
        return i == 0 or i == n - 1

    d = np.linalg.norm(y, axis=1)
    i_on = int(np.flatnonzero(d > fraction * d.max())[0]) if d.max() > EPS else 0
    i_h = int(np.argmax(y[:, VERTICAL_AXIS]))
    i_r = int(np.argmax(d))
    speed = np.linalg.norm(np.gradient(y, 1.0 / fs, axis=0), axis=1) if n > 1 else np.zeros(n)
    i_v = int(np.argmax(speed))
    feats = {"peak_height": float(y[i_h, VERTICAL_AXIS]), "reach_distance": float(d[i_r]),
             "duration": float(t[i_h] - t[i_on]), "peak_speed": float(speed[i_v]), "onset_time": float(t[i_on])}
    edges = {"peak_height": edge(i_h), "reach_distance": edge(i_r), "duration": i_on == 0 or edge(i_h),
             "peak_speed": edge(i_v), "onset_time": i_on == 0}
    return feats, edges


def feature_table(y_true: np.ndarray, t_true: np.ndarray, ids_true: np.ndarray, y_pred: np.ndarray,
                  t_pred: np.ndarray, ids_pred: np.ndarray, fs: float = 100.0) -> pd.DataFrame:
    """One row per trial present in both: ``true_<f>``, ``pred_<f>`` (NaN when the landmark is on an edge of its
    span) and ``edge_true_<f>``, ``edge_pred_<f>``. Target and prediction may use different spans (``t_*``)."""
    pred = dict(_trial_slices(ids_pred))
    rows = []
    for i, sl in _trial_slices(ids_true):
        if i not in pred:
            continue
        ft, et = trial_features(y_true[sl], t_true[sl], fs)
        fp, ep = trial_features(y_pred[pred[i]], t_pred[pred[i]], fs)
        rec = {"trial": i}
        for k in FEATURES:
            rec.update({f"true_{k}": np.nan if et[k] else ft[k], f"pred_{k}": np.nan if ep[k] else fp[k],
                        f"edge_true_{k}": et[k], f"edge_pred_{k}": ep[k]})
        rows.append(rec)
    return pd.DataFrame(rows)


def edge_counts(features: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per feature: trials excluded because the landmark is on an edge (prediction, target) and trials used."""
    return {k: {"pred": int(features[f"edge_pred_{k}"].sum()), "true": int(features[f"edge_true_{k}"].sum()),
                "used": int(features[[f"true_{k}", f"pred_{k}"]].notna().all(axis=1).sum())} for k in FEATURES}


def feature_correlation(features: pd.DataFrame) -> dict[str, float]:
    """Pearson r across trials between predicted and true value of each feature (NaN if either is constant)."""
    out = {}
    for k in FEATURES:
        d = features[[f"true_{k}", f"pred_{k}"]].dropna()
        a, b = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
        out[k] = float(np.corrcoef(a, b)[0, 1]) if len(d) > 2 and a.std() > EPS and b.std() > EPS else np.nan
    return out


def dtw_path(a: np.ndarray, b: np.ndarray, band: int | None = None) -> np.ndarray:
    """DTW alignment of (n, d) and (m, d) under Euclidean cost; returns the path as (L, 2) index pairs.
    ``band``: Sakoe-Chiba half-width in samples (None = unconstrained). Anti-diagonal vectorised dynamic programme."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    if band is not None:
        ii, jj = np.indices((n, m))
        cost[np.abs(ii - jj * (n / m)) > max(band, abs(n - m))] = np.inf
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for k in range(2, n + m + 1):  # cells (i, j) with i + j = k, 1-based
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        prev = np.minimum(np.minimum(acc[i - 1, j], acc[i, j - 1]), acc[i - 1, j - 1])
        acc[i, j] = cost[i - 1, j - 1] + prev
    path = [(n - 1, m - 1)]
    i, j = n, m
    while (i, j) != (1, 1):
        steps = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        i, j = min(steps, key=lambda s: acc[s] if s[0] >= 1 and s[1] >= 1 else np.inf)
        path.append((i - 1, j - 1))
    return np.array(path[::-1])


def dtw_rmse(y_true: np.ndarray, y_pred: np.ndarray, band: int | None = None) -> float:
    """RMSE (3-D Euclidean, per sample) along the DTW path of one trial."""
    p = dtw_path(y_true, y_pred, band)
    d = np.linalg.norm(y_true[p[:, 0]] - y_pred[p[:, 1]], axis=1)
    return float(np.sqrt(np.mean(d ** 2)))


def shape_error(y_true: np.ndarray, y_pred: np.ndarray, trial_ids: np.ndarray, band: int | None = None,
                downsample: int = 1) -> dict:
    """Mean over trials of the DTW RMSE and of the plain 3-D RMSE; ``downsample`` keeps every k-th sample (speed)."""
    dtw, plain = [], []
    for _, sl in _trial_slices(trial_ids):
        a, b = y_true[sl][::downsample], y_pred[sl][::downsample]
        dtw.append(dtw_rmse(a, b, None if band is None else max(1, band // downsample)))
        plain.append(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))))
    return {"dtw_rmse": float(np.mean(dtw)), "rmse": float(np.mean(plain)),
            "timing_share": float(1.0 - np.mean(dtw) / np.mean(plain)) if np.mean(plain) > EPS else np.nan}


def condition_sensitivity(features: pd.DataFrame, condition: pd.Series,
                          features_used: tuple[str, ...] = ("duration", "peak_height")) -> pd.DataFrame:
    """Per feature: mean of predictions and targets per condition level (``condition`` indexed like ``features.trial``),
    the highest - lowest level difference for both, and whether the signs agree. Edge trials (NaN) are skipped."""
    cond = condition.reindex(features.trial.to_numpy()).to_numpy()
    levels = np.sort(pd.unique(cond[~pd.isna(cond)]))
    rows = []
    for k in features_used:
        rec = {"feature": k, "low": levels[0], "high": levels[-1]}
        for src in ("true", "pred"):
            means = {lv: float(np.nanmean(features[f"{src}_{k}"].to_numpy()[cond == lv])) for lv in levels}
            rec.update({f"{src}_mean_{lv}": v for lv, v in means.items()})
            rec[f"{src}_diff"] = means[levels[-1]] - means[levels[0]]
        tol = 1e-6 * abs(rec["true_diff"]) + EPS  # a rigid predictor's difference is rounding noise
        rec["same_sign"] = bool(np.sign(rec["true_diff"]) == np.sign(rec["pred_diff"]) and abs(rec["pred_diff"]) > tol)
        rec["ratio"] = rec["pred_diff"] / rec["true_diff"] if abs(rec["true_diff"]) > EPS else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def residual_share(delta: np.ndarray, y_pred: np.ndarray) -> float:
    """sum ||Delta||^2 / sum ||base - mean(base)||^2 with base = y_pred - Delta (warp or template component)."""
    base = y_pred - delta
    var = np.sum((base - base.mean(axis=0)) ** 2)
    return float(np.sum(delta ** 2) / var) if var > EPS else np.nan


def subject_flexibility(y_true: np.ndarray, y_pred: np.ndarray, trial_ids: np.ndarray, t: np.ndarray,
                        condition: pd.Series | None = None, delta: np.ndarray | None = None,
                        y_pred_full: np.ndarray | None = None, t_full: np.ndarray | None = None,
                        ids_full: np.ndarray | None = None, fs: float = 100.0, dtw_band: int | None = None,
                        dtw_downsample: int = 1) -> dict:
    """All flexibility metrics of one subject. Prediction features use the whole-stream prediction (``*_full``) when
    given; otherwise the scored window (``feature_source`` = window: timing features then biased by the window)."""
    full = y_pred_full is not None
    feats = (feature_table(y_true, t, trial_ids, y_pred_full, t_full, ids_full, fs) if full
             else feature_table(y_true, t, trial_ids, y_pred, t, trial_ids, fs))
    out = {"spread_ratio": spread_ratio(y_true, y_pred, t, fs), "feature_r": feature_correlation(feats),
           "feature_edges": edge_counts(feats), "feature_source": "full_stream" if full else "window",
           "shape": shape_error(y_true, y_pred, trial_ids, dtw_band, dtw_downsample), "n_trials": int(len(feats))}
    if condition is not None:
        out["condition"] = condition_sensitivity(feats, condition).to_dict(orient="records")
    if delta is not None:
        out["residual_share"] = residual_share(delta, y_pred)
    return out
