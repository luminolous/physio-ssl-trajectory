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
MIN_SHARED_BINS = 50  # 0.5 s at 100 Hz; fewer shared time bins leave a trial pair undefined

METRIC_DEFINITIONS = {
    "r": "Pearson correlation per trial per axis; mean over trials, then over axes, then subjects",
    "r2": "1 - SS_res / SS_tot per trial per axis (SS_tot around the trial mean); same aggregation",
    "rmse": "root mean squared error per trial per axis (target units); same aggregation",
    "nrmse": "RMSE / (max - min of the true signal) per trial per axis; same aggregation",
    "pooled_r": "SECONDARY, labeled: Pearson r over all samples of all trials concatenated, per axis, "
                "then mean over axes; inflated by trial-to-trial stereotypy",
    "residual_r": "Pearson r per trial per axis between (prediction - T_cue) and (target - T_cue), T_cue being the "
                  "cue-locked template of the training subjects; same aggregation; NaN if the prediction equals T_cue",
    "residual_r_oracle": "ANALYSIS ONLY, labeled: residual_r with the template aligned on the test trial's own "
                         "kinematic onset, which a real decoder cannot know",
    "residual_r_other_trial": "CONTROL: residual_r with each trial's predicted residual paired with the target "
                              "residual of another trial of the same subject, on their shared bins of time since "
                              "LED-on; measures what is shared across trials",
    "residual_r_specific": "residual_r - residual_r_other_trial per subject; the part that needs the same trial",
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


def residual_r(y_true: np.ndarray, y_pred: np.ndarray, template: np.ndarray, trial_ids: np.ndarray) -> dict:
    """Pearson r between (y_pred - template) and (y_true - template), per trial and axis, then aggregated as in D7.

    A prediction equal to the template has no residual to correlate, so those values are NaN. Aggregation
    skips them and counts them in ``n_nan``.

    Returns ``{"per_axis": {axis: value}, "mean": value, "n_nan": count}``.
    """
    true_res = np.asarray(y_true, np.float64) - np.asarray(template, np.float64)
    pred_res = np.asarray(y_pred, np.float64) - np.asarray(template, np.float64)
    trial_ids = np.asarray(trial_ids)
    trials = pd.unique(trial_ids)
    values = np.array([[pearson_r(true_res[trial_ids == t, a], pred_res[trial_ids == t, a]) for a in range(true_res.shape[1])]
                       for t in trials])
    axes = AXES[:true_res.shape[1]]
    per_axis = pd.DataFrame(values, columns=axes).mean(axis=0)  # pandas mean skips NaN
    return {"per_axis": {a: float(v) for a, v in per_axis.items()}, "mean": float(per_axis.mean()),
            "n_nan": int(np.isnan(values).sum())}


def residual_r_other_trial(y_true: np.ndarray, y_pred: np.ndarray, template: np.ndarray, trial_ids: np.ndarray,
                           time_bins: np.ndarray, rng: np.random.Generator, n_permutations: int = 5,
                           min_shared_bins: int = MIN_SHARED_BINS) -> dict:
    """Control for ``residual_r``: pair each trial's predicted residual with another trial's target residual.

    Samples are matched on ``time_bins`` (integer time since a common event), and r uses the bins both
    trials have. Each permutation pairs every trial with the next one in a random cycle, so no trial
    meets itself. Pairs with fewer than ``min_shared_bins`` shared bins, or a constant residual, are NaN.

    Returns ``{"per_axis": {axis: value}, "mean": value, "n_nan": count}``; ``n_nan`` is over all permutations.
    """
    true_res = np.asarray(y_true, np.float64) - np.asarray(template, np.float64)
    pred_res = np.asarray(y_pred, np.float64) - np.asarray(template, np.float64)
    trial_ids, time_bins = np.asarray(trial_ids), np.asarray(time_bins)
    masks = [trial_ids == t for t in pd.unique(trial_ids)]
    n_axes, n_trials = true_res.shape[1], len(masks)
    values = np.full((n_permutations, n_trials, n_axes), np.nan)
    for k in range(n_permutations if n_trials > 1 else 0):
        order = rng.permutation(n_trials)
        for a, b, slot in zip(order, np.roll(order, -1), range(n_trials)):
            _, ia, ib = np.intersect1d(time_bins[masks[a]], time_bins[masks[b]], assume_unique=True, return_indices=True)
            if ia.size >= min_shared_bins:
                p, t = pred_res[masks[a]][ia], true_res[masks[b]][ib]
                values[k, slot] = [pearson_r(t[:, c], p[:, c]) for c in range(n_axes)]
    per_axis = pd.DataFrame(values.reshape(-1, n_axes), columns=AXES[:n_axes]).mean(axis=0)
    return {"per_axis": {a: float(v) for a, v in per_axis.items()}, "mean": float(per_axis.mean()),
            "n_nan": int(np.isnan(values).sum())}


def residual_r_matched(y_true: np.ndarray, y_pred: np.ndarray, template: np.ndarray, trial_ids: np.ndarray,
                       time_bins: np.ndarray, rng: np.random.Generator, n_permutations: int = 5,
                       min_shared_bins: int = MIN_SHARED_BINS) -> dict:
    """DIAGNOSTIC: same-trial and other-trial residual correlation on the SAME bins.

    ``residual_r`` uses a trial's whole window, ``residual_r_other_trial`` only the bins two trials share, so their
    difference also contains an effect of the window. Here both terms use the shared bins of a pair, and every pair
    enters in both directions, so any prediction that is one function of ``time_bins`` scores exactly 0 (``matched``).
    Trial pairs as in ``residual_r_other_trial`` (same ``rng`` state, same pairs).

    Returns ``own`` (same trial, shared bins), ``cross`` (other trial, shared bins), ``matched`` = own - cross,
    ``shared_fraction`` (shared bins / bins of the trial), and ``n_pairs`` (directed pairs used). Values are means over
    directed pairs per axis, then over axes; a pair with a constant residual on an axis is skipped on that axis for
    both terms and both directions.
    """
    true_res = np.asarray(y_true, np.float64) - np.asarray(template, np.float64)
    pred_res = np.asarray(y_pred, np.float64) - np.asarray(template, np.float64)
    trial_ids, time_bins = np.asarray(trial_ids), np.asarray(time_bins)
    index = [np.flatnonzero(trial_ids == t) for t in pd.unique(trial_ids)]
    own, cross, frac = [], [], []
    for _ in range(n_permutations if len(index) > 1 else 0):
        order = rng.permutation(len(index))
        for a, b in zip(order, np.roll(order, -1)):
            _, ia, ib = np.intersect1d(time_bins[index[a]], time_bins[index[b]], assume_unique=True, return_indices=True)
            if ia.size < min_shared_bins:
                continue
            ra, rb = index[a][ia], index[b][ib]
            frac += [ia.size / index[a].size, ib.size / index[b].size]
            pa, pb, ta, tb = (_standardize(x) for x in (pred_res[ra], pred_res[rb], true_res[ra], true_res[rb]))
            own += [(pa * ta).sum(axis=0), (pb * tb).sum(axis=0)]
            cross += [(pa * tb).sum(axis=0), (pb * ta).sum(axis=0)]
    if not own:
        return {"own": float("nan"), "cross": float("nan"), "matched": float("nan"), "shared_fraction": float("nan"),
                "n_pairs": 0}
    o, c = np.array(own), np.array(cross)
    bad = np.isnan(o + c).reshape(-1, 2, o.shape[1]).any(axis=1).repeat(2, axis=0)
    o[bad], c[bad] = np.nan, np.nan
    mo, mc = pd.DataFrame(o).mean(axis=0).mean(), pd.DataFrame(c).mean(axis=0).mean()  # pandas mean skips NaN
    return {"own": float(mo), "cross": float(mc), "matched": float(mo - mc), "shared_fraction": float(np.mean(frac)),
            "n_pairs": len(own)}


def _standardize(x: np.ndarray) -> np.ndarray:
    """Columns centered and scaled to unit norm; a constant column becomes NaN."""
    constant = np.var(x, axis=0) <= CONSTANT_TOL * np.maximum(1.0, np.mean(x * x, axis=0))
    z = x - x.mean(axis=0)
    norm = np.sqrt((z * z).sum(axis=0))
    norm[constant] = np.nan
    return z / norm


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
