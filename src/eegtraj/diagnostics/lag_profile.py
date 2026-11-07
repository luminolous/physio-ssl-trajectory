"""Single-lag ridge decoding: how well EEG at one time offset from the kinematic sample predicts it.

Analysis only, outside the causal protocol. A positive lag L uses EEG L samples before the target sample;
a negative lag uses EEG after it. These models must never be reported as decoders.

Each subject contributes sufficient statistics (X^T X, X^T y, sums) per lag, so a fold's ridge is a sum over its
training subjects and one solve per alpha. Alpha is chosen on the inner validation subjects by the per-trial
Pearson r (mean over trials and axes, then subjects), with the same relative grid as ``ridge_lag``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from eegtraj.data.windows import SubjectSamples


@dataclass
class LagStats:
    """Sufficient statistics of one subject at one lag (features = channels at that lag)."""

    xtx: np.ndarray  # (p, p)
    xty: np.ndarray  # (p, 3)
    sx: np.ndarray  # (p,)
    sy: np.ndarray  # (3,)
    n: int


def keep_lag_range(s: SubjectSamples, n_pre: int, min_lag: int, max_lag: int) -> tuple[SubjectSamples, int]:
    """Drop scored samples whose EEG at some lag in [min_lag, max_lag] falls outside their lift.

    Every lag then uses the same samples. Returns the restricted subject and the number of dropped samples.
    """
    start = s.trials.offset_start.to_numpy()[s.trial_row] - n_pre
    end = s.trials.offset_end.to_numpy()[s.trial_row]
    keep = (s.sample_idx - max_lag >= start) & (s.sample_idx - min_lag < end)
    return replace(s, sample_idx=s.sample_idx[keep], trial_row=s.trial_row[keep]), int((~keep).sum())


def check_lag_range(s: SubjectSamples, n_pre: int, min_lag: int, max_lag: int) -> None:
    """Raise if any scored sample's EEG at a lag in [min_lag, max_lag] falls outside its own lift (plus pre-lift EEG)."""
    start = s.trials.offset_start.to_numpy()[s.trial_row] - n_pre
    end = s.trials.offset_end.to_numpy()[s.trial_row]
    if (s.sample_idx - max_lag < start).any() or (s.sample_idx - min_lag >= end).any():
        raise ValueError(f"{s.subject}: lags [{min_lag}, {max_lag}] reach outside a lift")


def lag_features(z: np.ndarray, idx: np.ndarray, lag: int) -> np.ndarray:
    """(n, C) features: normalized EEG ``z`` (C, T) at ``idx - lag``."""
    return z[:, idx - lag].T.astype(np.float64)


def subject_stats(z: np.ndarray, idx: np.ndarray, y: np.ndarray, lag: int) -> LagStats:
    """Statistics of one subject. y: (n, 3) targets at ``idx``."""
    x = lag_features(z, idx, lag)
    y = y.astype(np.float64)
    return LagStats(x.T @ x, x.T @ y, x.sum(0), y.sum(0), len(x))


def fit_ridge(stats: list[LagStats], alphas: np.ndarray) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Weights and intercept per alpha. Alphas are relative to the mean diagonal of the centered X^T X."""
    n = sum(s.n for s in stats)
    xtx, xty = sum(s.xtx for s in stats), sum(s.xty for s in stats)
    mx, my = sum(s.sx for s in stats) / n, sum(s.sy for s in stats) / n
    xtx_c = xtx - n * np.outer(mx, mx)
    xty_c = xty - n * np.outer(mx, my)
    p = len(mx)
    scale = np.trace(xtx_c) / p
    out = {}
    for a in alphas:
        w = np.linalg.solve(xtx_c + a * scale * np.eye(p), xty_c)
        out[float(a)] = (w, my - mx @ w)
    return out


def mean_trial_r(y_true: np.ndarray, y_pred: np.ndarray, rows: np.ndarray) -> float:
    """Per-trial Pearson r per axis, mean over trials then axes (vectorized; rows must be grouped by trial)."""
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    counts = np.diff(np.r_[starts, len(rows)])

    def tmean(v):
        return np.add.reduceat(v, starts, axis=0) / counts[:, None]

    t = y_true - np.repeat(tmean(y_true), counts, axis=0)
    p = y_pred - np.repeat(tmean(y_pred), counts, axis=0)
    num = np.add.reduceat(t * p, starts, axis=0)
    den = np.sqrt(np.add.reduceat(t * t, starts, axis=0) * np.add.reduceat(p * p, starts, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return float(np.nanmean(np.nanmean(r, axis=0)))
