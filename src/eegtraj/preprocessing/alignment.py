"""Euclidean alignment (He and Wu, 2020): whiten each subject's EEG by its mean spatial covariance.

The covariance is the mean over the subject's lifts of each lift's covariance (lift samples only, each lift
demeaned). It uses no labels, so for the test subject it is transductive, like the per-subject normalization (D9).
After CAR the covariance has rank C - 1, so the inverse square root is taken on its non-null subspace; the
aligned mean covariance is then the projector onto that subspace instead of the identity.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from eegtraj.data.windows import SubjectSamples


def mean_trial_covariance(eeg: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Mean over trials of the (C, C) covariance of ``eeg[:, start:end]``, each trial demeaned."""
    covs = []
    for a, b in zip(starts, ends):
        x = np.asarray(eeg[:, a:b], dtype=np.float64)
        x = x - x.mean(axis=1, keepdims=True)
        covs.append(x @ x.T / x.shape[1])
    return np.mean(covs, axis=0)


def inverse_sqrt(cov: np.ndarray, rel_tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Symmetric inverse square root on the eigenvectors with eigenvalue > rel_tol * largest. Returns it and the rank."""
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > rel_tol * vals.max()
    return (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T, int(keep.sum())


def lift_bounds(s: SubjectSamples) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of each lift's own samples in ``s.eeg`` (pre-lift EEG excluded)."""
    return s.trials.offset_start.to_numpy(), s.trials.offset_end.to_numpy()


def euclidean_align(s: SubjectSamples) -> tuple[SubjectSamples, dict]:
    """Aligned copy of ``s`` (every EEG sample, pre-lift EEG included) and facts for the log."""
    starts, ends = lift_bounds(s)
    cov = mean_trial_covariance(s.eeg, starts, ends)
    w, rank = inverse_sqrt(cov)
    eeg = (w @ s.eeg.astype(np.float64)).astype(np.float32)
    kept = np.linalg.eigvalsh(cov)[-rank:]
    return replace(s, eeg=eeg), {"rank": rank, "n_channels": int(cov.shape[0]), "n_trials": int(len(starts)),
                                 "condition_number_on_subspace": float(kept.max() / kept.min())}
