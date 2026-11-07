"""Tests for the single-lag ridge of the lag profile (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegtraj.data.windows import SubjectSamples
from eegtraj.diagnostics.lag_profile import check_lag_range, fit_ridge, keep_lag_range, lag_features, mean_trial_r, subject_stats
from eegtraj.metrics import aggregate_subject, per_trial_metrics


def test_mean_trial_r_matches_the_reference_metric():
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(5), 40)
    y = rng.standard_normal((200, 3))
    p = y + rng.standard_normal((200, 3))
    assert mean_trial_r(y, p, rows) == pytest.approx(aggregate_subject(per_trial_metrics(y, p, rows))["r"]["mean"])


def test_lag_features_point_at_past_and_future_samples():
    z = np.arange(20, dtype=np.float32)[None, :].repeat(2, 0)
    idx = np.array([10, 11])
    assert lag_features(z, idx, 3)[:, 0].tolist() == [7, 8]  # EEG before the target sample
    assert lag_features(z, idx, -2)[:, 0].tolist() == [12, 13]  # EEG after it


def test_ridge_recovers_the_lag_that_carries_the_target():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 5000)).astype(np.float32)
    idx = np.arange(200, 4800)
    y = np.stack([z[1, idx - 15], -z[2, idx - 15], z[1, idx - 15] + z[2, idx - 15]], axis=1)  # EEG 15 samples earlier
    scores = {}
    for lag in (-15, 0, 15, 30):
        (w, b), = fit_ridge([subject_stats(z, idx, y, lag)], np.array([1e-6])).values()
        scores[lag] = mean_trial_r(y, lag_features(z, idx, lag) @ w + b, np.zeros(len(idx), int))
    assert scores[15] == pytest.approx(1, abs=1e-6)
    assert max(abs(scores[l]) for l in (-15, 0, 30)) < 0.1


def test_lag_range_check_stays_inside_each_lift():
    trials = pd.DataFrame({"offset_start": [10, 60], "offset_end": [50, 100]})
    s = SubjectSamples("P1", np.zeros((1, 100)), np.zeros((3, 100)), np.array([15, 45, 65, 90]), np.array([0, 0, 1, 1]), trials, 100.0)
    check_lag_range(s, n_pre=10, min_lag=-4, max_lag=15)
    with pytest.raises(ValueError):
        check_lag_range(s, n_pre=10, min_lag=-5, max_lag=15)  # 45 + 5 reaches the end of lift 0
    with pytest.raises(ValueError):
        check_lag_range(s, n_pre=10, min_lag=-4, max_lag=16)  # 15 - 16 is before lift 0 and its pre-lift EEG
    kept, dropped = keep_lag_range(s, n_pre=10, min_lag=-5, max_lag=15)  # drops only sample 45
    assert kept.sample_idx.tolist() == [15, 65, 90] and kept.trial_row.tolist() == [0, 1, 1] and dropped == 1
