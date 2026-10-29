"""Tests for the baseline interface, the template baseline, and runner helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegtraj.baselines import REGISTRY, FitContext, make_baseline
from eegtraj.baselines.runner import trial_filter
from eegtraj.baselines.template import time_since_onset
from eegtraj.data.windows import Standardizer, SubjectSamples, segment_samples

FS = 100.0
N = 700


def shape(tau: np.ndarray) -> np.ndarray:
    """Stereotyped trajectory as a function of time since onset (3, n)."""
    s = np.clip(tau, 0, None)
    return np.stack([10 * (1 - np.exp(-3 * s)), -20 * (1 - np.exp(-2 * s)), 5 * np.sin(np.clip(s, 0, 3))])


def synthetic_subject(name: str, onsets: list[float]) -> SubjectSamples:
    rows, targets = [], []
    for i, on in enumerate(onsets):
        rows.append(dict(subject=name, series=1, lift=i + 1, n_samples=N, offset_start=i * N, offset_end=(i + 1) * N,
                         onset_s=on, segment_start_s=on - 0.5, segment_end_s=on + 4.0,
                         onset_mismatch=(i == 0), filter_transient_lowfreq=False))
        t = (np.arange(N) + 1) / FS
        targets.append(shape(t - on))
    trials = pd.DataFrame(rows)
    idx, trow = segment_samples(trials, FS, 50)
    y = np.concatenate(targets, axis=1).astype(np.float32)
    return SubjectSamples(name, np.zeros((2, y.shape[1]), np.float32), y, idx, trow, trials, FS)


def ctx_for(subjects: list[SubjectSamples]) -> FitContext:
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subjects}
    unit = Standardizer(np.zeros((3, 1), np.float32), np.ones((3, 1), np.float32))
    return FitContext(eeg_norm={}, target_norm=unit, history=50, lag_step=2, seed=0, train_masks=ones, val_masks=ones)


def test_registry_contains_template():
    assert "template" in REGISTRY
    with pytest.raises(KeyError):
        make_baseline("does-not-exist")


def test_time_since_onset():
    s = synthetic_subject("P1", [2.0, 2.37])
    tau = time_since_onset(s)
    assert tau.min() == pytest.approx(-0.5, abs=0.01) and tau.max() == pytest.approx(4.0, abs=0.01)


def test_template_predicts_stereotyped_trials_with_different_onsets():
    train = [synthetic_subject("P1", [2.0, 2.4, 2.9]), synthetic_subject("P2", [2.2, 2.6])]
    test = synthetic_subject("P4", [2.13, 2.71])
    ctx = ctx_for(train + [test])
    model = make_baseline("template")
    model.fit(train, [], ctx)
    pred = model.predict(test, np.ones(len(test.sample_idx), bool), ctx)
    true = test.target[:, test.sample_idx].T
    assert pred.shape == true.shape
    # Errors come only from binning time-since-onset to 10 ms.
    assert np.abs(pred - true).max() < 0.35
    assert model.info()["oracle_timing"] is True


def test_trial_filter_excludes_flags_and_limits():
    s = synthetic_subject("P1", [2.0, 2.4, 2.9])
    kept = trial_filter(s, ["onset_mismatch"], None)
    assert set(kept.trial_row) == {1, 2}
    kept = trial_filter(s, [], 2)
    assert set(kept.trial_row) == {0, 1}
