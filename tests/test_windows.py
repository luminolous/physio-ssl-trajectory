"""Tests for eegtraj.data.windows: causal alignment and train-only normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegtraj.data.splits import Fold
from eegtraj.data.windows import (
    Standardizer,
    SubjectSamples,
    WindowDataset,
    fit_eeg_normalizers,
    fit_target_normalizer,
    lag_samples,
    lagged_features,
    recentered_wrist,
    sample_masks,
    segment_samples,
    window_at,
)

FS = 100.0
HISTORY = 50


def make_trials(subject: str, n_lifts: int = 3, n: int = 600) -> pd.DataFrame:
    rows = []
    for i in range(n_lifts):
        rows.append(dict(subject=subject, series=i + 1, lift=1, n_samples=n, offset_start=i * n,
                         offset_end=(i + 1) * n, onset_s=2.0, segment_start_s=1.5, segment_end_s=5.0))
    return pd.DataFrame(rows)


def make_subject(subject: str, scale: float, rng: np.random.Generator) -> SubjectSamples:
    trials = make_trials(subject)
    T = int(trials.offset_end.iloc[-1])
    idx, rows = segment_samples(trials, FS, HISTORY)
    return SubjectSamples(subject, rng.standard_normal((4, T)).astype(np.float32) * scale,
                          (rng.standard_normal((3, T)) * scale + scale).astype(np.float32), idx, rows, trials, FS)


def test_segment_samples_cover_segment_and_check_history():
    trials = make_trials("P1")
    idx, rows = segment_samples(trials, FS, HISTORY)
    t = (idx - trials.offset_start.to_numpy()[rows] + 1) / FS
    assert t.min() >= 1.5 and t.max() <= 5.0
    assert np.bincount(rows).tolist() == [351, 351, 351]  # samples at 1.50 ... 5.00 s
    bad = trials.assign(segment_start_s=0.2)  # history would start before the window
    with pytest.raises(ValueError):
        segment_samples(bad, FS, HISTORY)


@pytest.mark.parametrize("lag", [0, 7, 30])
def test_known_lag_recovered_by_lag_features(lag):
    # EEG leads the target by `lag` samples: eeg[t] = y[t + lag]  ->  y[t] = eeg[t - lag].
    rng = np.random.default_rng(0)
    T = 3000
    y = np.convolve(rng.standard_normal(T + lag), np.ones(5) / 5, mode="same")
    eeg = y[lag:lag + T][None, :].astype(np.float32)
    target = y[:T]
    idx = np.arange(HISTORY, T)
    lags = np.arange(HISTORY)
    x = lagged_features(eeg, idx, lags)
    r = [np.corrcoef(x[:, l], target[idx])[0, 1] for l in lags]
    assert int(np.argmax(r)) == lag and r[lag] > 0.9999


def test_window_is_causal_and_matches_lag_features():
    rng = np.random.default_rng(1)
    eeg = rng.standard_normal((3, 200)).astype(np.float32)
    i = 120
    w = window_at(eeg, i, HISTORY)
    assert w.shape == (3, HISTORY) and np.array_equal(w[:, -1], eeg[:, i])  # ends at the target sample
    x = lagged_features(eeg, np.array([i]), np.arange(HISTORY))[0].reshape(3, HISTORY)
    assert np.array_equal(x[:, ::-1], w)  # lag l == window column HISTORY-1-l
    with pytest.raises(ValueError):
        lagged_features(eeg, np.array([i]), np.array([-1]))


def test_lag_samples_stay_inside_window():
    lags = lag_samples(HISTORY, 2)
    assert lags[0] == 0 and lags[-1] == 48 and len(lags) == 25


def test_target_normalizer_uses_train_subjects_only():
    rng = np.random.default_rng(2)
    subs = [make_subject("P1", 1.0, rng), make_subject("P2", 1.0, rng), make_subject("P9", 1000.0, rng)]
    a = fit_target_normalizer(subs, train_subjects=("P1", "P2"))
    subs[2].target[:] = -5e6  # changing the held-out subject must not change the statistics
    b = fit_target_normalizer(subs, train_subjects=("P1", "P2"))
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.std, b.std)
    pooled = np.concatenate([s.target[:, s.sample_idx] for s in subs[:2]], axis=1)
    assert np.allclose(a.mean[:, 0], pooled.mean(axis=1), atol=1e-5)
    # Only scored samples count: changing a non-segment sample of a training subject changes nothing.
    outside = np.setdiff1d(np.arange(subs[0].target.shape[1]), subs[0].sample_idx)[0]
    subs[0].target[:, outside] = 1e9
    c = fit_target_normalizer(subs, train_subjects=("P1", "P2"))
    assert np.array_equal(a.mean, c.mean)


def test_target_normalizer_respects_within_subject_masks():
    rng = np.random.default_rng(3)
    s = make_subject("P1", 1.0, rng)
    fold = Fold("within_P1", ("P1",), ("P1",), ("P1",), train_series=(1,), val_series=(2,), test_series=(3,))
    masks = sample_masks(fold, [s])
    a = fit_target_normalizer([s], ("P1",), train_masks=masks["train"])
    s.target[:, s.sample_idx[masks["test"]["P1"]]] = 1e9  # test series of the same subject
    b = fit_target_normalizer([s], ("P1",), train_masks=masks["train"])
    assert np.array_equal(a.mean, b.mean)


def test_eeg_normalization_modes():
    rng = np.random.default_rng(4)
    subs = [make_subject("P1", 1.0, rng), make_subject("P2", 3.0, rng), make_subject("P9", 50.0, rng)]
    per = fit_eeg_normalizers(subs, ("P1", "P2"), "per_subject")
    assert np.allclose(per["P9"].transform(subs[2].eeg).std(axis=1), 1.0, atol=1e-3)  # transductive, own stats
    strict = fit_eeg_normalizers(subs, ("P1", "P2"), "train_subjects")
    expected = Standardizer.fit([subs[0].eeg, subs[1].eeg])
    assert np.array_equal(strict["P9"].mean, expected.mean) and np.array_equal(strict["P9"].std, expected.std)


def test_window_dataset_items():
    rng = np.random.default_rng(5)
    subs = [make_subject("P1", 1.0, rng)]
    eeg_n = fit_eeg_normalizers(subs, ("P1",), "per_subject")
    tgt_n = fit_target_normalizer(subs, ("P1",))
    ds = WindowDataset(subs, eeg_n, tgt_n, HISTORY)
    assert len(ds) == len(subs[0].sample_idx)
    x, y = ds[0]
    i = int(subs[0].sample_idx[0])
    assert tuple(x.shape) == (4, HISTORY) and tuple(y.shape) == (3,)
    assert np.allclose(x.numpy()[:, -1], eeg_n["P1"].transform(subs[0].eeg)[:, i])
    assert np.allclose(tgt_n.inverse(y.numpy()[:, None])[:, 0], subs[0].target[:, i], atol=1e-4)


def test_recentered_wrist_zero_at_onset():
    trials = make_trials("P1", n_lifts=2, n=400)
    T = 800
    kin = np.zeros((4, 3, T), np.float32)
    t = np.tile((np.arange(400) + 1) / FS, 2)
    kin[3] = np.stack([10 * t, 5 + t, np.full(T, 7.0)])
    rc = recentered_wrist(kin, trials, FS)
    j = int(round(2.0 * FS)) - 1  # sample at t = 2.00 s
    for start in (0, 400):
        assert np.allclose(rc[:, start + j], 0.0, atol=1e-5)
