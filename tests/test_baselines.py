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


def test_stereotypy_uses_other_trials_and_is_seeded():
    test = synthetic_subject("P4", [2.0, 2.3, 2.6, 2.9])
    # Give each trial a distinct offset so the donor identity is visible in the prediction.
    for row in range(4):
        sl = slice(test.trials.offset_start.iat[row], test.trials.offset_end.iat[row])
        test.target[:, sl] += 100.0 * row
    ctx = ctx_for([test])
    model = make_baseline("stereotypy", {"n_permutations": 3})
    assert model.n_repeats == 3
    model.fit([], [], ctx)
    mask = np.ones(len(test.sample_idx), bool)
    p0 = model.predict(test, mask, ctx, repeat=0)
    assert np.array_equal(p0, model.predict(test, mask, ctx, repeat=0))  # deterministic per repeat
    for row in range(4):
        sel = test.trial_row == row
        donor_offset = np.round((p0[sel, 0] - test.target[0, test.sample_idx[sel]] + 100.0 * row) / 100.0)
        assert len(set(donor_offset)) == 1 and donor_offset[0] != row  # one donor, never the trial itself
    draws = {tuple(np.round(model.predict(test, mask, ctx, repeat=k)[::200, 0], 3)) for k in range(3)}
    assert len(draws) > 1  # repeats draw different donors


def test_stereotypy_identical_trials_give_exact_prediction():
    test = synthetic_subject("P5", [2.0, 2.5, 3.0])
    ctx = ctx_for([test])
    model = make_baseline("stereotypy", {"n_permutations": 1})
    model.fit([], [], ctx)
    pred = model.predict(test, np.ones(len(test.sample_idx), bool), ctx)
    assert np.abs(pred - test.target[:, test.sample_idx].T).max() < 1e-3  # same shape, aligned on onset


def linear_subject(name: str, w_true: np.ndarray, noise: float, seed: int) -> SubjectSamples:
    """Subject whose target is a known causal lagged linear map of its EEG plus noise."""
    s = synthetic_subject(name, [2.0, 2.3, 2.6, 2.9])
    rng = np.random.default_rng(seed)
    T = s.target.shape[1]
    eeg = rng.standard_normal((2, T)).astype(np.float32)
    # y_a(t) = sum_c sum_l w[c, l, a] * eeg[c, t - 2 l]  (lags 0, 2, 4 samples)
    y = np.zeros((3, T), np.float32)
    for c in range(2):
        for li, lag in enumerate((0, 2, 4)):
            shifted = np.concatenate([np.zeros(lag, np.float32), eeg[c, :T - lag]])
            y += w_true[c, li][:, None] * shifted[None, :]
    y += noise * rng.standard_normal(y.shape).astype(np.float32)
    s.eeg, s.target = eeg, y
    return s


def test_ridge_recovers_known_lagged_map_and_picks_small_alpha():
    from eegtraj.data.windows import fit_eeg_normalizers, fit_target_normalizer

    w_true = np.random.default_rng(0).standard_normal((2, 3, 3))
    train = [linear_subject("P1", w_true, 0.05, 1), linear_subject("P2", w_true, 0.05, 2)]
    val = [linear_subject("P4", w_true, 0.05, 3)]
    test = linear_subject("P5", w_true, 0.05, 4)
    subs = train + val + [test]
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subs}
    ctx = FitContext(eeg_norm=fit_eeg_normalizers(subs, ("P1", "P2"), "train_subjects"),
                     target_norm=fit_target_normalizer(train, ("P1", "P2")), history=6, lag_step=2,
                     seed=0, train_masks=ones, val_masks=ones)
    model = make_baseline("ridge_lag", {"alphas": [1e-6, 1.0, 1e3]})
    model.fit(train, val, ctx)
    assert model.alpha == 1e-6  # nearly noise-free linear data: weakest regularization wins on val
    assert model.info()["n_features"] == 2 * 3 and model.info()["lags_samples"] == [0, 2, 4]
    pred = model.predict(test, ones["P5"], ctx)
    true = test.target[:, test.sample_idx].T
    assert min(np.corrcoef(pred[:, a], true[:, a])[0, 1] for a in range(3)) > 0.99
