"""Tests for the baseline interface, the template baseline, and runner helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eegtraj.baselines import REGISTRY, FitContext, make_baseline
from eegtraj.baselines.runner import trial_filter
from eegtraj.baselines.template import time_since_onset
from eegtraj.data.windows import Standardizer, SubjectSamples, segment_samples
from eegtraj.metrics import aggregate_subject, per_trial_metrics

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


def with_led(s: SubjectSamples, led: list[float]) -> SubjectSamples:
    s.trials = s.trials.assign(ev_LEDOn=led)
    return s


def scored_r(model, s: SubjectSamples, ctx: FitContext) -> float:
    pred = model.predict(s, np.ones(len(s.sample_idx), bool), ctx)
    return aggregate_subject(per_trial_metrics(s.target[:, s.sample_idx].T, pred, s.trial_row))["r"]["mean"]


def test_cue_template_equals_oracle_template_when_reaction_time_is_constant():
    rt = 0.3
    onsets = [[2.0, 2.4, 2.9], [2.2, 2.6], [2.13, 2.71]]
    subjects = [with_led(synthetic_subject(n, on), [t - rt for t in on]) for n, on in zip(("P1", "P2", "P4"), onsets)]
    train, test = subjects[:2], subjects[2]
    ctx = ctx_for(subjects)
    oracle, cue = make_baseline("template"), make_baseline("template_cue")
    oracle.fit(train, [], ctx)
    cue.fit(train, [], ctx)
    assert scored_r(cue, test, ctx) == pytest.approx(scored_r(oracle, test, ctx), abs=1e-3)
    info = cue.info()
    assert info["oracle_timing"] is False and info["cue_timing"] is True
    assert info["reaction_time"]["P4"]["median_s"] == pytest.approx(rt)
    assert info["reaction_time"]["P4"]["iqr_s"] == pytest.approx(0)


def test_cue_template_prediction_ignores_the_test_onset():
    train = [with_led(synthetic_subject("P1", [2.0, 2.4, 2.9]), [1.7, 2.1, 2.6])]
    test = with_led(synthetic_subject("P4", [2.13, 2.71]), [1.8, 2.4])
    ctx = ctx_for(train + [test])
    model = make_baseline("template_cue")
    model.fit(train, [], ctx)
    mask = np.ones(len(test.sample_idx), bool)
    first = model.predict(test, mask, ctx)
    test.trials = test.trials.assign(onset_s=test.trials.onset_s + 0.5)
    assert np.array_equal(model.predict(test, mask, ctx), first)


def test_cue_template_is_below_oracle_when_reaction_time_varies():
    train = [with_led(synthetic_subject("P1", [2.2, 2.4, 2.9, 2.5]), [2.0] * 4)]
    test = with_led(synthetic_subject("P4", [2.1, 2.95]), [2.0] * 2)
    ctx = ctx_for(train + [test])
    oracle, cue = make_baseline("template"), make_baseline("template_cue")
    oracle.fit(train, [], ctx)
    cue.fit(train, [], ctx)
    assert scored_r(cue, test, ctx) < scored_r(oracle, test, ctx)


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


def observed_subject(name: str, mix: np.ndarray, noise: float, seed: int) -> SubjectSamples:
    """Subject whose 'EEG' is a noisy linear mix of the true position and velocity (lag 0)."""
    rng = np.random.default_rng(seed)
    onsets = list(2.0 + 0.6 * rng.random(6))
    s = synthetic_subject(name, onsets)
    pos = s.target.astype(np.float64)
    vel = np.gradient(pos, 1 / FS, axis=1)
    state = np.vstack([pos, vel / 10.0])
    s.eeg = (mix @ state + noise * rng.standard_normal((mix.shape[0], pos.shape[1]))).astype(np.float32)
    return s


def test_kalman_decodes_linear_gaussian_observations_causally():
    from eegtraj.data.windows import fit_eeg_normalizers, fit_target_normalizer

    mix = np.random.default_rng(10).standard_normal((8, 6))
    train = [observed_subject("P1", mix, 0.5, 1), observed_subject("P2", mix, 0.5, 2)]
    val = [observed_subject("P4", mix, 0.5, 3)]
    test = observed_subject("P5", mix, 0.5, 4)
    subs = train + val + [test]
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subs}
    ctx = FitContext(eeg_norm=fit_eeg_normalizers(subs, ("P1", "P2"), "train_subjects"),
                     target_norm=fit_target_normalizer(train, ("P1", "P2")), history=2, lag_step=2,
                     seed=0, train_masks=ones, val_masks=ones)
    model = make_baseline("kalman", {"n_components": [2, 6]})
    model.fit(train, val, ctx)
    assert model.k == 6  # all 6 informative directions beat 2 on validation
    pred = model.predict(test, ones["P5"], ctx)
    true = test.target[:, test.sample_idx].T
    assert min(np.corrcoef(pred[:, a], true[:, a])[0, 1] for a in range(3)) > 0.95
    # Causality: changing EEG after a sample must not change the estimate at that sample.
    cut = int(test.sample_idx[100])
    test.eeg[:, cut + 1:] += 50.0
    pred2 = model.predict(test, ones["P5"], ctx)
    assert np.allclose(pred2[:101], pred[:101])


def test_kalman_prior_only_ignores_eeg():
    from eegtraj.data.windows import fit_eeg_normalizers, fit_target_normalizer

    mix = np.random.default_rng(11).standard_normal((8, 6))
    train = [observed_subject("P1", mix, 0.5, 1), observed_subject("P2", mix, 0.5, 2)]
    test = observed_subject("P5", mix, 0.5, 4)
    subs = train + [test]
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subs}
    ctx = FitContext(eeg_norm=fit_eeg_normalizers(subs, ("P1", "P2"), "train_subjects"),
                     target_norm=fit_target_normalizer(train, ("P1", "P2")), history=2, lag_step=2,
                     seed=0, train_masks=ones, val_masks=ones)
    model = make_baseline("kalman", {"n_components": [4], "prior_only": True})
    model.fit(train, [], ctx)
    a = model.predict(test, ones["P5"], ctx)
    test.eeg[:] = np.random.default_rng(0).standard_normal(test.eeg.shape)
    assert np.allclose(model.predict(test, ones["P5"], ctx), a)
