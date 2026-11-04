"""Tests for the Kalman diagnostics. Uses synthetic data only."""

import numpy as np

from eegtraj.baselines import make_baseline
from eegtraj.baselines.base import FitContext
from eegtraj.data.windows import fit_eeg_normalizers, fit_target_normalizer
from eegtraj.diagnostics.kalman import lag_autocorr, r2_decomposition, rmse_by_time, run_filter
from eegtraj.metrics import r2_score
from tests.test_baselines import observed_subject


def test_r2_decomposition_matches_r2_score():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.standard_normal((300, 3)), axis=0)
    p = 2.0 * t + rng.standard_normal((300, 3)) + 5.0
    rows = np.repeat([0, 1, 2], 100)
    d = r2_decomposition(t, p, rows)
    for trial, axis_i in [(0, 0), (1, 1), (2, 2)]:
        m = rows == trial
        got = d[(d.trial == trial) & (d.axis == "xyz"[axis_i])].r2.item()
        assert np.isclose(got, r2_score(t[m, axis_i], p[m, axis_i]))


def test_r2_decomposition_constant_prediction_has_only_offset_term():
    t = np.sin(np.linspace(0, 6, 200))[:, None].repeat(3, axis=1)
    d = r2_decomposition(t, np.ones_like(t), np.zeros(200, int))
    assert np.allclose(d.a, 0) and np.allclose(d.term_corr, 0)
    assert np.allclose(d.r2, -d.b ** 2)


def test_rmse_by_time_bins_known_error():
    t = np.zeros((300, 3))
    p = np.zeros((300, 3))
    p[100:200] = 3.0  # second bin of the first trial only
    starts, e = rmse_by_time(t, p, np.zeros(300, int), bin_s=1.0, max_s=4.0, fs_hz=100.0)
    assert starts.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(e[:3], [0.0, 3.0, 0.0]) and np.isnan(e[3])


def test_lag_autocorr_white_vs_ar():
    rng = np.random.default_rng(1)
    white = rng.standard_normal((50_000, 2))
    ar = np.empty_like(white)
    ar[0] = white[0]
    for i in range(1, len(ar)):
        ar[i] = 0.9 * ar[i - 1] + white[i]
    rows = np.zeros(len(white), int)
    assert abs(lag_autocorr(white, rows, [1])[1]) < 0.02
    assert abs(lag_autocorr(ar, rows, [1])[1] - 0.9) < 0.02


def test_lag_autocorr_ignores_pairs_across_trials():
    e = np.r_[np.ones(10), -np.ones(10)][:, None]
    assert np.isclose(lag_autocorr(e, np.repeat([0, 1], 10), [1])[1], 18 / 20)


def test_run_filter_reproduces_decoder():
    mix = np.random.default_rng(10).standard_normal((8, 6))
    train = [observed_subject("P1", mix, 0.5, 1), observed_subject("P2", mix, 0.5, 2)]
    test = observed_subject("P5", mix, 0.5, 4)
    subs = train + [test]
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subs}
    ctx = FitContext(eeg_norm=fit_eeg_normalizers(subs, ("P1", "P2"), "train_subjects"),
                     target_norm=fit_target_normalizer(train, ("P1", "P2")), history=2, lag_step=2,
                     seed=0, train_masks=ones, val_masks=ones)
    model = make_baseline("kalman", {"n_components": [6]})
    model.fit(train, [], ctx)
    trace = run_filter(model, test, ones["P5"], ctx, 6)
    assert np.allclose(trace.pred, model.predict(test, ones["P5"], ctx), atol=1e-3)
    prior = run_filter(model, test, ones["P5"], ctx, 6, prior_only=True)
    assert np.allclose(prior.gain_norm, 0.0)
    # Larger R must give smaller gains.
    assert run_filter(model, test, ones["P5"], ctx, 6, r_scale=100.0).gain_norm[-1] < trace.gain_norm[-1]
