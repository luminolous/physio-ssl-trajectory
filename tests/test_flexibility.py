"""Flexibility metrics on synthetic trials (no data needed)."""

import numpy as np
import pandas as pd
import pytest
import torch

from eegtraj.data.lift_streams import LiftStream, full_stream_lift
from eegtraj.metrics.flexibility import (condition_sensitivity, dtw_path, dtw_rmse, edge_counts, feature_correlation,
                                         feature_table, residual_share, shape_error, spread_ratio, subject_flexibility,
                                         trial_features)
from eegtraj.models.decoder_runner import predict
from tests.test_decoder import decoder

FS = 100.0


def lift(t: np.ndarray, onset: float, dur: float, h: float, d: float) -> np.ndarray:
    """Out-and-back reach with a lift: zero before ``onset``, back at the start after ``onset + dur``."""
    s = np.clip((t - onset) / dur, 0, 1)
    out = np.sin(np.pi * s)
    return np.stack([10 * d * out, -20 * d * out, h * out ** 2], axis=1)


def synthetic(n_trials: int = 30, seed: int = 0):
    """Lifts of varying onset, duration, height and reach; time since LED-on from 0 to 3 s for every trial."""
    rng = np.random.default_rng(seed)
    t1 = np.arange(0, 3.0, 1 / FS)
    ys, ts, ids, cond = [], [], [], {}
    for i in range(n_trials):
        onset, dur = 0.4 + 0.3 * rng.random(), 1.0 + 0.6 * rng.random()
        h, d = 4 + 4 * rng.random(), 0.8 + 0.4 * rng.random()  # height (cm), reach scale
        ys.append(lift(t1, onset, dur, h, d))
        ts.append(t1)
        ids.append(np.full(t1.size, i))
        cond[i] = 165 if h < 6 else 660
    return np.concatenate(ys), np.concatenate(ts), np.concatenate(ids), pd.Series(cond)


def rigid(y, t, ids):
    """Each sample replaced by the mean over trials at the same time point."""
    b = np.round(t * FS).astype(int)
    m = np.stack([np.bincount(b, weights=y[:, a]) / np.bincount(b) for a in range(3)], axis=1)
    return m[b]


def test_spread_ratio_rigid_zero_identity_one():
    y, t, ids, _ = synthetic()
    assert spread_ratio(y, rigid(y, t, ids), t)["mean"] == pytest.approx(0.0, abs=1e-9)
    assert spread_ratio(y, y, t)["mean"] == pytest.approx(1.0)
    assert spread_ratio(y, 0.5 * y, t)["mean"] == pytest.approx(0.5)


def test_identity_features_and_dtw():
    y, t, ids, _ = synthetic()
    feats = feature_table(y, t, ids, y, t, ids)
    r = feature_correlation(feats)
    assert all(v == pytest.approx(1.0) for v in r.values())
    assert all(c["pred"] == 0 and c["true"] == 0 and c["used"] == 30 for c in edge_counts(feats).values())
    sh = shape_error(y, y, ids)
    assert sh["dtw_rmse"] == pytest.approx(0.0, abs=1e-12) and sh["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_rigid_predictor_features_undefined_or_low():
    y, t, ids, _ = synthetic()
    r = feature_correlation(feature_table(y, t, ids, rigid(y, t, ids), t, ids))
    assert all(np.isnan(v) for v in r.values())  # constant features across trials


def test_trial_features_known_values():
    t = np.arange(0, 3.0, 1 / FS)
    s = np.clip((t - 0.5) / 1.0, 0, 1)
    u = np.sin(np.pi * s)
    y = np.stack([3 * u, 4 * u, 5 * u ** 2], axis=1)  # |y| = 5 sqrt(u^2 + u^4), max sqrt(50) at t = 1 s
    f, e = trial_features(y, t)
    assert not any(e.values())
    assert f["peak_height"] == pytest.approx(5.0, abs=1e-3)
    assert f["reach_distance"] == pytest.approx(np.sqrt(50), abs=1e-3)
    u_on = np.sqrt((np.sqrt(1 + 4 * 0.02) - 1) / 2)  # u^2 + u^4 = 2 * 0.1^2
    t_on = 0.5 + np.arcsin(u_on) / np.pi
    assert f["onset_time"] == pytest.approx(t_on, abs=0.011)
    assert f["duration"] == pytest.approx(1.0 - t_on, abs=0.011)
    _, e2 = trial_features(y[60:], t[60:])  # span starts after the movement started: onset on the edge
    assert e2["onset_time"] and e2["duration"] and not e2["peak_height"]


def test_dtw_removes_pure_timing_error():
    t = np.arange(0, 3.0, 1 / FS)
    shape = lambda on: np.stack([np.clip((t - on) / 1.0, 0, 1)] * 3, axis=1) * 10  # noqa: E731
    a, b = shape(0.5), shape(0.8)
    assert dtw_rmse(a, b) < 0.05 < np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
    p = dtw_path(a, b)
    assert tuple(p[0]) == (0, 0) and tuple(p[-1]) == (len(a) - 1, len(b) - 1)
    assert np.all(np.diff(p, axis=0) >= 0)
    assert dtw_rmse(a, b, band=5) > dtw_rmse(a, b)  # 0.3 s shift does not fit a 0.05 s band


def test_condition_sensitivity_signs():
    y, t, ids, cond = synthetic()
    feats = feature_table(y, t, ids, y, t, ids)
    tab = condition_sensitivity(feats, cond).set_index("feature")
    assert tab.loc["peak_height", "true_diff"] > 0 and tab.loc["peak_height", "same_sign"]
    assert tab.loc["peak_height", "ratio"] == pytest.approx(1.0)
    flat = condition_sensitivity(feature_table(y, t, ids, rigid(y, t, ids), t, ids), cond).set_index("feature")
    assert not flat.loc["peak_height", "same_sign"]


def test_residual_share():
    y, t, ids, _ = synthetic()
    assert residual_share(np.zeros_like(y), y) == 0.0
    base = rigid(y, t, ids)
    d = y - base
    share = residual_share(d, y)
    assert share == pytest.approx(np.sum(d ** 2) / np.sum((base - base.mean(0)) ** 2))


def test_subject_flexibility_runs_and_requires_contiguous_trials():
    y, t, ids, cond = synthetic(12)
    out = subject_flexibility(y, y, ids, t, condition=cond, delta=np.zeros_like(y), dtw_downsample=5)
    assert out["n_trials"] == 12 and out["spread_ratio"]["mean"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        feature_table(y, t, ids, y, t, np.r_[ids[1:], ids[:1]])


def test_prediction_features_ignore_the_true_onset_when_computed_on_the_full_stream():
    """Shifting the true onset (LED fixed) moves the scored window; features of a fixed full-stream prediction must
    not change, while the same features computed on the scored window do."""
    t_full = np.arange(-1.3, 6.0, 1 / FS)
    y_pred_full = lift(t_full, 0.6, 1.4, 6.0, 1.0)
    res = {}
    for onset in (0.4, 1.3):  # at 1.3 s the window starts after the predicted onset (0.6 s)
        y_true_full = lift(t_full, onset, 1.4, 6.0, 1.0)
        win = (t_full >= onset - 0.5) & (t_full <= onset + 1.4)
        ids, ids_full = np.zeros(win.sum(), int), np.zeros(t_full.size, int)
        full = feature_table(y_true_full[win], t_full[win], ids, y_pred_full, t_full, ids_full)
        window = feature_table(y_true_full[win], t_full[win], ids, y_pred_full[win], t_full[win], ids)
        res[onset] = (full, window)
    cols = [f"pred_{k}" for k in ("onset_time", "duration", "peak_height", "reach_distance", "peak_speed")]
    pd.testing.assert_frame_equal(res[0.4][0][cols], res[1.3][0][cols])
    assert res[0.4][0].true_onset_time[0] != pytest.approx(res[1.3][0].true_onset_time[0])
    w4, w9 = res[0.4][1], res[1.3][1]
    assert not w4[cols].fillna(-1).equals(w9[cols].fillna(-1))  # the scored window leaks the true onset


def stream_lift(out_idx: np.ndarray, seed: int = 0) -> LiftStream:
    g = np.random.default_rng(seed)
    n_patches = 40
    t = 180 + 20 * n_patches
    return LiftStream("P1", 0, g.standard_normal((32, t)).astype(np.float32),
                      g.standard_normal((2, t)).astype(np.float32), out_idx, np.arange(out_idx.size),
                      np.zeros((out_idx.size, 3), np.float32), np.zeros(out_idx.size, np.float32), j0=50,
                      n_patches=n_patches, lift_len=50 + 790, orig_offset=0, led_s=2.0)


def test_full_stream_prediction_independent_of_the_scored_window():
    """Two scored windows (true onsets 0.3 s apart, same LED and lift window): identical full-stream outputs."""
    a, b = stream_lift(np.arange(250, 700)), stream_lift(np.arange(280, 760))
    fa, fb = full_stream_lift(a), full_stream_lift(b)
    assert np.array_equal(fa.out_idx, fb.out_idx) and np.array_equal(fa.t_since_led, fb.t_since_led)
    assert fa.out_idx[0] == 19 and fa.out_idx[-1] == 790 - 1
    assert fa.t_since_led[0] == pytest.approx((50 + 19 + 1) / FS - 2.0)
    dec = decoder("proposed", prior=True)
    with torch.no_grad():
        for q in dec.out.parameters():
            q.normal_(std=0.3)
    cfg = {"batch_lifts": 1}
    pa = predict(dec, [fa], {}, None, torch.device("cpu"), cfg)[("P1", 0)]
    pb = predict(dec, [fb], {}, None, torch.device("cpu"), cfg)[("P1", 0)]
    assert np.array_equal(pa["y"], pb["y"])
    scored = predict(dec, [a], {}, None, torch.device("cpu"), cfg)[("P1", 0)]  # causal: same values when scored
    assert np.abs(pa["y"][a.out_idx - 19] - scored["y"]).max() < 1e-4
