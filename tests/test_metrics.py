"""Tests for eegtraj.metrics against scipy and hand-computed cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from eegtraj.metrics import (
    aggregate_across_subjects,
    aggregate_subject,
    nrmse,
    pearson_r,
    per_trial_metrics,
    pooled_r,
    r2_score,
    rmse,
)


@pytest.mark.parametrize("seed", range(5))
def test_pearson_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(300) * 10 + 30  # large offset, as for positions in cm
    b = 0.6 * a + rng.standard_normal(300) * 5
    assert pearson_r(a, b) == pytest.approx(scipy.stats.pearsonr(a, b).statistic, abs=1e-12)


def test_hand_computed_cases():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_r(t, 2 * t + 5) == pytest.approx(1.0)
    assert pearson_r(t, -t) == pytest.approx(-1.0)
    p = np.array([1.0, 2.0, 2.0, 5.0])
    # residuals 0,0,1,-1 -> SS_res = 2; SS_tot = 5 -> R2 = 0.6; RMSE = sqrt(2/4); range = 3
    assert r2_score(t, p) == pytest.approx(0.6)
    assert rmse(t, p) == pytest.approx(np.sqrt(0.5))
    assert nrmse(t, p) == pytest.approx(np.sqrt(0.5) / 3)
    assert r2_score(t, np.full(4, t.mean())) == pytest.approx(0.0)  # predicting the mean


def test_constant_signal_edge_cases():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    c = np.full(4, 7.0)
    assert np.isnan(pearson_r(t, c)) and np.isnan(pearson_r(c, t))  # undefined, scipy also gives nan
    assert np.isnan(r2_score(c, t)) and np.isnan(nrmse(c, t))
    assert rmse(c, c) == 0.0
    assert np.isnan(pearson_r(np.array([1.0]), np.array([2.0])))
    assert np.isnan(pearson_r(np.full(5, 30.0) + 1e-14, t.tolist() + [5.0]))  # numerically constant


def test_per_trial_then_axis_then_subject_differs_from_pooled():
    # Two trials with identical shape but different offsets: perfect per-trial r, and pooled r
    # is driven by the offset (here the predictions get the offsets wrong -> pooled r < 1).
    t = np.linspace(0, 1, 50)
    y1 = np.stack([t, t**2, np.sin(3 * t)], axis=1)
    y_true = np.vstack([y1, y1 + 10])
    y_pred = np.vstack([y1 + 10, y1])
    ids = np.repeat([0, 1], 50)
    df = per_trial_metrics(y_true, y_pred, ids)
    assert len(df) == 6 and np.allclose(df.r, 1.0)
    agg = aggregate_subject(df)
    assert agg["r"]["mean"] == pytest.approx(1.0)
    assert pooled_r(y_true, y_pred)["mean"] < 0


def test_aggregation_order_and_nan_handling():
    df = pd.DataFrame({
        "trial": [0, 0, 1, 1, 2, 2],
        "axis": ["x", "y", "x", "y", "x", "y"],
        "r": [0.2, 0.8, 0.4, np.nan, 0.6, 0.4],
        "r2": [0.0] * 6, "rmse": [1.0] * 6, "nrmse": [0.1] * 6, "n": [10] * 6,
    })
    agg = aggregate_subject(df)
    # mean over trials per axis: x = 0.4, y = mean(0.8, 0.4) = 0.6 (NaN skipped); then over axes = 0.5
    assert agg["r"]["per_axis"] == pytest.approx({"x": 0.4, "y": 0.6})
    assert agg["r"]["mean"] == pytest.approx(0.5)
    assert agg["r"]["n_nan"] == 1


def test_across_subjects_mean_sd():
    def subj(v):
        return {m: {"mean": v, "per_axis": {"x": v, "y": v, "z": v}, "n_nan": 0} for m in ("r", "r2", "rmse", "nrmse")}
    out = aggregate_across_subjects({"P1": subj(0.2), "P2": subj(0.4), "P4": subj(0.9)})
    assert out["r"]["mean"] == pytest.approx(0.5)
    assert out["r"]["sd"] == pytest.approx(np.std([0.2, 0.4, 0.9], ddof=1))
    assert out["r"]["n_subjects"] == 3 and out["r"]["per_axis_mean"]["z"] == pytest.approx(0.5)


def test_per_trial_metrics_matches_scipy_per_trial():
    rng = np.random.default_rng(9)
    y_true = rng.standard_normal((120, 3))
    y_pred = y_true + rng.standard_normal((120, 3))
    ids = np.repeat(["a", "b", "c"], 40)
    df = per_trial_metrics(y_true, y_pred, ids)
    for (trial, axis), r in df.set_index(["trial", "axis"]).r.items():
        a = "xyz".index(axis)
        m = ids == trial
        assert r == pytest.approx(scipy.stats.pearsonr(y_true[m, a], y_pred[m, a]).statistic, abs=1e-12)
    with pytest.raises(ValueError):
        per_trial_metrics(y_true, y_pred[:, :2], ids)
