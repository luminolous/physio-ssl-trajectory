"""Tests for the residual metric (synthetic data)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from eegtraj.diagnostics.residual import SubjectTemplates, score_predictions
from eegtraj.metrics import residual_r


def trials(seed=0, n_trials=6, n=100):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    template = np.stack([np.sin(np.pi * t), t, t**2], axis=1)
    template = np.tile(template, (n_trials, 1))
    ids = np.repeat(np.arange(n_trials), n)
    # A smooth, trial-specific deviation from the template.
    wiggle = np.concatenate([np.outer(np.sin(2 * np.pi * (t * rng.uniform(1, 3) + rng.uniform())), rng.normal(size=3))
                             for _ in range(n_trials)])
    return template + wiggle, template, ids


def test_prediction_equal_to_target_gives_one():
    y, template, ids = trials()
    res = residual_r(y, y, template, ids)
    assert res["mean"] == pytest.approx(1.0) and res["n_nan"] == 0


def test_scaled_residual_still_gives_one():
    y, template, ids = trials()
    res = residual_r(y, template + 0.2 * (y - template) + 5.0, template, ids)
    assert res["mean"] == pytest.approx(1.0)


def test_prediction_equal_to_template_is_undefined_and_counted():
    y, template, ids = trials()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no all-NaN warning from the aggregation
        res = residual_r(y, template, template, ids)
    assert np.isnan(res["mean"]) and all(np.isnan(v) for v in res["per_axis"].values())
    assert res["n_nan"] == 6 * 3


def test_prediction_from_noise_is_near_zero():
    y, template, ids = trials()
    noise = np.random.default_rng(1).normal(size=y.shape)
    assert abs(residual_r(y, template + noise, template, ids)["mean"]) < 0.1


def test_trials_are_scored_separately():
    y, template, ids = trials()
    # A residual flipped in half of the trials cancels; pooled r over concatenated trials would not show this per trial.
    sign = np.where(ids % 2 == 0, 1.0, -1.0)[:, None]
    assert residual_r(y, template + sign * (y - template), template, ids)["mean"] == pytest.approx(0.0)


def templates_for(y, cue, oracle, ids):
    return {"P1": SubjectTemplates(y.astype(np.float32), ids.astype(np.int32), cue.astype(np.float32), oracle.astype(np.float32))}


def saved(y, pred, ids):
    return {"P1__y_true": y.astype(np.float32), "P1__y_pred": pred.astype(np.float32), "P1__trial_row": ids.astype(np.int32)}


def test_score_predictions_uses_each_template():
    y, cue, ids = trials()
    oracle = cue + 0.5 * (y - cue)
    row = score_predictions(saved(y, cue, ids), templates_for(y, cue, oracle, ids))[0]
    assert np.isnan(row["residual_r"]) and row["residual_r_n_nan"] == 18
    assert row["residual_r_oracle"] == pytest.approx(-1.0, abs=1e-4)  # cue - oracle opposes y - oracle


def test_score_predictions_rejects_misaligned_samples():
    y, cue, ids = trials()
    with pytest.raises(ValueError, match="differ"):
        score_predictions(saved(y[::-1], cue, ids), templates_for(y, cue, cue, ids))


def test_summary_averages_seeds_per_subject():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("summarize_runs", Path("scripts/summarize_runs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = pd.DataFrame({"run": ["a", "a", "b", "b"], "subject": ["P1", "P2", "P1", "P2"], "residual_r": [0.1, 0.3, 0.3, np.nan]})
    out = module.residual_per_subject(table, ["a", "b"], "residual_r")
    assert out["P1"] == pytest.approx(0.2) and out["P2"] == pytest.approx(0.3)
