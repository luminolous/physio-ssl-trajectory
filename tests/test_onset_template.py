"""Tests for the onset decoder + template floor (synthetic data)."""

import numpy as np
import pytest

from eegtraj.baselines import FitContext, make_baseline
from eegtraj.baselines.onset_template import first_crossing, onset_error_stats
from eegtraj.data.windows import Standardizer
from tests.test_baselines import FS, synthetic_subject

PARAMS = {"thresholds": [0.5, 0.9, 1.0], "label_window_s": 0.5, "shrinkage": 0.01, "deadline_quantile": 0.95,
          "within_ms": 100}
N_CHANNELS = 4
LED_S = 2.0


def subject(name: str, onsets: list[float], seed: int, signal: float = 1.0):
    """Subject whose EEG steps up at the kinematic onset on channel 0, with noise on all channels."""
    s = synthetic_subject(name, onsets)
    s.trials = s.trials.assign(ev_LEDOn=LED_S)
    rng = np.random.default_rng(seed)
    eeg = 0.3 * rng.standard_normal((N_CHANNELS, s.target.shape[1]))
    for tr in s.trials.itertuples(index=False):
        t = (np.arange(tr.n_samples) + 1) / FS
        eeg[0, tr.offset_start:tr.offset_end] += signal * (t >= tr.onset_s)
    s.eeg = eeg.astype(np.float32)
    return s


def context(subjects, train_names, val_names):
    unit = Standardizer(np.zeros((N_CHANNELS, 1), np.float32), np.ones((N_CHANNELS, 1), np.float32))
    target = Standardizer(np.zeros((3, 1), np.float32), np.ones((3, 1), np.float32))
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subjects}
    return FitContext(eeg_norm={s.subject: unit for s in subjects}, target_norm=target, history=50, lag_step=10, seed=0,
                      train_masks=ones, val_masks=ones)


def fitted():
    train = [subject("P1", [2.2, 2.4, 2.6, 2.3, 2.5, 2.7, 2.35, 2.55], 1), subject("P2", [2.25, 2.45, 2.65, 2.3, 2.5, 2.6], 2)]
    val = [subject("P4", [2.3, 2.5, 2.7, 2.4], 3)]
    test = subject("P5", [2.15, 2.6, 2.45], 4)
    ctx = context(train + val + [test], None, None)
    model = make_baseline("onset_template", PARAMS)
    model.fit(train, val, ctx)
    return model, test, ctx


def test_first_crossing():
    assert first_crossing(np.array([0.1, 0.6, 0.9]), 0.5) == 1
    assert first_crossing(np.array([0.1, 0.2]), 0.5) is None


def test_onset_error_stats():
    stats = onset_error_stats(np.array([1.05, 1.0, 1.5]), np.array([1.0, 1.0, 1.0]), 100.0)
    assert stats["median_ms"] == pytest.approx(50.0)
    assert stats["pct_within"] == pytest.approx(100 * 2 / 3)


def test_decodes_onset_and_follows_the_template():
    model, test, ctx = fitted()
    pred = model.predict(test, np.ones(len(test.sample_idx), bool), ctx)
    err = model.info()["onset_error"]["P5"]
    assert abs(err["median_ms"]) <= 100 and err["frac_deadline"] == 0.0
    true = test.target[:, test.sample_idx].T
    assert min(np.corrcoef(pred[:, a], true[:, a])[0, 1] for a in range(3)) > 0.9
    assert model.threshold in PARAMS["thresholds"][:-1]  # a detector beats the deadline-only setting


def test_prediction_uses_only_eeg_up_to_each_sample():
    model, test, ctx = fitted()
    mask = np.ones(len(test.sample_idx), bool)
    first = model.predict(test, mask, ctx)
    cut = len(test.sample_idx) // 2
    test.eeg[:, int(test.sample_idx[cut]) + 1:] += 50.0
    second = model.predict(test, mask, ctx)
    assert np.array_equal(first[: cut + 1], second[: cut + 1])
    assert not np.array_equal(first[cut + 1:], second[cut + 1:])


def test_prediction_ignores_the_test_onset():
    model, test, ctx = fitted()
    mask = np.ones(len(test.sample_idx), bool)
    first = model.predict(test, mask, ctx)
    test.trials = test.trials.assign(onset_s=test.trials.onset_s + 0.4)
    assert np.array_equal(model.predict(test, mask, ctx), first)


def test_no_detection_before_led_and_deadline_fallback():
    model, test, ctx = fitted()
    mask = np.ones(len(test.sample_idx), bool)
    model.threshold = 1.0  # never fires
    pred = model.predict(test, mask, ctx)
    rows = test.trial_row[mask]
    assert model.info()["onset_error"]["P5"]["frac_deadline"] == 1.0
    for row in np.unique(rows):
        p = pred[rows == row]
        assert not np.allclose(p, p[0])  # every lift still moves, so r stays defined
    t = (test.sample_idx - test.trials.offset_start.to_numpy()[rows] + 1) / FS
    before = t < LED_S + model.deadline_s - 1 / FS
    assert np.allclose(pred[before], model.pre_value)
