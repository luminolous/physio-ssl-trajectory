"""Tests for the movement timelapse helpers (eegtraj.visualization.timelapse)."""

import numpy as np
import pandas as pd
import pytest

from eegtraj.visualization.timelapse import (
    event_frames,
    frame_times,
    hold_interval,
    lift_predictions,
    lift_times,
    movement_phases,
    phase_at,
    phase_samples,
    sample_at,
    select_lifts,
)

FS = 100.0


def test_frame_times_follow_fps_and_speed():
    t = frame_times(-1.0, 7.0, 25, 1.0)
    assert t.size == 201 and t[0] == -1.0 and t[-1] == pytest.approx(7.0)
    np.testing.assert_allclose(np.diff(t), 0.04)
    t2 = frame_times(-1.0, 7.0, 25, 2.0)
    assert t2.size == 101
    np.testing.assert_allclose(np.diff(t2), 0.08)


def test_cursor_sample_matches_the_100hz_grid():
    tr = pd.Series({"n_samples": 900, "ev_LEDOn": 2.0})
    t = lift_times(tr, FS)  # sample j at (j + 1) / fs - 2 s
    assert t[0] == pytest.approx(-1.99) and t[199] == pytest.approx(0.0)
    # Real time at 25 fps: 4 samples per frame; the cursor sample is the last one at or before the frame time.
    times = frame_times(-1.0, 3.0, 25, 1.0)
    samples = np.array([sample_at(t, x) for x in times])
    np.testing.assert_array_equal(np.diff(samples), 4)
    np.testing.assert_allclose(t[samples], times, atol=1e-9)
    assert sample_at(t, 0.005) == 199  # between samples: the earlier one


def test_event_frames_put_the_cursor_on_each_event():
    events = {"LED on": 0.0, "onset": 0.42, "first touch": 1.37, "hand stop": 5.9}
    fr = event_frames(events, -1.0, 25, 1.0)
    times = frame_times(-1.0, 6.4, 25, 1.0)
    for label, k in fr.items():
        assert abs(times[k] - events[label]) <= 0.5 / 25 + 1e-12
    assert fr["LED on"] == 25


EVENTS = {"LED on": 0.0, "onset": 0.4, "first touch": 1.4, "lift-off": 1.9, "replace": 5.0, "release": 5.3,
          "hand stop": 6.0}


def test_hold_interval_from_object_height():
    t = np.arange(1, 801) / FS - 1.0  # -0.99 .. 7.0 s
    z = np.full(t.size, 5.0)
    rise = (t >= 1.9) & (t < 2.4)
    z[rise] = 5.0 + 6.0 * (t[rise] - 1.9) / 0.5  # rises 6 cm in 0.5 s, holds, lowers 4.0-5.0 s
    z[(t >= 2.4) & (t < 4.0)] = 11.0
    low = (t >= 4.0) & (t < 5.0)
    z[low] = 11.0 - 6.0 * (t[low] - 4.0)
    a, b = hold_interval(t, z, 1.9, 5.0)
    assert a == pytest.approx(2.35) and b == pytest.approx(4.1)  # 90 % of the 6 cm rise = 10.4 cm


def test_phases_and_phase_names():
    phases = movement_phases(EVENTS, (2.4, 4.0))
    assert list(phases) == ["reach", "lift", "hold", "replace", "return"]
    assert phases["reach"] == (0.4, 1.4) and phases["hold"] == (2.4, 4.0) and phases["return"] == (5.3, 6.0)
    assert phase_at(phases, EVENTS, -0.5) == "rest"
    assert phase_at(phases, EVENTS, 0.2) == "reaction"
    assert phase_at(phases, EVENTS, 1.0) == "reach"
    assert phase_at(phases, EVENTS, 2.0) == "lift"
    assert phase_at(phases, EVENTS, 3.0) == "hold"
    assert phase_at(phases, EVENTS, 4.5) == "replace"
    assert phase_at(phases, EVENTS, 5.5) == "return"
    assert phase_at(phases, EVENTS, 6.5) == "rest"
    with pytest.raises(ValueError, match="order"):
        movement_phases(EVENTS, (4.0, 2.4))


def test_phase_samples_are_continuous_and_stop_at_the_cursor():
    t = np.arange(1, 801) / FS - 1.0
    phases = movement_phases(EVENTS, (2.4, 4.0))
    i0 = sample_at(t, -0.5)
    full = phase_samples(t, phases, i0, len(t) - 1)
    assert list(full) == ["before", "reach", "lift", "hold", "replace", "return", "after"]
    pieces = list(full.values())
    assert pieces[0].start == i0 and pieces[-1].stop == len(t)
    for p, q in zip(pieces, pieces[1:]):
        assert p.stop - 1 == q.start  # shared boundary sample
    assert t[full["reach"].start] == pytest.approx(0.4) and t[full["return"].stop - 1] == pytest.approx(6.0)
    mid = phase_samples(t, phases, i0, sample_at(t, 1.0))  # cursor inside the reach
    assert list(mid) == ["before", "reach"] and mid["reach"].stop - 1 == sample_at(t, 1.0)
    assert list(phase_samples(t, phases, i0, sample_at(t, 0.4))) == ["before"]


def test_lift_predictions_place_saved_samples_on_the_window(tmp_path):
    trials = pd.DataFrame({"subject": "P1", "series": 1, "lift": [1, 2], "n_samples": [50, 60],
                           "segment_start_s": [0.1, 0.2], "segment_end_s": [0.3, 0.45], "offset_start": [0, 50],
                           "offset_end": [50, 110]})
    rows = np.r_[np.zeros(21, np.int32), np.ones(26, np.int32)]  # samples at 0.10..0.30 s and 0.20..0.45 s
    y_true = np.arange(47 * 3, dtype=np.float32).reshape(47, 3)
    np.savez(tmp_path / "predictions.npz", P1__trial_row=rows, P1__y_pred=y_true + 0.5, P1__y_true=y_true)
    wrist = np.full((60, 3), -1.0)
    wrist[19:45] = y_true[21:]
    out = lift_predictions(tmp_path, "P1", trials, 1, FS, wrist)
    assert out.shape == (60, 3)
    assert np.isnan(out[:19]).all() and np.isnan(out[45:]).all()
    np.testing.assert_array_equal(out[19:45], y_true[21:] + 0.5)  # sample j at (j + 1) / fs: 0.20 s is j = 19
    with pytest.raises(ValueError, match="differ"):
        lift_predictions(tmp_path, "P1", trials, 1, FS, wrist + 1)


def test_select_lifts_uses_distinct_subjects_and_rules():
    rng = np.random.default_rng(0)
    trials = {}
    for s in ("P1", "P2", "P7", "P8"):
        n = 40
        trials[s] = pd.DataFrame({"series": 1, "lift": np.arange(1, n + 1), "weight_g": rng.choice([165, 330, 660], n),
                                  "surface": "silk", "ev_LEDOn": 2.0, "onset_s": 2.0 + rng.uniform(0.2, 1.0, n),
                                  "onset_mismatch": False, "filter_transient_broadband": False})
    rules = {"typical": {"rt_percentile": [40, 60]}, "slow": {"rt_percentile": [95, 100]}, "P7": {"subject": "P7"},
             "heavy": {"weight_g": 660}}
    a = select_lifts(trials, rules, seed=3)
    b = select_lifts(trials, rules, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.rule) == list(rules)
    assert a.subject.nunique() == 4
    assert a.set_index("rule").loc["P7", "subject"] == "P7"
    assert a.set_index("rule").loc["heavy", "weight_g"] == 660
