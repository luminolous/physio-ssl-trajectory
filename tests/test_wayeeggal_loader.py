"""Loader tests on the real P1 files (skipped when dataset/raw/P1 is missing)."""

from __future__ import annotations

import numpy as np
import pytest

from eegtraj.data import raw_io
from eegtraj.data.wayeeggal import load_series, load_trials

SUBJECT = "P1"
pytestmark = pytest.mark.skipif(
    not (raw_io.DEFAULT_RAW_DIR / SUBJECT).is_dir(), reason="dataset/raw/P1 not available"
)

EVENT_ORDER = ["LEDOn", "tFirstDigitTouch", "tBothDigitTouch", "tLiftOff", "tReplace", "tBothReleased", "tHandStop"]


@pytest.fixture(scope="module")
def lifts():
    return raw_io.load_alllifts(SUBJECT)


@pytest.fixture(scope="module")
def trials_by_series(lifts):
    return {k: load_trials(SUBJECT, k, lifts=lifts) for k in raw_io.SERIES}


def test_trial_counts_match_alllifts(trials_by_series, lifts):
    for k, trials in trials_by_series.items():
        assert len(trials) == int((lifts.Run == int(k[1:])).sum()), k
        assert [t.lift for t in trials] == list(range(1, len(trials) + 1))
    assert sum(len(t) for t in trials_by_series.values()) == len(lifts) == 294


def test_shapes_dtypes_and_rates(trials_by_series):
    for trials in trials_by_series.values():
        for t in trials:
            assert t.eeg.dtype == t.emg.dtype == t.kin_pos.dtype == np.float32
            assert t.eeg.shape == (32, t.n_times)
            assert t.emg.shape == (5, 8 * t.n_times)
            assert t.kin_pos.shape == (4, 3, t.n_times)
            assert t.eeg_fs_hz == 500.0 and t.emg_fs_hz == 4000.0
            assert len(t.eeg_names) == 32 and t.eeg_names[0] == "Fp1" and len(t.emg_names) == 5
            assert np.isfinite(t.kin_pos).all()


def test_event_ordering_and_window_bounds(trials_by_series):
    for trials in trials_by_series.values():
        for t in trials:
            # Non-decreasing: both digits can touch (or release) in the same sample.
            times = [t.events[e] for e in EVENT_ORDER]
            assert all(a <= b for a, b in zip(times, times[1:])), (t.series, t.lift, times)
            assert t.events["tLiftOff"] < t.events["tReplace"] < t.events["tHandStop"]
            assert t.events["LEDOn"] == 2.0
            assert 0 < t.events["tHandStart"] < t.events["tFirstDigitTouch"]
            assert t.events["tHandStop"] <= t.n_times / t.eeg_fs_hz
            assert "tPeakVelHandReach" not in t.events and "tPeakVelHandReach" in t.measures


def test_conditions(trials_by_series):
    for trials in trials_by_series.values():
        for t in trials:
            assert t.weight_g in (165, 330, 660)
            assert t.surface in ("silk", "suede", "sandpaper")
            assert t.block_type in (1, 2, 3)
        assert trials[0].prev_weight_g is None and trials[0].prev_surface is None
        assert trials[1].prev_weight_g == trials[0].weight_g


def test_ws_windows_are_hs_slices(trials_by_series):
    series = load_series(SUBJECT, "S1")
    assert series.eeg.shape[0] == 32 and series.kin_pos.shape[:2] == (4, 3)
    for t in trials_by_series["S1"]:
        s = t.hs_start_sample
        np.testing.assert_array_equal(series.eeg[:, s:s + t.n_times], t.eeg)
        np.testing.assert_array_equal(series.kin_pos[:, :, s:s + t.n_times], t.kin_pos)
        np.testing.assert_array_equal(series.emg[:, 8 * s:8 * (s + t.n_times)], t.emg)


def test_st_series_has_eeg_only():
    st = load_series(SUBJECT, "ST")
    assert st.eeg.shape[0] == 32 and st.emg is None and st.kin_pos is None
