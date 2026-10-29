"""Tests for eegtraj.preprocessing.onset on synthetic reaches."""

from __future__ import annotations

import numpy as np

from eegtraj.preprocessing.onset import kinematic_onset

FS = 500.0
PARAMS = dict(fs_hz=FS, threshold_frac=0.05, cutoff_hz=10.0, order=2)


def minimum_jerk(t: np.ndarray, t0: float, duration: float, distance: float) -> np.ndarray:
    tau = np.clip((t - t0) / duration, 0.0, 1.0)
    return distance * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def test_onset_of_minimum_jerk_reach():
    t = np.arange(1, int(8 * FS) + 1) / FS
    pos = np.zeros((t.size, 3))
    pos[:, 1] = minimum_jerk(t, t0=2.4, duration=1.0, distance=-25.0)
    onset, peak, _ = kinematic_onset(t, pos, t_search_start_s=2.0, t_search_end_s=4.0, **PARAMS)
    assert abs(peak - 2.9) < 0.02  # minimum-jerk speed peaks at mid-movement
    assert 2.4 <= onset <= 2.55  # 5% of peak speed is reached shortly after true onset


def test_earlier_movement_outside_search_is_ignored():
    # A larger reach before LED-on (as in mislabeled lifts) must not be picked when
    # the search starts at LED-on; a later small adjustment is found instead.
    t = np.arange(1, int(8 * FS) + 1) / FS
    pos = np.zeros((t.size, 3))
    pos[:, 1] = minimum_jerk(t, 0.5, 1.0, -25.0) + minimum_jerk(t, 3.0, 0.5, -1.0)
    onset, peak, _ = kinematic_onset(t, pos, t_search_start_s=2.0, t_search_end_s=4.0, **PARAMS)
    assert 3.0 <= onset <= 3.1
    assert 3.2 <= peak <= 3.3
