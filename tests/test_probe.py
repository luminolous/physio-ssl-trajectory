"""Tests for the linear-probe health check (real cache only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegtraj.baselines.template import time_since_event
from eegtraj.models.probe import load_probe_subject

CACHE = Path("dataset/cache/98f5fa6e66")


@pytest.mark.skipif(not (CACHE / "P3" / "pre_lift_eeg_broadband_car.npy").exists(), reason="needs the main cache")
def test_probe_windows_end_at_their_target_on_a_shared_grid():
    p = load_probe_subject(str(CACHE), "P3", window=200, stride=20)
    s = p.samples
    bins = np.round(time_since_event(s, "ev_LEDOn", p.mask) * s.fs_hz).astype(np.int64)
    assert (bins % 20 == 0).all()  # same grid for every lift
    assert p.windows.shape == (p.mask.sum(), 32, 200) and p.target.shape == (p.mask.sum(), 3)
    stats = s.eeg_for_stats().astype(np.float64)
    mean, sd = stats.mean(axis=1), stats.std(axis=1)
    for k in (0, len(p.windows) // 2, len(p.windows) - 1):
        i = s.sample_idx[p.mask][k]
        expected = (s.eeg[:, i - 199:i + 1] - mean[:, None]) / sd[:, None]
        assert np.allclose(p.windows[k], expected, atol=1e-3)  # last column is the target sample: no later EEG
    assert abs(float(p.windows.mean())) < 0.05 and 0.9 < float(p.windows.std()) < 1.1
