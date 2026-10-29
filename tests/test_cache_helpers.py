"""Tests for the pure helpers in eegtraj.data.cache."""

from __future__ import annotations

import numpy as np

from eegtraj.data.cache import PHASES, config_hash, keep_indices, phase_labels

FS, FS_OUT, FS_EMG = 500, 100, 4000


def test_config_hash_ignores_paths_and_subjects_only():
    base = {"paths": {"raw_dir": "a"}, "subjects": ["P1"], "fs_out_hz": 100}
    assert config_hash(base) == config_hash({**base, "paths": {"raw_dir": "b"}, "subjects": ["P2", "P3"]})
    assert config_hash(base) != config_hash({**base, "fs_out_hz": 50})


def test_kept_samples_share_one_time_grid_across_streams():
    keep = keep_indices(4714, FS // FS_OUT)
    t_eeg = (keep + 1) / FS
    assert np.allclose(t_eeg, np.arange(1, len(keep) + 1) / FS_OUT)
    # EMG sample used for EEG sample k is 8 (k + 1) - 1; its timestamp must match.
    s0 = 1234
    emg_idx = (FS_EMG // FS) * (s0 + keep + 1) - 1
    assert np.allclose((emg_idx + 1) / FS_EMG, (s0 + keep + 1) / FS)


def test_phase_labels():
    events = {"onset": 2.4, "tFirstDigitTouch": 3.5, "tLiftOff": 3.9, "tReplace": 7.0,
              "tBothReleased": 7.4, "tHandStop": 8.0}
    t = np.array([0.1, 2.39, 2.4, 3.0, 3.6, 5.0, 7.2, 7.9, 8.5])
    labels = phase_labels(t, events)
    assert [PHASES[i] for i in labels] == ["pre", "pre", "reach", "reach", "load", "hold", "replace", "release", "post"]
    assert labels.dtype == np.int8
