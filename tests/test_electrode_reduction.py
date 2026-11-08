"""Tests for the electrode reduction montages and the linked-mastoid re-reference (synthetic data)."""

import json

import numpy as np
import pandas as pd
import pytest

from eegtraj.data.windows import load_subject_samples
from eegtraj.preprocessing.reference import CHANNEL_SETS, LINKED_MASTOID, channel_indices, common_average, rereference_rows

NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9",
         "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]
MONTAGES = {"central9": 9, "contralateral": 6, "minimal3": 3, "cz": 1, "headband_ring": 10}


def test_montages_exist_and_have_the_expected_size():
    for name, n in MONTAGES.items():
        idx = channel_indices(NAMES, name)
        assert len(idx) == n == len(set(idx))
    assert set(CHANNEL_SETS["contralateral"]) == {"Cz", "FC1", "FC5", "C3", "CP1", "CP5"}
    assert not set(CHANNEL_SETS["headband_ring"]) & {"Cz", "C3", "C4", "FC1", "FC2"}  # the ring misses the vertex


def test_linked_mastoid_is_independent_of_the_input_reference():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(32, 500))
    car = common_average(raw)
    other_ref = raw - raw[9]  # e.g. an online reference at one electrode
    rows = channel_indices(NAMES, "minimal3")
    expected = raw[rows] - raw[[NAMES.index(c) for c in LINKED_MASTOID]].mean(axis=0)
    for x in (raw, car, other_ref):
        np.testing.assert_allclose(rereference_rows(x, NAMES, rows, "linked_mastoid"), expected, atol=1e-5)
    np.testing.assert_array_equal(rereference_rows(car, NAMES, rows, None), car[rows].astype(np.float32))
    # One channel keeps a signal (a subset average would be zero).
    assert np.abs(rereference_rows(car, NAMES, channel_indices(NAMES, "cz"), "linked_mastoid")).max() > 0.1
    with pytest.raises(ValueError, match="unknown rereference"):
        rereference_rows(car, NAMES, rows, "average_subset")


def test_rereference_works_on_trailing_axes():
    pre = np.random.default_rng(1).normal(size=(32, 3, 7)).astype(np.float32)  # (C, n_lifts, n_stored)
    rows = channel_indices(NAMES, "cz")
    out = rereference_rows(pre, NAMES, rows, "linked_mastoid")
    assert out.shape == (1, 3, 7)
    np.testing.assert_allclose(out[0], pre[13] - (pre[16] + pre[21]) / 2, atol=1e-6)


def small_cache(root):
    subject = root / "P1"
    subject.mkdir()
    trials = pd.DataFrame({"subject": "P1", "series": 1, "lift": [1, 2], "n_samples": 40, "segment_start_s": 0.1,
                           "segment_end_s": 0.3, "offset_start": [0, 40], "offset_end": [40, 80]})
    trials.to_csv(subject / "trials.csv", index=False)
    (subject / "manifest.json").write_text(json.dumps({"fs_hz": 100.0, "eeg_names": NAMES}), encoding="utf-8")
    rng = np.random.default_rng(2)
    np.save(subject / "eeg_broadband_car.npy", rng.normal(size=(32, 80)).astype(np.float32))
    np.save(subject / "pre_lift_eeg_broadband_car.npy", rng.normal(size=(32, 2, 5)).astype(np.float32))
    np.save(subject / "wrist_vel.npy", np.zeros((3, 80), np.float32))
    np.save(subject / "kin_pos.npy", np.zeros((3, 80), np.float32))


def test_loader_rereferences_lift_and_pre_lift_eeg(tmp_path):
    small_cache(tmp_path)
    args = (tmp_path, "P1", "broadband", "car")
    every = load_subject_samples(*args, "all", "velocity", 6, pre_lift=True)
    ring = load_subject_samples(*args, "headband_ring", "velocity", 6, pre_lift=True, rereference="linked_mastoid")
    ref = every.eeg[[NAMES.index(c) for c in LINKED_MASTOID]].mean(axis=0)
    expected = every.eeg[channel_indices(NAMES, "headband_ring")] - ref
    np.testing.assert_allclose(ring.eeg, expected, atol=1e-5)  # every sample, including the pre-lift ones
    assert np.array_equal(ring.sample_idx, every.sample_idx)
    assert np.array_equal(ring.eeg_stat_mask, every.eeg_stat_mask)


def test_montage_cost_counts_parameters_and_times_cpu():
    from eegtraj.diagnostics.compute_cost import montage_cost

    kwargs = dict(n_temporal=8, temporal_kernel=25, n_spatial=16, pool=2, hidden=32, lstm_layers=1, drop_prob=0.25)
    a = montage_cost("cnn_lstm", kwargs, 32, 200, ["cpu"], 1, 3, 0)
    b = montage_cost("cnn_lstm", kwargs, 1, 200, ["cpu"], 1, 3, 0)
    assert a["n_params"] == 10843  # value logged by the 32-channel runs
    assert a["n_params"] - b["n_params"] == a["n_params_spatial"] - b["n_params_spatial"] == 31 * 8 * 16
    assert a["latency_cpu_median_ms"] > 0
