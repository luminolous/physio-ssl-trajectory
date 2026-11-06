"""Tests for EEG before the lift window (synthetic data)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eegtraj.baselines.template import time_since_onset
from eegtraj.data.pre_lift import insert_pre_lift, pre_lift_raw_indices
from eegtraj.data.windows import SubjectSamples, fit_eeg_normalizers, load_subject_samples, window_at


def two_lifts():
    trials = pd.DataFrame({"offset_start": [0, 4], "offset_end": [4, 10], "onset_s": [0.02, 0.03]})
    eeg = np.arange(20, dtype=np.float32).reshape(2, 10)
    target = 10 * eeg[:1].repeat(3, axis=0)
    pre = -np.arange(1, 37, dtype=np.float32).reshape(2, 2, 9)  # (C, n_lifts, n_stored)
    return eeg, target, trials, pre


def test_raw_indices_follow_the_lift_grid():
    # Lift sample j = 0 is raw hs_start + 4 at factor 5; j = -1 is 5 raw samples earlier.
    assert pre_lift_raw_indices(100, 5, 3).tolist() == [89, 94, 99]
    assert pre_lift_raw_indices(1, 5, 2).tolist() == [-5, 0]


def test_insert_keeps_each_lift_and_puts_pre_samples_right_before_it():
    eeg, target, trials, pre = two_lifts()
    x, y, tr, keep = insert_pre_lift(eeg, target, trials, pre, 3)
    assert x.shape == (2, 16) and y.shape == (3, 16) and keep.sum() == 10
    assert tr.offset_start.tolist() == [3, 10] and tr.offset_end.tolist() == [7, 16]
    assert np.array_equal(x[:, keep], eeg) and np.array_equal(y[:, keep], target)
    assert np.array_equal(x[:, :3], pre[:, 0, -3:])  # newest stored samples, oldest first
    assert np.array_equal(x[:, 7:10], pre[:, 1, -3:])


def test_long_window_reads_this_lifts_own_past_not_the_previous_lift():
    eeg, target, trials, pre = two_lifts()
    x, _, tr, _ = insert_pre_lift(eeg, target, trials, pre, 3)
    w = window_at(x, int(tr.offset_start[1]) + 1, 5)  # lift 2, sample j = 1, history 5
    expected = np.concatenate([pre[:, 1, -3:], eeg[:, 4:6]], axis=1)
    assert np.array_equal(w, expected)


def subject_with_pre():
    eeg, target, trials, pre = two_lifts()
    x, y, tr, keep = insert_pre_lift(eeg, target, trials, pre, 3)
    idx = tr.offset_start.to_numpy() + 1
    return (SubjectSamples("P1", x, y, idx, np.array([0, 1], np.int32), tr, 100.0, eeg_stat_mask=keep), eeg, target, idx)


def test_normalizer_uses_lift_samples_only():
    s, eeg, _, _ = subject_with_pre()
    norm = fit_eeg_normalizers([s], ("P1",), "per_subject")["P1"]
    assert np.allclose(norm.mean[:, 0], eeg.mean(axis=1))


def test_targets_and_event_times_are_unchanged():
    s, _, target, _ = subject_with_pre()
    assert np.array_equal(s.target[:, s.sample_idx], target[:, [1, 5]])
    # Sample j = 1 is at (j + 1) / fs = 0.02 s.
    assert np.allclose(time_since_onset(s), [0.0, -0.01])


def compare_history():
    spec = importlib.util.spec_from_file_location("compare_history", Path("scripts/compare_history.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(r, r2, lrs=None):
    folds = [{"baseline_info": {"selected_train": {"lr": lr}}} for lr in lrs or []]
    return {"seed": 1, "aggregate": {"r": {"mean": r}, "r2": {"mean": r2}}, "per_fold": folds}


def test_fixed_lr_is_safe_only_if_r_and_r2_stay_within_tolerance():
    check = compare_history().lr_check
    fixed = fake_run(0.5, 0.2)
    assert check(fake_run(0.509, 0.191, [1e-3, 3e-4]), fixed)["safe"]
    assert not check(fake_run(0.52, 0.2, [3e-4]), fixed)["safe"]
    assert not check(fake_run(0.5, 0.22, [3e-4]), fixed)["safe"]
    assert not check(fake_run(0.48, 0.2, [3e-4]), fixed)["safe"]  # a worse grid also counts as a difference
    assert check(fake_run(0.5, 0.2, [1e-3, 1e-3, 3e-4]), fixed)["selected_lr_counts"] == {"0.0003": 1, "0.001": 2}


def tiny_cache(root):
    """Two lifts of 40 samples, 32 channels, 5 stored pre-lift samples; values differ per channel."""
    names = [f"E{i}" for i in range(32)]
    names[0], names[1] = "Fp1", "Fp2"
    subject = root / "P1"
    subject.mkdir()
    trials = pd.DataFrame({"subject": "P1", "series": 1, "lift": [1, 2], "n_samples": 40, "segment_start_s": 0.1,
                           "segment_end_s": 0.3, "offset_start": [0, 40], "offset_end": [40, 80]})
    trials.to_csv(subject / "trials.csv", index=False)
    (subject / "manifest.json").write_text(json.dumps({"fs_hz": 100.0, "eeg_names": names}), encoding="utf-8")
    eeg = np.arange(32 * 80, dtype=np.float32).reshape(32, 80)
    np.save(subject / "eeg_broadband_car.npy", eeg)
    np.save(subject / "pre_lift_eeg_broadband_car.npy", -1 - np.arange(32 * 2 * 5, dtype=np.float32).reshape(32, 2, 5))
    np.save(subject / "wrist_vel.npy", np.zeros((3, 80), np.float32))
    np.save(subject / "kin_pos.npy", np.zeros((3, 80), np.float32))
    return names


def test_channel_subset_reads_the_same_channels_before_and_inside_the_lift(tmp_path):
    tiny_cache(tmp_path)
    args = (tmp_path, "P1", "broadband", "car")
    every = load_subject_samples(*args, "all", "velocity", 6, pre_lift=True)
    part = load_subject_samples(*args, "prefrontal", "velocity", 6, pre_lift=True)
    assert part.eeg.shape[0] == 2
    assert np.array_equal(part.eeg, every.eeg[:2])
    assert np.array_equal(part.sample_idx, every.sample_idx)
    assert part.eeg[0, 0] < 0  # the first samples are pre-lift


def test_emg_is_rejected_with_pre_lift(tmp_path):
    tiny_cache(tmp_path)
    np.save(tmp_path / "P1" / "emg_env.npy", np.zeros((5, 80), np.float32))
    with pytest.raises(ValueError, match="EEG only"):
        load_subject_samples(tmp_path, "P1", "broadband", "car", "emg", "velocity", 6, pre_lift=True)
