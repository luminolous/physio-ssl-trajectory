"""Tests for band and channel screening: envelopes, stacked bands, and inner-validation scoring (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eegtraj.baselines import FitContext
from eegtraj.baselines.base import Baseline
from eegtraj.baselines.scoring import score_inner_validation
from eegtraj.data.band_variant import rectified_band
from eegtraj.data.windows import SubjectSamples, load_subject_samples, segment_samples
from eegtraj.preprocessing.filters import apply_sos, butter_sos
from eegtraj.preprocessing.reference import CHANNEL_SETS

FS = 500.0
BAND = butter_sos(FS, 2, 8.0, 12.0)
LOW = butter_sos(FS, 2, high_hz=5.0)
T = np.arange(int(40 * FS)) / FS


def envelope(x: np.ndarray, reference: str = "car") -> np.ndarray:
    return apply_sos(rectified_band(x, BAND, reference), LOW, causal=True)


def test_envelope_follows_a_power_decrease_on_the_right_channel():
    amp = np.where((T > 20) & (T < 30), 0.25, 1.0)
    x = np.zeros((3, T.size))
    x[0] = amp * np.sin(2 * np.pi * 10 * T)
    env = envelope(x, reference="orig")
    before, during = env[0, (T > 15) & (T < 20)].mean(), env[0, (T > 25) & (T < 30)].mean()
    assert during / before == pytest.approx(0.25, abs=0.03)
    assert before == pytest.approx(2 / np.pi, abs=0.05)  # mean of a rectified unit sine
    assert env[1].max() == 0


def test_envelope_is_causal():
    x = np.random.default_rng(0).standard_normal((3, T.size))
    y = x.copy()
    cut = int(30 * FS)
    y[:, cut:] += 50.0
    assert np.array_equal(envelope(x)[:, :cut], envelope(y)[:, :cut])


def test_car_is_applied_before_rectifying():
    common = np.tile(np.sin(2 * np.pi * 10 * T), (4, 1))  # the same signal on every channel
    assert envelope(common).max() < 1e-9
    assert envelope(common, reference="orig").max() > 0.5


def test_stacked_bands_keep_order_and_targets(tmp_path: Path):
    subj = tmp_path / "P1"
    subj.mkdir()
    names = [f"E{i}" for i in range(32)]
    names[:2] = ["Fp1", "Fp2"]
    trials = pd.DataFrame({"subject": ["P1"], "series": [1], "lift": [1], "n_samples": [100], "offset_start": [0], "offset_end": [100],
                           "onset_s": [0.5], "segment_start_s": [0.3], "segment_end_s": [0.9]})
    trials.to_csv(subj / "trials.csv", index=False)
    (subj / "manifest.json").write_text('{"fs_hz": 100, "eeg_names": ' + str(names).replace("'", '"') + "}", encoding="utf-8")
    np.save(subj / "kin_pos.npy", np.zeros((4, 3, 100), np.float32))
    np.save(subj / "wrist_vel.npy", np.zeros((3, 100), np.float32))
    for value, band in ((1.0, "a"), (2.0, "b")):
        np.save(subj / f"eeg_{band}_car.npy", np.full((32, 100), value, np.float32))
        np.save(subj / f"pre_lift_eeg_{band}_car.npy", np.full((32, 1, 5), value, np.float32))
    one = load_subject_samples(tmp_path, "P1", "a", "car", "prefrontal", "position", 5, pre_lift=True)
    both = load_subject_samples(tmp_path, "P1", ["a", "b"], "car", "prefrontal", "position", 5, pre_lift=True)
    assert both.eeg.shape == (4, one.eeg.shape[1])
    assert (both.eeg[:2] == 1).all() and (both.eeg[2:] == 2).all()
    assert np.array_equal(both.sample_idx, one.sample_idx) and np.array_equal(both.target, one.target)
    assert np.array_equal(both.eeg_stat_mask, one.eeg_stat_mask)


def subject(name: str, seed: int, fs: float = 100.0) -> SubjectSamples:
    rng = np.random.default_rng(seed)
    n, rows, targets = 300, [], []
    for i in range(8):
        on = 1.0 + rng.uniform(0.2, 0.6)
        rows.append(dict(subject=name, series=1, lift=i + 1, n_samples=n, offset_start=i * n, offset_end=(i + 1) * n, onset_s=on,
                         ev_LEDOn=1.0, segment_start_s=on - 0.5, segment_end_s=on + 1.5))
        tau = np.clip((np.arange(n) + 1) / fs - on, 0, None)
        targets.append(np.stack([tau, -2 * tau, tau ** 2]) + 0.1 * rng.standard_normal((3, n)))
    trials = pd.DataFrame(rows)
    idx, trow = segment_samples(trials, fs, 1)
    y = np.concatenate(targets, axis=1).astype(np.float32)
    return SubjectSamples(name, np.zeros((2, y.shape[1]), np.float32), y, idx, trow, trials, fs)


class Oracle(Baseline):
    def predict(self, s, mask, ctx, repeat=0):
        return s.target[:, s.sample_idx[mask]].T


def test_inner_validation_scores_val_subjects_with_the_training_template():
    train, val = [subject("P1", 1), subject("P2", 2)], [subject("P4", 3)]
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in [*train, *val]}
    ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0, train_masks=ones, val_masks=ones)
    out = score_inner_validation(Oracle(), train, val, ctx, seed=1, n_permutations=2)
    assert set(out) == {"P4"}
    assert out["P4"]["r"] == pytest.approx(1) and out["P4"]["residual_r"] == pytest.approx(1)
    assert out["P4"]["residual_r_specific"] == pytest.approx(out["P4"]["residual_r"] - out["P4"]["residual_r_other_trial"])
    assert out["P4"]["residual_r_specific"] > 0.5


def test_screening_grid_matches_the_decision():
    grid = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in Path("configs/diagnostics/screening").glob("*.yaml")}
    assert len(grid) == 14  # 11 configs, 2 prefrontal controls, and the broadband reference with inner-validation metrics
    env = ["mu_env", "beta_env"]
    expected = {("mrcp", "all"), (str(env), "all"), (str(["mrcp", *env]), "all"), ("broadband", "sensorimotor"),
                ("broadband", "contralateral"), ("broadband", "ipsilateral"), ("broadband", "no_prefrontal"),
                ("mrcp", "contralateral"), ("mrcp", "ipsilateral"), (str(env), "contralateral"), (str(env), "ipsilateral"),
                ("broadband", "prefrontal"), ("mrcp", "prefrontal"), ("broadband", "all")}
    assert {(str(c["data"]["band"]), c["data"]["channel_set"]) for c in grid.values()} == expected
    for c in grid.values():
        assert c["data"]["channel_set"] == "all" or c["data"]["channel_set"] in CHANNEL_SETS
        assert c["inner_val_scoring"] and c["data"]["reference"] == "car" and c["splits"]["mode"] == "loso"
        assert c["baseline"]["params"]["train"]["lr"] == 0.001 and c["baseline"]["name"] == "cnn_lstm"
