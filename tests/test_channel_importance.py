"""Tests for channel occlusion and Haufe activation patterns (synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eegtraj.baselines import FitContext
from eegtraj.baselines.base import Baseline
from eegtraj.baselines.ridge import haufe_pattern
from eegtraj.baselines.scoring import channel_occlusion, fit_cue_template, occluded
from eegtraj.data.windows import SubjectSamples, Standardizer, input_channel_names
from tests.test_screening import subject


class FirstChannel(Baseline):
    """Predicts the target from EEG row 0 only: target + 10 * (normalized row 0)."""

    def predict(self, s, mask, ctx, repeat=0):
        z = ctx.eeg_norm[s.subject].transform(s.eeg)[0, s.sample_idx[mask]]
        return s.target[:, s.sample_idx[mask]].T + 10 * z[:, None]


def with_eeg(s: SubjectSamples, seed: int) -> SubjectSamples:
    eeg = np.random.default_rng(seed).standard_normal((3, s.target.shape[1])).astype(np.float32) * 5 + 2
    return SubjectSamples(s.subject, eeg, s.target, s.sample_idx, s.trial_row, s.trials, s.fs_hz)


def context(subjects: list[SubjectSamples]) -> FitContext:
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subjects}
    return FitContext(eeg_norm={s.subject: Standardizer.fit([s.eeg]) for s in subjects}, target_norm=None, history=1,
                      lag_step=1, seed=0, train_masks=ones, val_masks=ones)


def test_occluded_rows_are_zero_after_normalization_and_others_unchanged():
    s = with_eeg(subject("P1", 1), 0)
    ctx = context([s])
    x = occluded(s, [1], ctx)
    z = ctx.eeg_norm["P1"].transform(x.eeg)
    assert np.allclose(z[1], 0, atol=1e-5)
    assert np.array_equal(x.eeg[[0, 2]], s.eeg[[0, 2]]) and not np.shares_memory(x.eeg, s.eeg)


def test_only_the_used_channel_and_its_groups_matter():
    train, test = [with_eeg(subject("P1", 1), 0), with_eeg(subject("P2", 2), 1)], with_eeg(subject("P4", 3), 2)
    ctx = context([*train, test])
    template = fit_cue_template(train, ctx)
    mask = np.ones(len(test.sample_idx), bool)
    rows = channel_occlusion(FirstChannel(), test, mask, ctx, template, ["Fp1", "C3", "O1"],
                             {"prefrontal": ["Fp1", "Fp2"], "occipital": ["O1"], "absent": ["T7"]}, seed=1, n_permutations=2)
    by = {r["occluded"]: r for r in rows}
    assert set(by) == {"none", "Fp1", "C3", "O1", "prefrontal", "occipital"}  # a group with no rows is skipped
    assert by["Fp1"]["r"] == pytest.approx(1) and by["prefrontal"]["r"] == pytest.approx(1)
    assert by["none"]["r"] < 0.5
    for name in ("C3", "O1", "occipital"):
        assert by[name]["r"] == pytest.approx(by["none"]["r"])
        assert by[name]["residual_r_specific"] == pytest.approx(by["none"]["residual_r_specific"])


def test_haufe_pattern_finds_the_source_not_the_distractor():
    rng = np.random.default_rng(0)
    source, distractor = rng.standard_normal(20000), rng.standard_normal(20000)
    x = np.stack([source + distractor, distractor], axis=1)  # the decoder must use channel 2 to cancel the distractor
    w = np.linalg.lstsq(x, source[:, None], rcond=None)[0]
    assert w[1, 0] == pytest.approx(-1, abs=0.02)
    pattern = haufe_pattern(np.cov(x.T), w)
    assert pattern[0, 0] == pytest.approx(1, abs=0.03) and abs(pattern[1, 0]) < 0.03


def test_input_channel_names_follow_channel_set_and_band_order(tmp_path: Path):
    names = ["Fp1", "Fp2", "C3", "C4"]
    (tmp_path / "P1").mkdir()
    (tmp_path / "P1" / "manifest.json").write_text(json.dumps({"eeg_names": names}), encoding="utf-8")
    assert input_channel_names(tmp_path, "P1", "mrcp", "all") == names
    assert input_channel_names(tmp_path, "P1", "mrcp", "prefrontal") == ["Fp1", "Fp2"]
    assert input_channel_names(tmp_path, "P1", ["a", "b"], "prefrontal") == ["a:Fp1", "a:Fp2", "b:Fp1", "b:Fp2"]
