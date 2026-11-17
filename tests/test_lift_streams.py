"""Tests for the decoder's lift streams and batching (real cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eegtraj.data.lift_streams import aux_patch_targets, build_subject_streams
from eegtraj.data.windows import load_subject_samples
from eegtraj.models.decoder import aux_classes
from eegtraj.models.decoder_runner import collate, occluded_lifts

CACHE = Path("dataset/cache/98f5fa6e66")
pytestmark = pytest.mark.skipif(not (CACHE / "P1" / "pre_lift_eeg_broadband_car.npy").exists(),
                                reason="needs the main cache")


@pytest.fixture(scope="module")
def p1():
    keep = np.arange(294) < 30
    return build_subject_streams(CACHE, "P1", keep_trial=keep)


def test_stream_start_is_fixed_relative_to_the_cue_not_the_onset(p1):
    for l in p1.lifts:
        tr = p1.samples.trials.iloc[l.trial_row]
        assert l.j0 == round((tr.ev_LEDOn - 1.5) * 100)  # independent of onset_s
        assert l.out_idx.min() >= 19  # every scored sample after the first patch end
        assert l.out_idx.max() < 20 * l.n_patches
        assert l.eeg.shape == (32, 180 + 20 * l.n_patches) and l.eog.shape == (2, l.eeg.shape[1])
    onsets = p1.samples.trials.onset_s.iloc[[l.trial_row for l in p1.lifts]]
    assert onsets.std() > 0.02 and len({l.j0 for l in p1.lifts}) == 1


def test_stream_samples_match_the_cache_and_targets(p1):
    s = load_subject_samples(CACHE, "P1", "broadband", "car", "all", "position", history=300, pre_lift=True)
    st = s.eeg_for_stats().astype(np.float64)
    mean, sd = st.mean(axis=1), st.std(axis=1)
    for l in p1.lifts[:5]:
        tr = s.trials.iloc[l.trial_row]
        a = tr.offset_start + l.j0 - 180
        n_in = min(l.eeg.shape[1], tr.offset_end - a)
        expect = (s.eeg[:, a:a + n_in] - mean[:, None]) / (sd[:, None] + 1e-8)
        assert np.allclose(l.eeg[:, :n_in], expect, atol=1e-3)
        assert not l.eeg[:, n_in:].any()  # after the lift window: zeros
        j = s.sample_idx[l.sample_pos] - tr.offset_start
        assert np.array_equal(j - l.j0, l.out_idx)
        assert np.allclose(l.y, s.target[:, s.sample_idx[l.sample_pos]].T)
        assert np.allclose(l.t_since_led, (j + 1) / 100 - tr.ev_LEDOn, atol=1e-5)


def test_aux_targets_follow_the_cached_phase_labels(p1):
    aux = aux_classes(np.load(CACHE / "P1" / "phase.npy"))
    for l in p1.lifts[:5]:
        t = aux_patch_targets(l, aux)
        assert t.shape == (l.n_patches,) and set(np.unique(t)) <= {-1, 0, 1, 2, 3, 4}
        assert t[0] == 0  # before the cue: pre-onset
        nondec = t[t >= 0]
        assert (np.diff(nondec) >= 0).all()  # phases advance in order


def test_collate_pads_and_masks(p1):
    lifts = p1.lifts[:3]
    cue = {(l.subject, l.trial_row): np.ones((len(l.out_idx), 3), np.float32) for l in lifts}
    b = collate(lifts, cue, None, None, 180, 20)
    s_max = max(len(l.out_idx) for l in lifts)
    assert b["y"].shape == (3, s_max, 3) and b["valid"].sum() == sum(len(l.out_idx) for l in lifts)
    for i, l in enumerate(lifts):
        assert torch.equal(b["out_idx"][i, :len(l.out_idx)], torch.from_numpy(l.out_idx))
        assert not b["valid"][i, len(l.out_idx):].any()
    assert (b["aux"] == -1).all()


def test_occlusion_zeroes_neural_input_only(p1):
    lifts = p1.lifts[:3]
    occ = occluded_lifts(lifts, [0, 1])
    for a, b in zip(lifts, occ):
        assert not b.eeg[[0, 1]].any() and np.array_equal(b.eeg[2:], a.eeg[2:])
        assert np.array_equal(b.eog, a.eog) and np.array_equal(b.y, a.y)
        assert a.eeg[[0, 1]].any()  # original untouched
