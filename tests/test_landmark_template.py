"""Tests for landmark-registered templates (synthetic data, plus a real-data round trip if the cache exists)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from eegtraj.data.splits import Fold
from eegtraj.data.windows import SubjectSamples, load_subject_samples
from eegtraj.models.landmark_template import (LANDMARK_EVENTS, TemplateBank, fit_landmark_templates, fold_templates,
                                              landmark_fractions, lift_landmarks, phase_to_time, register_lift,
                                              time_to_phase)

FS = 100.0
CACHE = Path("dataset/cache/98f5fa6e66")


def shape_of(phi: np.ndarray, scale: float) -> np.ndarray:
    """(3, n) smooth trajectory of phase, 0 at phi = 0."""
    return scale * np.stack([np.sin(np.pi * phi), phi ** 2, np.sin(2 * np.pi * phi) * 0.5])


def fake_subject(name: str, n_lifts: int, scale: float, rng: np.random.Generator, conditions=(165, 330, 660)) -> SubjectSamples:
    rows, parts, offset = [], [], 0
    for i in range(n_lifts):
        n = 800
        led = 2.0
        onset = led + rng.uniform(0.3, 0.6)
        dur = rng.uniform(2.0, 3.0)
        frac = np.array([0.0, 0.25, 0.35, 0.7, 1.0]) + np.r_[0, rng.uniform(-0.03, 0.03, 3), 0]
        marks = onset + frac * dur
        t = (np.arange(n) + 1) / FS
        phi = np.interp(t, marks, [0.0, 0.25, 0.35, 0.7, 1.0])  # true phase on the reference fractions
        y = shape_of(phi, scale)
        parts.append(y - np.array([np.interp(onset, t, y[a]) for a in range(3)])[:, None])  # re-centered as D1
        rows.append({"subject": name, "series": 1, "lift": i + 1, "weight_g": conditions[i % len(conditions)],
                     "ev_LEDOn": led, "offset_start": offset, "offset_end": offset + n,
                     **dict(zip(LANDMARK_EVENTS, marks))})
        offset += n
    target = np.concatenate(parts, axis=1).astype(np.float32)
    trials = pd.DataFrame(rows)
    return SubjectSamples(name, np.zeros((2, offset), np.float32), target, np.arange(offset), np.zeros(offset, np.int32),
                          trials, FS)


def test_phase_map_round_trip_and_bounds():
    marks = np.array([2.4, 3.0, 3.3, 4.5, 5.2])
    frac = np.array([0.0, 0.2, 0.3, 0.75, 1.0])
    t = np.linspace(2.4, 5.2, 50)
    assert np.allclose(phase_to_time(time_to_phase(t, marks, frac), marks, frac), t)
    assert time_to_phase(np.array([0.0, 9.0]), marks, frac).tolist() == [0.0, 1.0]  # before onset 0, after stop 1
    assert np.allclose(time_to_phase(marks, marks, frac), frac)


def test_registering_a_lift_and_mapping_back_reproduces_it():
    rng = np.random.default_rng(0)
    s = fake_subject("P1", 3, 10.0, rng)
    m = lift_landmarks(s.trials)
    frac = np.median(landmark_fractions(m), axis=0)
    grid = np.linspace(0, 1, 401)
    for row in range(3):
        a, b = int(s.trials.offset_start[row]), int(s.trials.offset_end[row])
        y = s.target[:, a:b]
        reg = register_lift(y, FS, m[row], frac, grid)
        t = (np.arange(b - a) + 1) / FS
        inside = (t >= m[row, 0]) & (t <= m[row, -1])
        phi = time_to_phase(t[inside], m[row], frac)
        back = np.stack([np.interp(phi, grid, reg[k]) for k in range(3)])
        assert np.abs(back - y[:, inside]).max() < 0.02  # cm, on a 10 cm trajectory


def test_test_subject_lifts_never_enter_the_templates():
    rng = np.random.default_rng(1)
    subjects = [fake_subject(f"P{i}", 6, 10.0, rng) for i in (1, 2, 4)] + [fake_subject("P5", 6, 1000.0, rng)]
    fold = Fold(name="loso_P5", train_subjects=("P1", "P2"), val_subjects=("P4",), test_subjects=("P5",))
    out = fold_templates(subjects, fold, n_grid=101)
    assert out.subjects == ("P1", "P2")
    assert np.abs(out.values).max() < 11.0  # the 1000x test subject (and the val subject) did not contribute
    alone = fit_landmark_templates([subjects[0], subjects[1]], ("P1", "P2"), n_grid=101)
    assert np.array_equal(out.values, alone.values)
    assert sum(out.counts.values()) == 12
    with pytest.raises(RuntimeError):
        bad = Fold(name="bad", train_subjects=("P1", "P5"), val_subjects=(), test_subjects=("P5",))
        fold_templates(subjects, bad, n_grid=101)


def test_templates_start_at_rest_and_recover_the_shape_per_condition():
    rng = np.random.default_rng(2)
    subjects = [fake_subject(f"P{i}", 9, 10.0, rng) for i in (1, 2)]
    out = fit_landmark_templates(subjects, ("P1", "P2"), n_grid=201)
    assert out.values.shape == (3, 3, 201)
    assert np.abs(out.values[:, :, 0]).max() < 1e-6  # T(0) = rest position at onset
    ref = shape_of(out.grid, 10.0)
    assert np.abs(out.values - ref[None]).max() < 0.5  # small blur from jittered landmarks
    assert out.fractions[0] == 0.0 and out.fractions[-1] == 1.0
    assert np.allclose(out(np.array([-1.0, 2.0]), 330), out.values[1][:, [0, -1]].T)


def test_torch_bank_matches_numpy_and_is_differentiable():
    rng = np.random.default_rng(3)
    out = fit_landmark_templates([fake_subject("P1", 9, 10.0, rng)], ("P1",), n_grid=201)
    bank = TemplateBank.from_templates(out)
    phi = torch.tensor([0.0, 0.123, 0.5, 0.999, 1.0, 1.3], dtype=torch.float32, requires_grad=True)
    all_c = bank(phi)
    assert all_c.shape == (6, 3, 3)
    for c, cond in enumerate(out.conditions):
        assert np.allclose(all_c[:, c].detach().numpy(), out(phi.detach().numpy(), cond), atol=1e-4)
    w = torch.softmax(torch.randn(6, 3, requires_grad=True), dim=-1)
    mix = bank(phi, w)
    assert mix.shape == (6, 3)
    assert torch.allclose(mix, (w.unsqueeze(-1) * all_c).sum(1), atol=1e-6)
    mix.sum().backward()
    assert phi.grad is not None and torch.isfinite(phi.grad).all() and phi.grad[1:4].abs().sum() > 0


@pytest.mark.skipif(not (CACHE / "P1" / "trials.csv").exists(), reason="needs the main cache")
def test_real_lifts_round_trip_through_their_own_landmarks():
    s = load_subject_samples(CACHE, "P1", "broadband", "car", "all", "position", history=1)
    m = lift_landmarks(s.trials)
    frac = np.median(landmark_fractions(m), axis=0)
    grid = np.linspace(0, 1, 201)
    errs = []
    for row in range(0, len(s.trials), 7):
        tr = s.trials.iloc[row]
        y = s.target[:, int(tr.offset_start):int(tr.offset_end)]
        reg = register_lift(y, s.fs_hz, m[row], frac, grid)
        t = (np.arange(y.shape[1]) + 1) / s.fs_hz
        inside = (t >= m[row, 0]) & (t <= m[row, -1])
        back = np.stack([np.interp(time_to_phase(t[inside], m[row], frac), grid, reg[k]) for k in range(3)])
        errs.append(np.abs(back - y[:, inside]).max())
    assert max(errs) < 0.2  # cm
