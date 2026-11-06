"""Tests for the residual formulation of the deep decoders (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from eegtraj.baselines import FitContext, make_baseline
from eegtraj.baselines.deep import residual_subject
from eegtraj.baselines.template import TemplateCueBaseline
from eegtraj.data.windows import Standardizer, SubjectSamples, segment_samples

FS = 100.0
N = 400
HISTORY = 20
TRAIN_CFG = dict(max_epochs=1, samples_per_epoch=256, val_samples=128, batch_size=64, amp=False)
MODEL = dict(n_temporal=2, temporal_kernel=5, n_spatial=2, pool=2, hidden=4, lstm_layers=1, drop_prob=0.0)


def subject(name: str, scale: float, seed: int) -> SubjectSamples:
    rng = np.random.default_rng(seed)
    rows, targets = [], []
    for i, rt in enumerate(rng.uniform(0.2, 0.6, 6)):
        on = 1.0 + rt
        rows.append(dict(subject=name, series=1, lift=i + 1, n_samples=N, offset_start=i * N, offset_end=(i + 1) * N,
                         onset_s=on, ev_LEDOn=1.0, segment_start_s=on - 0.5, segment_end_s=on + 2.0))
        tau = np.clip((np.arange(N) + 1) / FS - on, 0, None)
        targets.append(scale * np.stack([tau, -2 * tau, np.sin(tau)]))
    trials = pd.DataFrame(rows)
    idx, trow = segment_samples(trials, FS, HISTORY)
    y = np.concatenate(targets, axis=1).astype(np.float32)
    return SubjectSamples(name, rng.standard_normal((2, y.shape[1])).astype(np.float32), y, idx, trow, trials, FS)


def context(subjects: list[SubjectSamples]) -> FitContext:
    unit = Standardizer(np.zeros((2, 1), np.float32), np.ones((2, 1), np.float32))
    target = Standardizer(np.full((3, 1), 2.0, np.float32), np.full((3, 1), 5.0, np.float32))
    ones = {s.subject: np.ones(len(s.sample_idx), bool) for s in subjects}
    return FitContext(eeg_norm={s.subject: unit for s in subjects}, target_norm=target, history=HISTORY, lag_step=2, seed=0,
                      train_masks=ones, val_masks=ones)


class Constant(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 3), self.value, device=x.device)


def fitted():
    train, val, test = [subject("P1", 1.0, 1), subject("P2", 1.5, 2)], [subject("P4", 1.2, 3)], subject("P5", 0.7, 4)
    ctx = context([*train, *val, test])
    model = make_baseline("cnn_lstm", {"model": MODEL, "train": TRAIN_CFG, "residual": True})
    model.fit(train, val, ctx)
    floor = TemplateCueBaseline()
    floor.fit(train, val, ctx)
    return model, floor, test, ctx, np.ones(len(test.sample_idx), bool)


def test_zero_delta_reproduces_template_cue_exactly():
    model, floor, test, ctx, mask = fitted()
    model.model = Constant(0.0)
    assert np.array_equal(model.predict(test, mask, ctx), floor.predict(test, mask, ctx))


def test_constant_delta_is_scaled_by_the_training_sd_of_delta():
    model, floor, test, ctx, mask = fitted()
    model.model = Constant(1.0)
    expected = floor.predict(test, mask, ctx) + model.output_norm.std.ravel()
    assert np.allclose(model.predict(test, mask, ctx), expected)


def test_perfect_delta_recovers_the_target():
    _, floor, test, ctx, mask = fitted()
    delta = residual_subject(test, floor, ctx).target[:, test.sample_idx].T
    assert np.allclose(floor.predict(test, mask, ctx) + delta, test.target[:, test.sample_idx].T)


def test_template_comes_from_training_subjects_only():
    model, floor, _, _, _ = fitted()
    assert np.array_equal(model.template.template, floor.template)
    train, val = [subject("P1", 1.0, 1), subject("P2", 1.5, 2)], [subject("P4", 50.0, 3)]
    other = TemplateCueBaseline()
    other.fit(train, val, context([*train, *val]))
    assert np.array_equal(other.template, floor.template)


def test_delta_scale_ignores_validation_subjects():
    train, val = [subject("P1", 1.0, 1), subject("P2", 1.5, 2)], [subject("P4", 50.0, 3)]
    model = make_baseline("cnn_lstm", {"model": MODEL, "train": TRAIN_CFG, "residual": True})
    model.fit(train, val, context([*train, *val]))
    assert np.array_equal(model.output_norm.std, fitted()[0].output_norm.std)
    assert (model.output_norm.mean == 0).all()


def test_direct_prediction_still_uses_the_target_normalizer():
    train, val, test = [subject("P1", 1.0, 1)], [subject("P4", 1.2, 3)], subject("P5", 0.7, 4)
    ctx = context([*train, *val, test])
    model = make_baseline("cnn_lstm", {"model": MODEL, "train": TRAIN_CFG})
    model.fit(train, val, ctx)
    model.model = Constant(0.0)
    out = model.predict(test, np.ones(len(test.sample_idx), bool), ctx)
    assert (out == 2.0).all()  # z = 0 maps to the target mean
