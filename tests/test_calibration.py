"""Tests for calibration within subject: the k-lift template and the absence of S7-S9 in training."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from eegtraj.data.windows import Standardizer, SubjectSamples, segment_samples
from eegtraj.diagnostics.calibration import (
    ALL,
    Series,
    calibration_rows,
    context,
    delta_scale,
    rows_mask,
    scratch_model,
    series_mask,
    subject_template,
    train_residual,
)
from eegtraj.models.trainer import TrainConfig

HISTORY = 20
N = 120  # samples per lift
LOG = logging.getLogger("test")


def subject(seed: int = 0, lifts_per_series: int = 4) -> SubjectSamples:
    rng = np.random.default_rng(seed)
    rows, targets = [], []
    for series in range(1, 10):
        for lift in range(lifts_per_series):
            i = len(rows)
            on = 0.6 + rng.uniform(0, 0.2)
            rows.append(dict(subject="P1", series=series, lift=lift + 1, n_samples=N, offset_start=i * N,
                             offset_end=(i + 1) * N, onset_s=on, ev_LEDOn=0.3, segment_start_s=on - 0.3,
                             segment_end_s=on + 0.5))
            tau = np.clip((np.arange(N) + 1) / 100 - on, 0, None)
            targets.append(np.stack([tau, -2 * tau, tau ** 2]) * (1 + 0.3 * rng.standard_normal()))
    trials = pd.DataFrame(rows)
    idx, trow = segment_samples(trials, 100.0, HISTORY)
    y = np.concatenate(targets, axis=1).astype(np.float32)
    eeg = rng.standard_normal((4, y.shape[1])).astype(np.float32)
    return SubjectSamples("P1", eeg, y, idx, trow, trials, 100.0)


def test_calibration_rows_come_from_s1_to_s5_and_are_reproducible():
    s = subject()
    rows = calibration_rows(s, 6, draw=0, seed=1)
    assert len(rows) == 6 and set(s.trials.series.to_numpy()[rows]) <= {1, 2, 3, 4, 5}
    assert np.array_equal(rows, calibration_rows(s, 6, draw=0, seed=1))
    assert not np.array_equal(rows, calibration_rows(s, 6, draw=1, seed=1))
    assert np.array_equal(calibration_rows(s, ALL, draw=0, seed=1), np.flatnonzero(s.trials.series <= 5))


def test_subject_template_uses_only_the_k_calibration_lifts():
    s = subject()
    rows = calibration_rows(s, 5, draw=0, seed=1)
    cal = rows_mask(s, rows)
    ctx = context(s, Standardizer.fit([s.eeg]), cal, series_mask(s, Series().validation), HISTORY, 0, LOG)
    before = subject_template(s, cal, ctx)
    other = np.setdiff1d(np.arange(len(s.trials)), rows)  # every other lift, including S6-S9
    y = s.target.copy()
    for r in other:
        y[:, s.trials.offset_start[r]:s.trials.offset_end[r]] += 100.0
    after = subject_template(replace(s, target=y), cal, ctx)
    assert np.array_equal(before.template, after.template)
    y[:, s.trials.offset_start[rows[0]]:s.trials.offset_end[rows[0]]] += 100.0  # a calibration lift does count
    assert not np.array_equal(before.template, subject_template(replace(s, target=y), cal, ctx).template)


def test_no_s7_to_s9_sample_enters_training():
    s = subject()
    cal = rows_mask(s, calibration_rows(s, 8, draw=0, seed=1))
    val = series_mask(s, Series().validation)
    assert not (cal | val)[series_mask(s, Series().test)].any()
    eeg_norm = Standardizer.fit([s.eeg])  # label-free and shared by both runs, as in D9

    def fit(x: SubjectSamples):
        ctx = context(x, eeg_norm, cal, val, HISTORY, 7, LOG)
        template = subject_template(x, cal, ctx)
        norm = delta_scale(x, template, ctx)
        model = scratch_model("cnn_lstm", 4, HISTORY, dict(n_temporal=2, temporal_kernel=5, n_spatial=2, hidden=4), 7)
        cfg = TrainConfig(lr=1e-2, batch_size=64, max_epochs=3, samples_per_epoch=256, val_samples=128, patience=5,
                          amp=False)
        model, history = train_residual(model, x, template, norm, ctx, cfg, torch.device("cpu"))
        return model.state_dict(), history, norm.std

    first = int(s.trials.offset_start[s.trials.series == 7].min())  # S7-S9 are the last samples in time
    y, eeg = s.target.copy(), s.eeg.copy()
    y[:, first:] = 1e3
    eeg[:, first:] = -1e3
    a, b = fit(s), fit(replace(s, target=y, eeg=eeg))
    assert all(torch.equal(a[0][k], b[0][k]) for k in a[0])
    assert [h["val_loss"] for h in a[1]] == [h["val_loss"] for h in b[1]]
    assert np.array_equal(a[2], b[2])


def test_paired_comparison_averages_draws_and_counts_subjects():
    from eegtraj.diagnostics.calibration import paired, per_subject

    rows = []
    for i, subj in enumerate(["P1", "P2", "P4"]):
        rows.append(dict(subject=subj, condition="template", k="10", draw=0, r=0.8, r2=0.5, nrmse=0.2,
                         residual_r_specific=float("nan")))
        for draw, bump in enumerate([0.01, 0.03]):
            rows.append(dict(subject=subj, condition="finetune", k="10", draw=draw, r=0.8 + bump * (i + 1), r2=0.5,
                             nrmse=0.2 - bump, residual_r_specific=0.1))
    t = per_subject(pd.DataFrame(rows), "uncleaned")
    out = {r["metric"]: r for r in paired(t, ("uncleaned", "finetune"), ("uncleaned", "template"), "10")}
    assert out["r"]["diff"] == pytest.approx(0.02 * 2) and out["r"]["n_a_better"] == 3
    assert out["nrmse"]["n_a_better"] == 3 and out["r2"]["n_a_better"] == 0
    assert out["residual_r_specific"]["diff"] == pytest.approx(0.1)
