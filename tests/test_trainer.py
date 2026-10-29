"""Tests for eegtraj.models.trainer (CPU, synthetic)."""

from __future__ import annotations

import numpy as np
import torch

from eegtraj.baselines.deep import build_eegnet
from eegtraj.data.windows import Standardizer, window_at
from eegtraj.models.trainer import GpuWindows, TrainConfig, count_parameters, predict_windows, train_model
from tests.test_baselines import observed_subject

CPU = torch.device("cpu")
HISTORY = 4


def unit(n: int) -> Standardizer:
    return Standardizer(np.zeros((n, 1), np.float32), np.ones((n, 1), np.float32))


def make(names, seed=0):
    mix = np.random.default_rng(10).standard_normal((8, 6))
    subs = [observed_subject(n, mix, 0.3, seed + i) for i, n in enumerate(names)]
    masks = {s.subject: np.ones(len(s.sample_idx), bool) for s in subs}
    norms = {s.subject: unit(8) for s in subs}
    return subs, masks, norms


def test_gather_matches_window_at_across_subjects():
    subs, masks, norms = make(["P1", "P2"])
    w = GpuWindows(subs, masks, norms, unit(3), HISTORY, CPU)
    assert len(w) == sum(len(s.sample_idx) for s in subs)
    k = torch.tensor([0, 5, len(subs[0].sample_idx) + 3])
    x, y = w.batch(k)
    assert tuple(x.shape) == (3, 8, HISTORY)
    s2, i2 = subs[1], int(subs[1].sample_idx[3])
    assert np.allclose(x[2].numpy(), window_at(s2.eeg, i2, HISTORY))
    assert np.allclose(y[2].numpy(), s2.target[:, i2])
    assert np.allclose(x[0].numpy(), window_at(subs[0].eeg, int(subs[0].sample_idx[0]), HISTORY))


def test_training_reduces_loss_and_predict_order():
    torch.manual_seed(0)
    subs, masks, norms = make(["P1", "P2", "P4"])
    y_norm = Standardizer.fit([s.target[:, s.sample_idx] for s in subs[:2]])
    tr = GpuWindows(subs[:2], masks, norms, y_norm, HISTORY, CPU)
    va = GpuWindows(subs[2:], masks, norms, y_norm, HISTORY, CPU)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * HISTORY, 3))
    cfg = TrainConfig(lr=1e-2, batch_size=256, max_epochs=8, samples_per_epoch=4000, val_samples=2000,
                      patience=3, amp=False)
    model, hist = train_model(model, tr, va, cfg, seed=0)
    assert hist[-1]["val_loss"] < 0.5 * hist[0]["val_loss"] or min(h["val_loss"] for h in hist) < 0.2
    pred = predict_windows(model, va, cfg)
    x, _ = va.batch(torch.arange(len(va)))
    with torch.no_grad():
        assert np.allclose(pred, model(x).numpy(), atol=1e-5)  # same order as the windows


def test_eegnet_builds_for_window_and_regresses_three_outputs():
    m = build_eegnet(n_chans=32, n_times=50, kernel_length=50)
    out = m(torch.randn(2, 32, 50))
    assert tuple(out.shape) == (2, 3) and count_parameters(m) > 0


def test_shallowconvnet_builds_for_window_with_scaled_pooling():
    from eegtraj.baselines.deep import build_shallowconvnet

    m = build_shallowconvnet(n_chans=32, n_times=50, filter_time_length=10, pool_time_length=30, pool_time_stride=6)
    out = m(torch.randn(2, 32, 50))
    assert tuple(out.shape) == (2, 3) and torch.isfinite(out).all()


def test_cnn_lstm_shape_params_and_window_causality():
    from eegtraj.baselines.deep import build_cnn_lstm

    m = build_cnn_lstm(n_chans=32, n_times=50)
    m.eval()
    x = torch.randn(2, 32, 50)
    out = m(x)
    assert tuple(out.shape) == (2, 3)
    n = count_parameters(m)
    assert 1_000 < n < 60_000  # comparable to EEGNet / ShallowConvNet
    # Output depends on the whole (past-only) window, including its first sample.
    x2 = x.clone()
    x2[:, :, 0] += 10.0
    with torch.no_grad():
        assert not torch.allclose(m(x2), m(x))
