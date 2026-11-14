"""Tests for the physiological prior and the pretraining target view."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from eegtraj.data.target_view import prior_vector, target_view
from eegtraj.data.unlabeled import UnlabeledWindows

CACHE = Path("dataset/cache/98f5fa6e66")
RAW = Path("dataset/raw")
PRIOR = Path("configs/pretrain/physio_prior.yaml")


def names():
    return json.loads((CACHE / "P1" / "manifest.json").read_text(encoding="utf-8"))["eeg_names"] if CACHE.exists() else [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9",
        "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]


def test_prior_config_covers_every_channel_once_with_the_decided_values():
    cfg = yaml.safe_load(PRIOR.read_text(encoding="utf-8"))
    n = names()
    w = prior_vector(cfg, n)
    assert w.shape == (32,) and w.dtype == np.float32
    by = dict(zip(n, w.tolist()))
    assert by["Cz"] == pytest.approx(0.9) and by["C3"] == pytest.approx(0.9) and by["CP5"] == pytest.approx(0.9)
    assert by["C4"] == pytest.approx(0.6) and by["Oz"] == pytest.approx(0.6) and by["F7"] == pytest.approx(0.6)
    assert by["Fp1"] == pytest.approx(0.2) and by["Fp2"] == pytest.approx(0.2)
    assert (w > 0).all() and (w < 1).all()  # the gate is a sigmoid


def test_prior_rejects_missing_or_duplicated_channels():
    cfg = yaml.safe_load(PRIOR.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        prior_vector(cfg, names() + ["EOG"])
    dup = yaml.safe_load(PRIOR.read_text(encoding="utf-8"))
    dup["groups"]["prefrontal"]["channels"].append("Cz")
    with pytest.raises(ValueError):
        prior_vector(dup, names())


def test_target_view_guided_and_unguided():
    raw = torch.randn(4, 32, 200)
    cleaned = torch.randn(4, 32, 200)
    w = torch.linspace(0.1, 1.0, 32)
    assert torch.equal(target_view(raw, cleaned, w, guided=False), raw)
    out = target_view(raw, cleaned, w, guided=True)
    assert out.shape == raw.shape and torch.allclose(out[:, 5], cleaned[:, 5] * w[5])
    assert torch.equal(target_view(raw, raw, torch.ones(32), guided=True), raw)
    with pytest.raises(ValueError):
        target_view(raw, cleaned[..., :100], w, guided=True)


def test_raw_and_target_windows_share_shape_and_time_index():
    if not (CACHE / "P1" / "stream_eeg_broadband_ica_car.npy").exists():
        pytest.skip("P1 ICA stream not built")
    ds = UnlabeledWindows(CACHE, ["P1"], ["broadband_car", "broadband_ica_car"], window=200, stride=500)
    seg_raw = np.load(CACHE / "P1" / "stream_eeg_broadband_car.npy", mmap_mode="r")
    seg_ica = np.load(CACHE / "P1" / "stream_eeg_broadband_ica_car.npy", mmap_mode="r")
    assert seg_raw.shape == seg_ica.shape
    raw, ica = ds[len(ds) // 2]
    assert raw.shape == ica.shape == (32, 200)
    w = torch.from_numpy(prior_vector(yaml.safe_load(PRIOR.read_text(encoding="utf-8")), names()))
    assert target_view(raw, ica, w, guided=True).shape == raw.shape


@pytest.mark.skipif(not (RAW / "P1").exists() or not (CACHE / "P1" / "stream_eeg_broadband_car.npy").exists(),
                    reason="needs dataset/raw/P1 and the P1 raw stream")
def test_zero_components_and_unit_weights_reproduce_the_raw_window_exactly(tmp_path):
    """Real P1: the cleaned chain with an ICA that removes nothing, and w_phys = 1, gives the raw stream and windows."""
    from eegtraj.data.band_variant import with_band
    from eegtraj.data.unlabeled import build_stream

    src, dst = CACHE / "P1", tmp_path / "P1"
    dst.mkdir()
    for f in ("manifest.json", "trials.csv", "stream_segments.csv", "stream_eeg_broadband_car.npy"):
        shutil.copy(src / f, dst / f)
    shutil.copy(src / "eeg_broadband_car.npy", dst / "eeg_broadband_zero_car.npy")  # alignment reference
    z = dict(np.load(src / "ica_broadband_ica_car.npz"))
    z["selected"] = np.zeros(0, np.int64)
    np.savez(dst / "ica_zero.npz", **z)
    base = yaml.safe_load(Path("configs/cache/default.yaml").read_text(encoding="utf-8"))
    build_stream("P1", with_band(base, "broadband_zero", [0.1, 40.0]), RAW, tmp_path, "broadband_zero", "car", "zero")
    a = np.load(dst / "stream_eeg_broadband_car.npy")
    b = np.load(dst / "stream_eeg_broadband_zero_car.npy")
    assert np.array_equal(a, b)
    ds = UnlabeledWindows(tmp_path, ["P1"], ["broadband_car", "broadband_zero_car"], window=200, stride=200)
    for k in (0, len(ds) // 2, len(ds) - 1):
        raw, cleaned = ds[k]
        assert torch.equal(target_view(raw, cleaned, torch.ones(32), guided=True), raw)
