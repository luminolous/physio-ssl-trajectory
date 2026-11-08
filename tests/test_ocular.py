"""Tests for ICA ocular cleaning (synthetic data)."""

from __future__ import annotations

import numpy as np

from eegtraj.preprocessing.ocular import OcularICA, eog_proxies, fit_ocular_ica, remove_components

NAMES = ["Fp1", "Fp2", "F7", "F8", "C3", "C4", "Cz", "O1"]


def recording(seed: int = 0, n: int = 20000):
    """Brain sources everywhere plus a blink source strong at Fp1/Fp2 and a saccade source on F7 vs F8."""
    rng = np.random.default_rng(seed)
    brain = rng.laplace(size=(6, n))
    blink = np.zeros(n)
    for t in rng.integers(0, n - 50, 40):
        blink[t:t + 50] += np.hanning(50) * 60  # eye activity dominates Fp, as in the data
    saccade = np.repeat(rng.choice([-1.0, 0.0, 1.0], n // 200 + 1), 200)[:n] * 20
    eye_map = np.zeros((8, 2))
    eye_map[[0, 1], 0] = [1.0, 0.9]
    eye_map[[2, 3], 1] = [1.0, -1.0]
    eye_map[[4, 5, 6, 7], 0] = 0.05
    x = rng.standard_normal((8, 6)) @ brain + eye_map @ np.stack([blink, saccade])
    return x, brain, np.stack([blink, saccade])


def test_nothing_selected_returns_the_input_unchanged():
    x, _, _ = recording()
    empty = OcularICA(np.eye(8), np.eye(8), [], {}, 8)
    assert remove_components(x, empty) is x
    assert remove_components(x, None) is x
    ica = fit_ocular_ica(x, NAMES, threshold=1.01, max_components=4, random_state=0)  # nothing can pass
    assert ica.selected == [] and ica.removed_variance == 0.0
    assert np.array_equal(remove_components(x, ica), x)


def test_ica_removes_the_eye_sources_and_keeps_the_brain():
    x, brain, eye = recording()
    ica = fit_ocular_ica(x, NAMES, threshold=0.4, max_components=4, random_state=0)
    clean = remove_components(x, ica)
    assert 1 <= len(ica.selected) <= 2
    proxies = eog_proxies(clean, NAMES)
    assert abs(np.corrcoef(proxies["VEOG"], eye[0])[0, 1]) < 0.1
    assert abs(np.corrcoef(proxies["HEOG"], eye[1])[0, 1]) < 0.1
    # The brain part of every channel survives: regress clean on brain sources.
    coef, res, *_ = np.linalg.lstsq(brain.T, clean.T, rcond=None)
    assert (res / (clean ** 2).sum(axis=1)).max() < 0.05


def test_rank_deficient_data_uses_the_rank():
    x, _, _ = recording(1)
    x = np.vstack([x[:7], x[6:7] * 0.5 + x[5:6] * 0.5])  # last channel interpolated from two others
    ica = fit_ocular_ica(x, NAMES, threshold=0.4, max_components=4, random_state=0)
    assert ica.rank == 7 and ica.mixing.shape == (8, 7)


RAW_P1 = __import__("pathlib").Path("dataset/raw/P1/HS_P1_S1.mat")
CACHE_P1 = __import__("pathlib").Path("dataset/cache/98f5fa6e66/P1/eeg_broadband_car.npy")


@__import__("pytest").mark.skipif(not (RAW_P1.exists() and CACHE_P1.exists()), reason="needs raw data and the main cache")
def test_zero_components_reproduce_the_uncleaned_cache_exactly():
    from pathlib import Path

    import yaml

    from eegtraj.data.band_variant import compute_band_variant

    cfg = yaml.safe_load(Path("configs/cache/default.yaml").read_text(encoding="utf-8"))
    cache = CACHE_P1.parent.parent
    empty = OcularICA(np.eye(32), np.eye(32), [], {}, 32)
    eeg, pre, _, _ = compute_band_variant("P1", cfg, Path(cfg["paths"]["raw_dir"]), cache, "broadband", "car", 200, ica=empty)
    assert np.array_equal(eeg, np.load(CACHE_P1))
    assert np.array_equal(pre, np.load(cache / "P1" / "pre_lift_eeg_broadband_car.npy")[:, :, -200:])
