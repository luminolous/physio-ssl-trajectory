"""Synthetic tests for eegtraj.preprocessing (filters, resampling, reference, bad channels, kinematics)."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from eegtraj.preprocessing.bad_channels import MONTAGE, amplitude_ratio, detect_bad_channels, interpolate_bad_channels
from eegtraj.preprocessing.filters import butter_sos, filter_signal, settle_time_s
from eegtraj.preprocessing.kinematics import recenter, velocity
from eegtraj.preprocessing.reference import CHANNEL_SETS, channel_indices, common_average
from eegtraj.preprocessing.resample import decimate, decimation_indices

FS = 500.0
NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
         "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]
SETTLE_S = 30.0  # skip this much at the start before measuring steady-state gain


def sine(freq_hz: float, duration_s: float, fs: float = FS) -> np.ndarray:
    t = np.arange(1, int(duration_s * fs) + 1) / fs
    return np.sin(2 * np.pi * freq_hz * t)


def steady_gain(y: np.ndarray, x: np.ndarray, fs: float = FS) -> float:
    s = int(SETTLE_S * fs)
    return float(np.std(y[s:]) / np.std(x[s:]))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("band, f_pass, f_stop", [((0.1, 4.0), 1.0, 20.0), ((0.1, 40.0), 10.0, 120.0)])
def test_bandpass_passband_and_stopband(causal, band, f_pass, f_stop):
    for f, check in ((f_pass, lambda g: 0.9 < g < 1.1), (f_stop, lambda g: g < 0.1)):
        x = sine(f, 60.0)
        g = steady_gain(filter_signal(x, FS, 2, *band, causal=causal), x)
        assert check(g), (band, f, g)


def test_causal_filter_ignores_future_but_zero_phase_does_not():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    x2 = x.copy()
    x2[3000:] += 100.0  # change only the future
    for causal, same in ((True, True), (False, False)):
        y1 = filter_signal(x, FS, 2, 0.1, 40.0, causal=causal)
        y2 = filter_signal(x2, FS, 2, 0.1, 40.0, causal=causal)
        assert np.allclose(y1[:3000], y2[:3000]) == same


def test_causal_zi_removes_dc_step():
    x = 500.0 + sine(1.0, 20.0)  # large DC offset as in the raw EEG
    y = filter_signal(x, FS, 2, 0.1, 4.0, causal=True)
    # Without zi the 500-unit step would ring at hundreds of units; with zi the output stays near the sine.
    assert np.abs(y).max() < 2.0


def test_settle_time_is_measured_and_shorter_for_higher_cutoff():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, int(200 * FS)))
    t_low = settle_time_s(x, FS, butter_sos(FS, 2, 0.1, 4.0), int(100 * FS), 0.01)
    t_high = settle_time_s(x, FS, butter_sos(FS, 2, 1.0, 4.0), int(100 * FS), 0.01)
    assert 0 < t_high < t_low < 60


def test_decimate_keeps_low_rejects_high_and_grid():
    # 70 Hz is above the 50 Hz output Nyquist; without anti-aliasing it would fold to 30 Hz.
    x = sine(2.0, 60.0) + sine(70.0, 60.0)
    y, fs_out = decimate(x, FS, 5, antialias_order=8, antialias_frac=0.8, causal=False)
    assert fs_out == 100.0
    ref = sine(2.0, 60.0, fs=100.0)
    s = int(SETTLE_S * 100)
    assert np.corrcoef(y[s:], ref[s:])[0, 1] > 0.999  # 2 Hz kept on the aligned grid
    assert np.std(y[s:] - ref[s:]) < 0.02  # no 30 Hz alias
    idx = decimation_indices(len(x), 5)
    t_in = (idx + 1) / FS
    assert np.allclose(t_in, np.arange(1, len(idx) + 1) / 100.0)


@pytest.mark.parametrize("lag_ms", [0, 150, 300])
def test_known_lag_preserved_through_decimation(lag_ms):
    # "EEG" is the kinematic signal delayed by lag_ms; both are decimated the same way.
    rng = np.random.default_rng(2)
    n = int(40 * FS)
    kin = filter_signal(rng.standard_normal(n), FS, 4, high_hz=3.0, causal=False)
    lag = int(lag_ms * FS / 1000)
    eeg = np.concatenate([np.zeros(lag), kin[: n - lag]])
    k_ds, fs_out = decimate(kin, FS, 5, 8, 0.8, causal=False)
    e_ds, _ = decimate(eeg, FS, 5, 8, 0.8, causal=False)
    lags = np.arange(0, 50)
    s = 200
    r = [np.corrcoef(e_ds[s + L:], k_ds[s:len(k_ds) - L])[0, 1] for L in lags]
    assert lags[int(np.argmax(r))] == lag_ms * fs_out / 1000


def test_common_average_and_channel_sets():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 100)) + 7.0
    y = common_average(x)
    assert np.allclose(y.mean(axis=0), 0.0)
    all_set = set()
    for name, chans in CHANNEL_SETS.items():
        assert set(chans) <= set(NAMES), name
        if name != "prefrontal":  # a deliberate subset of frontal, checked below
            assert not (set(chans) & all_set), "channel sets must be disjoint"
            all_set |= set(chans)
        assert [NAMES[i] for i in channel_indices(NAMES, name)] == list(chans)
    assert set(CHANNEL_SETS["prefrontal"]) < set(CHANNEL_SETS["frontal"])
    assert channel_indices(NAMES, "all") == list(range(32))


def test_bad_channel_detection_flags_flat_channel():
    rng = np.random.default_rng(4)
    ratios = []
    for _ in range(5):
        x = rng.standard_normal((32, int(20 * FS))) * 50
        x[NAMES.index("Fz")] *= 0.05
        ratios.append(amplitude_ratio(x, FS, 1.0, 4))
    assert detect_bad_channels(np.array(ratios), NAMES, 0.1, 5.0) == ["Fz"]


def test_interpolation_recovers_smooth_field():
    # A spatially smooth field (linear in electrode position) is recoverable by spherical splines.
    pos = mne.channels.make_standard_montage(MONTAGE).get_positions()["ch_pos"]
    xyz = np.array([pos[n] for n in NAMES])
    rng = np.random.default_rng(5)
    weights = rng.standard_normal((3, 200))
    field = xyz @ weights  # (32, 200)
    corrupted = field.copy()
    corrupted[NAMES.index("C3")] = rng.standard_normal(200)
    out = interpolate_bad_channels(corrupted, NAMES, FS, ["C3"])
    keep = [i for i, n in enumerate(NAMES) if n != "C3"]
    assert np.array_equal(out[keep], corrupted[keep])
    assert np.corrcoef(out[NAMES.index("C3")], field[NAMES.index("C3")])[0, 1] > 0.95


def test_recenter_and_velocity():
    t = np.arange(1, 501) / FS
    pos = np.stack([3.0 * t + 1.0, -2.0 * t, np.full_like(t, 5.0)])
    rc = recenter(pos, t, t_ref_s=0.5)
    assert np.allclose(rc[:, np.argmin(np.abs(t - 0.5))], 0.0, atol=1e-9)
    v = velocity(pos, FS)
    assert np.allclose(v[:, 1:-1], np.array([[3.0], [-2.0], [0.0]]))
