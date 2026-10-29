"""Butterworth filters, causal or zero-phase (protocol decision D4).

EEG decoder inputs use causal filters (``sosfilt``), so no future sample leaks into
a history window. Kinematic targets may use zero-phase filters (``sosfiltfilt``).

Causal filters are initialized with ``sosfilt_zi`` scaled by the first sample, which
removes the start-up step caused by DC offsets. The remaining transient of the
high-pass edge is measured with :func:`settle_time_s` and handled by flagging.
"""

from __future__ import annotations

import numpy as np
import scipy.signal


def butter_sos(fs_hz: float, order: int, low_hz: float | None = None, high_hz: float | None = None) -> np.ndarray:
    """Second-order sections for a Butterworth band-pass, high-pass, or low-pass.

    ``low_hz`` is the high-pass edge, ``high_hz`` the low-pass edge; give one or both.
    """
    if low_hz is not None and high_hz is not None:
        return scipy.signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")
    if low_hz is not None:
        return scipy.signal.butter(order, low_hz, btype="highpass", fs=fs_hz, output="sos")
    if high_hz is not None:
        return scipy.signal.butter(order, high_hz, btype="lowpass", fs=fs_hz, output="sos")
    raise ValueError("give low_hz and/or high_hz")


def apply_sos(x: np.ndarray, sos: np.ndarray, causal: bool, axis: int = -1) -> np.ndarray:
    """Filter ``x`` along ``axis``.

    Args:
        x: float array, any shape; time along ``axis``.
        sos: second-order sections from :func:`butter_sos`.
        causal: True for ``sosfilt`` initialized with ``sosfilt_zi`` x first sample;
            False for zero-phase ``sosfiltfilt`` (uses future samples).

    Returns:
        float64 array with the shape of ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    if not causal:
        return scipy.signal.sosfiltfilt(sos, x, axis=axis)
    xm = np.moveaxis(x, axis, -1)
    zi = scipy.signal.sosfilt_zi(sos)  # (n_sections, 2)
    zi = zi.reshape(zi.shape[0], *([1] * (xm.ndim - 1)), 2) * xm[..., :1][None]
    y, _ = scipy.signal.sosfilt(sos, xm, axis=-1, zi=zi)
    return np.moveaxis(y, -1, axis)


def filter_signal(
    x: np.ndarray,
    fs_hz: float,
    order: int,
    low_hz: float | None = None,
    high_hz: float | None = None,
    causal: bool = True,
    axis: int = -1,
) -> np.ndarray:
    """Butterworth band/high/low-pass of ``x`` along ``axis`` (see :func:`apply_sos`)."""
    return apply_sos(x, butter_sos(fs_hz, order, low_hz, high_hz), causal=causal, axis=axis)


def settle_time_s(
    x: np.ndarray,
    fs_hz: float,
    sos: np.ndarray,
    start_sample: int,
    tolerance: float,
) -> float:
    """Start-up transient length of a causal filter, measured on real data.

    Compares the causal filter started at ``start_sample`` (with ``sosfilt_zi``) against
    the same filter already running from sample 0. Returns the time after
    ``start_sample`` until the deviation stays below ``tolerance`` x SD of the
    reference output, for the worst channel.

    Args:
        x: float array (n_channels, n_times); needs ``start_sample`` plus at least the
            expected settle time of samples.
    """
    ref = apply_sos(x, sos, causal=True)[:, start_sample:]
    late = apply_sos(x[:, start_sample:], sos, causal=True)
    exceed = np.abs(late - ref) > tolerance * ref.std(axis=1, keepdims=True)
    last = np.flatnonzero(exceed.any(axis=0))
    return float((last[-1] + 1) / fs_hz) if last.size else 0.0
