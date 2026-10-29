"""Kinematic target preparation: smoothing, re-centering, velocity (D1, D4)."""

from __future__ import annotations

import numpy as np

from eegtraj.preprocessing.filters import filter_signal


def smooth_position(pos: np.ndarray, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase low-pass of positions along the last axis.

    Args:
        pos: float array (..., n_times), cm.

    Returns:
        float64 array with the same shape.
    """
    return filter_signal(pos, fs_hz, order, high_hz=cutoff_hz, causal=False, axis=-1)


def recenter(pos: np.ndarray, t_s: np.ndarray, t_ref_s: float) -> np.ndarray:
    """Subtract the position at ``t_ref_s`` (linear interpolation), per axis.

    Args:
        pos: float array (n_axes, n_times).
        t_s: float array (n_times,), sample times.
    """
    ref = np.array([np.interp(t_ref_s, t_s, p) for p in pos])
    return pos - ref[:, None]


def velocity(pos: np.ndarray, fs_hz: float) -> np.ndarray:
    """Central-difference derivative along the last axis: cm -> cm/s."""
    return np.gradient(pos, 1.0 / fs_hz, axis=-1)
