"""Integer-factor downsampling that keeps EEG and kinematics on one time grid (D3).

Raw samples have timestamps ``(k + 1) / fs`` (k = 0-based index). Keeping indices
``k = factor - 1, 2 * factor - 1, ...`` gives output timestamps ``(j + 1) / fs_out``,
so every stream decimated with the same factor stays sample-aligned.
"""

from __future__ import annotations

import numpy as np

from eegtraj.preprocessing.filters import filter_signal


def decimation_indices(n_times: int, factor: int) -> np.ndarray:
    """Input sample indices kept by :func:`decimate` (0-based)."""
    return np.arange(factor - 1, n_times, factor)


def decimate(
    x: np.ndarray,
    fs_hz: float,
    factor: int,
    antialias_order: int,
    antialias_frac: float,
    causal: bool,
    axis: int = -1,
) -> tuple[np.ndarray, float]:
    """Anti-alias low-pass, then keep every ``factor``-th sample.

    Args:
        x: float array; time along ``axis``.
        fs_hz: input rate.
        factor: integer decimation factor.
        antialias_order: Butterworth order of the anti-alias low-pass.
        antialias_frac: cutoff as a fraction of the output Nyquist frequency.
        causal: causal anti-alias filter (EEG) or zero-phase (kinematics).

    Returns:
        ``(y, fs_out_hz)``, with y as float32.
    """
    fs_out = fs_hz / factor
    y = filter_signal(x, fs_hz, antialias_order, high_hz=antialias_frac * fs_out / 2, causal=causal, axis=axis)
    idx = decimation_indices(y.shape[axis], factor)
    return np.take(y, idx, axis=axis).astype(np.float32), fs_out
