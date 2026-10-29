"""Kinematic movement onset from wrist speed (protocol decision D2).

The labeled ``tHandStart`` is wrong in some lifts (the hand had already reached
the object), so movement onset is re-derived from the wrist trajectory:

1. Smooth the wrist position with a zero-phase low-pass (kinematics are targets,
   not decoder inputs, so zero-phase smoothing does not leak into EEG features).
2. Find the peak wrist speed between LED-on and first digit touch.
3. Onset = the last sample before that peak where speed is below
   ``threshold_frac`` x peak speed.
"""

from __future__ import annotations

import numpy as np
import scipy.signal


def smooth_kinematics(pos: np.ndarray, fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass along time.

    Args:
        pos: float array (n_times, n_dims), any position units.
        fs_hz: sampling rate of ``pos``.
        cutoff_hz: low-pass cutoff.
        order: Butterworth order (applied forward and backward, so the effective order doubles).

    Returns:
        Smoothed array with the same shape as ``pos``.
    """
    sos = scipy.signal.butter(order, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
    return scipy.signal.sosfiltfilt(sos, pos, axis=0)


def kinematic_onset(
    t_s: np.ndarray,
    wrist_pos: np.ndarray,
    fs_hz: float,
    t_search_start_s: float,
    t_search_end_s: float,
    threshold_frac: float,
    cutoff_hz: float,
    order: int,
) -> tuple[float, float, float]:
    """Movement onset from wrist speed.

    Args:
        t_s: float array (n_times,), sample times in seconds (window time).
        wrist_pos: float array (n_times, 3), wrist position (cm, lab frame).
        fs_hz: sampling rate.
        t_search_start_s: start of the peak search, e.g. LED-on (window time, s).
        t_search_end_s: end of the peak search, e.g. first digit touch (window time, s).
        threshold_frac: onset threshold as a fraction of the peak speed.
        cutoff_hz, order: smoothing low-pass parameters.

    Returns:
        ``(onset_s, peak_time_s, peak_speed)``. ``onset_s`` is NaN if speed never drops
        below the threshold before the peak.
    """
    smooth = smooth_kinematics(wrist_pos, fs_hz, cutoff_hz, order)
    speed = np.linalg.norm(np.gradient(smooth, t_s, axis=0), axis=1)
    in_search = np.flatnonzero((t_s >= t_search_start_s) & (t_s <= t_search_end_s))
    if in_search.size == 0:
        raise ValueError("empty search interval for the reach speed peak")
    i_peak = in_search[np.argmax(speed[in_search])]
    below = np.flatnonzero(speed[:i_peak] < threshold_frac * speed[i_peak])
    onset_s = float(t_s[below[-1]]) if below.size else float("nan")
    return onset_s, float(t_s[i_peak]), float(speed[i_peak])
