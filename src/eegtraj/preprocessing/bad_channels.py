"""Label-free bad-channel detection and spherical-spline interpolation (D5).

Detection uses only EEG amplitude, never labels, so it may run on any subject
(including test subjects) before splitting.
"""

from __future__ import annotations

import logging

import mne
import numpy as np

from eegtraj.preprocessing.filters import filter_signal

logger = logging.getLogger(__name__)

MONTAGE = "colin27_1020"  # contains all 32 WAY-EEG-GAL channels; replaces deprecated "standard_1020"


def amplitude_ratio(eeg: np.ndarray, fs_hz: float, highpass_hz: float, order: int) -> np.ndarray:
    """Robust amplitude per channel relative to the median channel.

    Args:
        eeg: float array (n_channels, n_times), one continuous recording.

    Returns:
        float array (n_channels,): MAD of the high-passed signal / median MAD over channels.
        The high-pass is zero-phase; this is an inspection statistic, not a decoder input.
    """
    x = filter_signal(eeg, fs_hz, order, low_hz=highpass_hz, causal=False)
    mad = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    return mad / np.median(mad)


def detect_bad_channels(ratios: np.ndarray, names: list[str], flat_ratio: float, noisy_ratio: float) -> list[str]:
    """Channels flagged in the majority of recordings.

    Args:
        ratios: float array (n_recordings, n_channels) from :func:`amplitude_ratio`.
        names: channel names, length n_channels.
        flat_ratio, noisy_ratio: a channel is flagged in one recording when its ratio is
            below ``flat_ratio`` or above ``noisy_ratio``.
    """
    flagged = (ratios < flat_ratio) | (ratios > noisy_ratio)
    majority = flagged.mean(axis=0) > 0.5
    return [n for n, bad in zip(names, majority) if bad]


def interpolate_bad_channels(eeg: np.ndarray, names: list[str], fs_hz: float, bads: list[str]) -> np.ndarray:
    """Replace ``bads`` by spherical-spline interpolation from the other channels.

    Uses fixed template electrode positions (:data:`MONTAGE`); nothing is fitted to data.

    Args:
        eeg: float array (n_channels, n_times).

    Returns:
        float array with the same shape; good channels are unchanged.
    """
    if not bads:
        return eeg
    info = mne.create_info(names, fs_hz, ch_types="eeg", verbose="error")
    raw = mne.io.RawArray(np.asarray(eeg, dtype=np.float64), info, verbose="error")
    raw.set_montage(mne.channels.make_standard_montage(MONTAGE), verbose="error")
    raw.info["bads"] = list(bads)
    raw.interpolate_bads(reset_bads=True, verbose="error")
    logger.info("Interpolated bad channels: %s", bads)
    return raw.get_data().astype(eeg.dtype)
