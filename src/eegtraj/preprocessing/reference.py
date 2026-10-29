"""Re-referencing and named channel sets (protocol decision D5)."""

from __future__ import annotations

import numpy as np

# Defined from the 32 WAY-EEG-GAL channel names.
CHANNEL_SETS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "temporal": ("T7", "T8", "TP9", "TP10"),
    "sensorimotor": ("FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"),
}


def common_average(eeg: np.ndarray) -> np.ndarray:
    """Common average reference.

    Args:
        eeg: float array (n_channels, n_times).

    Returns:
        ``eeg`` minus its across-channel mean at every sample, same dtype.
    """
    return eeg - eeg.mean(axis=0, keepdims=True)


def channel_indices(names: list[str], set_name: str) -> list[int]:
    """Indices of a named channel set (``"all"``, or a key of :data:`CHANNEL_SETS`) in ``names``."""
    if set_name == "all":
        return list(range(len(names)))
    return [names.index(ch) for ch in CHANNEL_SETS[set_name]]
