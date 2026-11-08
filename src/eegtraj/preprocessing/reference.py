"""Re-referencing and named channel sets (protocol decision D5)."""

from __future__ import annotations

import numpy as np

# Defined from the 32 WAY-EEG-GAL channel names.
CHANNEL_SETS: dict[str, tuple[str, ...]] = {
    "prefrontal": ("Fp1", "Fp2"),
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "temporal": ("T7", "T8", "TP9", "TP10"),
    "sensorimotor": ("FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"),
    # Band and channel screening. These overlap the sets above. "Contralateral" assumes right-hand movement.
    "contralateral": ("C3", "FC5", "FC1", "CP5", "CP1", "Cz"),
    "ipsilateral": ("C4", "FC6", "FC2", "CP6", "CP2", "Cz"),
    "no_prefrontal": ("F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9",
                      "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"),
    # Electrode reduction curve (final checks). The 6-channel contralateral montage is "contralateral" above.
    "central9": ("Cz", "FC1", "FC2", "C3", "C4", "CP1", "CP2", "FC5", "CP5"),
    "minimal3": ("Cz", "C3", "FC1"),
    "cz": ("Cz",),
    "headband_ring": ("Fp1", "Fp2", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"),
}

# Re-references applied after loading (see ``rereference_rows``). None keeps the cached reference.
LINKED_MASTOID = ("TP9", "TP10")
REREFERENCES = ("linked_mastoid",)


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


def rereference_rows(eeg: np.ndarray, names: list[str], rows: list[int], rereference: str | None) -> np.ndarray:
    """Rows ``rows`` of ``eeg`` (channels on axis 0, any trailing shape), optionally re-referenced.

    ``linked_mastoid`` subtracts the mean of TP9 and TP10. Any common reference of the input (CAR or the original
    online reference) cancels in that difference, so the result from CAR data equals the result from the original
    recording. Returns float32.
    """
    if rereference is None:
        return np.asarray(eeg[rows], dtype=np.float32)
    if rereference not in REREFERENCES:
        raise ValueError(f"unknown rereference {rereference!r}; expected one of {REREFERENCES}")
    ref = np.asarray(eeg[[names.index(c) for c in LINKED_MASTOID]], dtype=np.float64).mean(axis=0)
    return (np.asarray(eeg[rows], dtype=np.float64) - ref).astype(np.float32)
