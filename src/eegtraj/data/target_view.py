"""Physiological prior and the pretraining target view (proposed-model, design.md "Stage 1").

Context view: the raw window X (broadband CAR, D9 z-score). Target view:
- guided: the same window from the ICA-cleaned stream (its own D9 z-score), multiplied per channel by w_phys;
- unguided: the raw window itself.

w_phys and the gate prior g_prior are one vector built from ``configs/pretrain/physio_prior.yaml``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

LEVELS = ("high", "medium", "low")


def prior_vector(cfg: dict[str, Any], names: Sequence[str]) -> np.ndarray:
    """float32 (n_channels,) prior weight per channel, in the order of ``names``.

    Raises unless every channel of ``names`` appears in exactly one group and every group channel is in ``names``.
    """
    values = cfg["values"]
    if set(values) != set(LEVELS):
        raise ValueError(f"values must define exactly {LEVELS}")
    weight: dict[str, float] = {}
    for group, spec in cfg["groups"].items():
        for ch in spec["channels"]:
            if ch in weight:
                raise ValueError(f"{ch} is in more than one group ({group})")
            weight[ch] = float(values[spec["level"]])
    missing, extra = set(names) - set(weight), set(weight) - set(names)
    if missing or extra:
        raise ValueError(f"prior does not match the channels: missing {sorted(missing)}, unknown {sorted(extra)}")
    return np.array([weight[n] for n in names], dtype=np.float32)


def target_view(raw, cleaned, w_phys, guided: bool):
    """Target-view window(s).

    Args:
        raw, cleaned: z-scored windows (..., n_channels, n_times), numpy arrays or torch tensors, aligned sample by
            sample. w_phys: (n_channels,), same array type.
        guided: True for cleaned x w_phys, False for the raw window (unguided variants).
    """
    if not guided:
        return raw
    if cleaned.shape != raw.shape:
        raise ValueError(f"raw {tuple(raw.shape)} and cleaned {tuple(cleaned.shape)} windows differ")
    return cleaned * w_phys[:, None]
