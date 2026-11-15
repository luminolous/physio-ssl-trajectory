"""Training-only labels for the decoder (design.md: "The kinematic events and condition labels used for the auxiliary
head and for template construction come from training trials only").

- Auxiliary phase classes come from the cached per-sample phase labels (derived from kinematic events). They are read
  only for a fold's training subjects; any other subject raises.
- Condition labels enter only through the landmark-registered templates (``eegtraj.models.landmark_template
  .fold_templates``), which also refuses validation and test subjects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eegtraj.data.cache import load_subject
from eegtraj.data.splits import Fold
from eegtraj.models.decoder import aux_classes


class LabelLeakageError(RuntimeError):
    """A training-only label was requested for a non-training subject."""


def training_aux_classes(cache_dir: str | Path, subject: str, fold: Fold) -> np.ndarray:
    """Auxiliary class (0..4) per cached sample (T,) of a training subject of ``fold``."""
    if subject not in fold.train_subjects:
        raise LabelLeakageError(f"{subject} is not a training subject of {fold.name}; auxiliary labels are training-only")
    data, _, _ = load_subject(cache_dir, subject, arrays=["phase"])
    return aux_classes(data["phase"])


def patch_aux_targets(aux: np.ndarray, end_samples: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Auxiliary target per patch: the class at each patch's last sample (``end_samples``, indices into ``aux``);
    -1 where ``valid`` is False (patch end outside the lift window or padding)."""
    out = np.full(end_samples.shape, -1, dtype=np.int64)
    out[valid] = aux[end_samples[valid]]
    return out
