"""Common baseline interface and registry.

A baseline is fitted on training subjects (and may use the inner validation subjects
for tuning or early stopping), then predicts targets in original units (cm or cm/s)
for selected target samples of one subject. Normalizers are fitted by the runner on
training data only and handed over in :class:`FitContext`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eegtraj.data.windows import Standardizer, SubjectSamples

REGISTRY: dict[str, type["Baseline"]] = {}


def register(name: str):
    """Class decorator: make a baseline available to the runner under ``name``."""
    def wrap(cls):
        if name in REGISTRY:
            raise ValueError(f"baseline {name!r} registered twice")
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return wrap


@dataclass
class FitContext:
    """Everything a baseline may use besides the data itself.

    train_masks / val_masks: per subject, boolean over ``SubjectSamples.sample_idx`` selecting
    that part's target samples (all True for LOSO, series-based for within-subject).
    """

    eeg_norm: dict[str, Standardizer]
    target_norm: Standardizer
    history: int
    lag_step: int
    seed: int
    train_masks: dict[str, np.ndarray]
    val_masks: dict[str, np.ndarray]
    smoke: bool = False
    run_dir: Path | None = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("eegtraj.baselines"))


class Baseline:
    """Base class. Subclasses take their config ``params`` as keyword arguments."""

    name: str = ""
    uses_eeg: bool = True  # False for floors that only use target timing (template, stereotypy)

    def __init__(self, **params: Any):
        self.params = params

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        raise NotImplementedError

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext) -> np.ndarray:
        """Predictions (n_selected, 3) in target units for ``subject.sample_idx[mask]``."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        """Extra facts for metrics.json (chosen hyperparameters, notes such as oracle timing)."""
        return {}


def make_baseline(name: str, params: dict[str, Any] | None = None) -> Baseline:
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
