"""Leakage-safe split schemes (protocol decision D6).

- Screening subjects are drawn once with the config seed and are excluded from
  every fold built here (they belong to Exp 1 only).
- ``loso_folds``: leave-one-subject-out over the remaining subjects; the inner
  validation subjects are drawn from that fold's training subjects only.
- ``within_subject_folds``: secondary diagnostic, train on early series, test on
  later series of the same subject. Confound: S7-S9 are always weight series,
  so train and test differ in condition mix. Never a main result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ALL_SERIES = tuple(range(1, 10))


@dataclass(frozen=True)
class Fold:
    """One train/val/test assignment. Units are (subject, series) pairs."""

    name: str
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_series: tuple[int, ...] = ALL_SERIES
    val_series: tuple[int, ...] = ALL_SERIES
    test_series: tuple[int, ...] = ALL_SERIES

    def masks(self, trials: pd.DataFrame) -> dict[str, np.ndarray]:
        """Boolean row masks for ``train``/``val``/``test`` over a table with ``subject`` and ``series`` columns."""
        def sel(subjects: tuple[str, ...], series: tuple[int, ...]) -> np.ndarray:
            return (trials.subject.isin(subjects) & trials.series.isin(series)).to_numpy()
        return {
            "train": sel(self.train_subjects, self.train_series),
            "val": sel(self.val_subjects, self.val_series),
            "test": sel(self.test_subjects, self.test_series),
        }

    def units(self, part: str) -> set[tuple[str, int]]:
        """(subject, series) pairs of ``part`` (train, val, or test)."""
        subjects = getattr(self, f"{part}_subjects")
        series = getattr(self, f"{part}_series")
        return {(s, k) for s in subjects for k in series}


def draw_screening(subjects: list[str], n: int, seed: int) -> list[str]:
    """Seeded draw of the reserved screening subjects (same method as scripts/draw_screening_subjects.py)."""
    idx = np.random.default_rng(seed).choice(len(subjects), size=n, replace=False)
    return sorted((subjects[i] for i in idx), key=lambda s: int(s[1:]))


def load_split_config(path: str | Path) -> dict[str, Any]:
    """Load the split config and check that the recorded screening subjects match the seeded draw."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    drawn = draw_screening(cfg["subjects"], cfg["n_screening"], cfg["seed"])
    if cfg.get("screening_subjects") != drawn:
        raise ValueError(f"screening_subjects {cfg.get('screening_subjects')} != seeded draw {drawn}")
    return cfg


def main_subjects(cfg: dict[str, Any]) -> list[str]:
    """Subjects available for Exp 2 (all minus screening), in config order."""
    return [s for s in cfg["subjects"] if s not in cfg["screening_subjects"]]


def loso_folds(cfg: dict[str, Any]) -> list[Fold]:
    """LOSO over the non-screening subjects with a seeded inner validation per fold."""
    subjects = main_subjects(cfg)
    folds = []
    for i, test in enumerate(subjects):
        train_pool = [s for s in subjects if s != test]
        rng = np.random.default_rng([cfg["seed"], i])
        val_idx = rng.choice(len(train_pool), size=cfg["n_inner_val"], replace=False)
        val = tuple(sorted((train_pool[j] for j in val_idx), key=lambda s: int(s[1:])))
        train = tuple(s for s in train_pool if s not in val)
        folds.append(Fold(name=f"loso_{test}", train_subjects=train, val_subjects=val, test_subjects=(test,)))
    return folds


def within_subject_folds(cfg: dict[str, Any]) -> list[Fold]:
    """Per non-screening subject: train/val/test on disjoint series of that subject (diagnostic only)."""
    ws = cfg["within_subject"]
    train, val, test = (tuple(ws[k]) for k in ("train_series", "val_series", "test_series"))
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("within-subject train/val/test series must be disjoint")
    return [
        Fold(name=f"within_{s}", train_subjects=(s,), val_subjects=(s,), test_subjects=(s,),
             train_series=train, val_series=val, test_series=test)
        for s in main_subjects(cfg)
    ]


def make_folds(cfg: dict[str, Any], mode: str) -> list[Fold]:
    """Folds for ``mode`` = ``"loso"`` or ``"within_subject"``."""
    if mode == "loso":
        return loso_folds(cfg)
    if mode == "within_subject":
        return within_subject_folds(cfg)
    raise ValueError(f"unknown split mode {mode!r}")
