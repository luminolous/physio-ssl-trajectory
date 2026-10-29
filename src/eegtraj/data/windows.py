"""Lagged EEG windows aligned to kinematic targets (D1-D3, D9).

Target samples are the cached 100 Hz samples of each lift inside the scored segment
[onset - pre_onset_margin, tHandStop] (window time; sample j of a lift is at
``(j + 1) / fs``). For a target sample at index i, the EEG input is the causal history
``eeg[:, i - history + 1 : i + 1]``: it ends at the target sample and uses no later EEG.

Normalization (D9):
- EEG, ``per_subject`` (transductive, label-free): per-channel mean/SD over all cached
  EEG of that subject, including the test subject.
- EEG, ``train_subjects`` (strict variant): per-channel mean/SD pooled over the fold's
  training subjects only.
- Targets: per-axis mean/SD over target samples of the training subjects only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.data.cache import load_subject
from eegtraj.preprocessing.reference import channel_indices

WRIST_INDEX = 3  # sensor 4 in kin_pos (4, 3, T)
EPS = 1e-8


@dataclass
class SubjectSamples:
    """One subject's inputs and targets.

    eeg: float32 (n_channels, T), raw cached units (not normalized).
    target: float32 (3, T): wrist position re-centered per lift at onset (cm), or velocity (cm/s).
    sample_idx: int64 (n,), indices into T of the scored target samples.
    trial_row: int32 (n,), row of ``trials`` each target sample belongs to.
    """

    subject: str
    eeg: np.ndarray
    target: np.ndarray
    sample_idx: np.ndarray
    trial_row: np.ndarray
    trials: pd.DataFrame
    fs_hz: float


def segment_samples(trials: pd.DataFrame, fs_hz: float, history: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (into the subject's concatenated arrays) of scored target samples, and their trial row.

    Raises if a target sample's EEG history would reach before its own lift's first sample.
    """
    idx_parts, row_parts = [], []
    for row, tr in enumerate(trials.itertuples(index=False)):
        t = (np.arange(tr.n_samples) + 1) / fs_hz
        j = np.flatnonzero((t >= tr.segment_start_s) & (t <= tr.segment_end_s))
        if j.size == 0:
            raise ValueError(f"{tr.subject} S{tr.series} lift {tr.lift}: empty segment")
        if j[0] - (history - 1) < 0:
            raise ValueError(f"{tr.subject} S{tr.series} lift {tr.lift}: EEG history starts before the window")
        idx_parts.append(tr.offset_start + j)
        row_parts.append(np.full(j.size, row, dtype=np.int32))
    return np.concatenate(idx_parts).astype(np.int64), np.concatenate(row_parts)


def recentered_wrist(kin_pos: np.ndarray, trials: pd.DataFrame, fs_hz: float) -> np.ndarray:
    """Wrist position (3, T) minus the wrist position at each lift's kinematic onset (D1)."""
    wrist = np.asarray(kin_pos[WRIST_INDEX], dtype=np.float32).copy()
    for tr in trials.itertuples(index=False):
        sl = slice(tr.offset_start, tr.offset_end)
        t = (np.arange(tr.n_samples) + 1) / fs_hz
        ref = np.array([np.interp(tr.onset_s, t, wrist[a, sl]) for a in range(3)], dtype=np.float32)
        wrist[:, sl] -= ref[:, None]
    return wrist


def load_subject_samples(cache_dir: str | Path, subject: str, band: str, reference: str,
                         channel_set: str, target: str, history: int) -> SubjectSamples:
    """Load one subject from the cache as inputs + targets.

    Args:
        band: ``lowfreq`` | ``broadband``; reference: ``car`` | ``orig``.
        channel_set: ``all`` or a key of ``eegtraj.preprocessing.reference.CHANNEL_SETS``,
            or ``emg`` to use the EMG envelope as input (artifact control).
        target: ``position`` (re-centered wrist) | ``velocity`` (wrist).
        history: EEG window length in samples.
    """
    arrays = ["kin_pos", "wrist_vel", "emg_env" if channel_set == "emg" else f"eeg_{band}_{reference}"]
    data, trials, manifest = load_subject(cache_dir, subject, arrays=arrays)
    fs = float(manifest["fs_hz"])
    if channel_set == "emg":
        eeg = np.asarray(data["emg_env"], dtype=np.float32)
    else:
        eeg = np.asarray(data[f"eeg_{band}_{reference}"][channel_indices(manifest["eeg_names"], channel_set)],
                         dtype=np.float32)
    if target == "position":
        y = recentered_wrist(data["kin_pos"], trials, fs)
    elif target == "velocity":
        y = np.asarray(data["wrist_vel"], dtype=np.float32)
    else:
        raise ValueError(f"unknown target {target!r}")
    idx, rows = segment_samples(trials, fs, history)
    return SubjectSamples(subject, eeg, y, idx, rows, trials, fs)


@dataclass
class Standardizer:
    """Per-row (channel or axis) z-score: (x - mean) / std along the last axis."""

    mean: np.ndarray  # (n_rows, 1)
    std: np.ndarray  # (n_rows, 1)

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "Standardizer":
        """Fit on (n_rows, n_i) arrays pooled along the last axis."""
        total = sum(a.shape[-1] for a in arrays)
        mean = sum(a.astype(np.float64).sum(axis=-1) for a in arrays) / total
        var = sum(((a.astype(np.float64) - mean[:, None]) ** 2).sum(axis=-1) for a in arrays) / total
        return cls(mean[:, None].astype(np.float32), np.sqrt(var + EPS)[:, None].astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return (z * self.std + self.mean).astype(np.float32)


def fit_eeg_normalizers(subjects: list[SubjectSamples], train_subjects: tuple[str, ...],
                        mode: str) -> dict[str, Standardizer]:
    """EEG standardizer per subject (D9).

    ``per_subject``: each subject's own statistics (transductive, no labels).
    ``train_subjects``: one standardizer fitted on the training subjects, used for all.
    """
    if mode == "per_subject":
        return {s.subject: Standardizer.fit([s.eeg]) for s in subjects}
    if mode == "train_subjects":
        shared = Standardizer.fit([s.eeg for s in subjects if s.subject in train_subjects])
        return {s.subject: shared for s in subjects}
    raise ValueError(f"unknown EEG normalization mode {mode!r}")


def fit_target_normalizer(subjects: list[SubjectSamples], train_subjects: tuple[str, ...],
                          train_masks: dict[str, np.ndarray] | None = None) -> Standardizer:
    """Target standardizer from training subjects' scored samples only.

    ``train_masks`` optionally restricts each training subject to a subset of its target
    samples (boolean over ``sample_idx``), e.g. training series in the within-subject mode.
    """
    parts = []
    for s in subjects:
        if s.subject not in train_subjects:
            continue
        idx = s.sample_idx if train_masks is None else s.sample_idx[train_masks[s.subject]]
        parts.append(s.target[:, idx])
    if not parts:
        raise ValueError("no training samples to fit the target normalizer")
    return Standardizer.fit(parts)


def sample_masks(fold, subjects: list[SubjectSamples]) -> dict[str, dict[str, np.ndarray]]:
    """Per-part (train/val/test), per-subject boolean masks over ``sample_idx`` for a ``splits.Fold``."""
    out: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
    for s in subjects:
        rows = fold.masks(s.trials)
        for part in out:
            out[part][s.subject] = rows[part][s.trial_row]
    return out


def lag_samples(history: int, step: int) -> np.ndarray:
    """Lags 0, step, 2*step, ... strictly inside a ``history``-sample window."""
    return np.arange(0, history, step)


def lagged_features(eeg: np.ndarray, sample_idx: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Flattened causal lag features for linear models.

    Args:
        eeg: float32 (n_channels, T).
        sample_idx: int (n,), target sample indices.
        lags: int (n_lags,), non-negative lags in samples (0 = target sample itself).

    Returns:
        float32 (n, n_channels * n_lags), ordered channel-major: column ``c * n_lags + l``
        holds ``eeg[c, sample_idx - lags[l]]``.
    """
    lags = np.asarray(lags)
    if (lags < 0).any():
        raise ValueError("lags must be non-negative (causal)")
    x = eeg[:, sample_idx[:, None] - lags[None, :]]  # (C, n, L)
    return np.ascontiguousarray(x.transpose(1, 0, 2).reshape(len(sample_idx), -1), dtype=np.float32)


def window_at(eeg: np.ndarray, i: int, history: int) -> np.ndarray:
    """Causal EEG window (n_channels, history) ending at sample ``i`` (inclusive)."""
    return eeg[:, i - history + 1:i + 1]


class WindowDataset:
    """torch-style dataset of (EEG window, target) pairs for deep models.

    Items: ``(x, y)`` with x float32 tensor (n_channels, history) and y float32 tensor (3,).
    Inputs and targets are normalized with the given standardizers. Also exposes
    ``subject`` and ``trial_row`` per item for trial-level evaluation.
    """

    def __init__(self, subjects: list[SubjectSamples], eeg_norm: dict[str, Standardizer],
                 target_norm: Standardizer, history: int, masks: dict[str, np.ndarray] | None = None):
        import torch  # local import keeps sklearn-only use free of torch

        self._torch = torch
        self.history = history
        self.eeg = {s.subject: eeg_norm[s.subject].transform(s.eeg) for s in subjects}
        self.target = {s.subject: target_norm.transform(s.target) for s in subjects}
        subj, idx, rows = [], [], []
        for s in subjects:
            m = np.ones(len(s.sample_idx), bool) if masks is None else masks[s.subject]
            subj.extend([s.subject] * int(m.sum()))
            idx.append(s.sample_idx[m])
            rows.append(s.trial_row[m])
        self.subject = np.array(subj)
        self.sample_idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        self.trial_row = np.concatenate(rows) if rows else np.zeros(0, np.int32)

    def __len__(self) -> int:
        return len(self.sample_idx)

    def __getitem__(self, k: int):
        s, i = self.subject[k], int(self.sample_idx[k])
        x = window_at(self.eeg[s], i, self.history)
        y = self.target[s][:, i]
        return self._torch.from_numpy(np.ascontiguousarray(x)), self._torch.from_numpy(np.ascontiguousarray(y))
