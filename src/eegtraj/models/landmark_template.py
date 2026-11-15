"""Landmark-registered movement templates T(phi) for the biomechanics-informed head (proposed-model, design.md).

Phase. Each lift is mapped to a normalized phase phi in [0, 1], from the kinematic onset (phi = 0) to the hand stop
(phi = 1). The five landmarks (kinematic onset, first digit touch, lift-off, replace, hand stop) are mapped to fixed
phi positions: their median fractions of the onset-to-stop duration over the fold's training lifts. Between
landmarks the map is linear in time. Before the onset phi = 0, after the hand stop phi = 1.

Template. Each training lift's target (wrist position re-centered at onset, D1) is resampled on a phi grid through
the inverse of its own map, then averaged per weight condition (165, 330, 660 g). So T(0) is the rest position at the
onset (0 by the re-centering) and T(1) the position at the hand stop.

Leakage. Templates use only the fold's training subjects (``fold.train_subjects``; inner-validation and test
subjects are excluded). The kinematic events and condition labels are read for those lifts only.

``TemplateBank`` exposes the templates to torch: linear interpolation on the phi grid, differentiable in phi
(almost everywhere) and in the mixture weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from eegtraj.data.windows import SubjectSamples

LANDMARK_EVENTS = ("onset_s", "ev_tFirstDigitTouch", "ev_tLiftOff", "ev_tReplace", "ev_tHandStop")
CUE_COLUMN = "ev_LEDOn"


def lift_landmarks(trials) -> np.ndarray:
    """float64 (n_lifts, 5) landmark times in window seconds (sample j of a lift is at (j + 1) / fs)."""
    return trials[list(LANDMARK_EVENTS)].to_numpy(np.float64)


def ordered(landmarks: np.ndarray) -> np.ndarray:
    """bool (n_lifts,): landmarks strictly increasing (required for a valid map)."""
    return (np.diff(landmarks, axis=1) > 0).all(axis=1)


def landmark_fractions(landmarks: np.ndarray) -> np.ndarray:
    """(n_lifts, 5) fraction of the onset-to-stop duration at each landmark (0 at onset, 1 at hand stop)."""
    return (landmarks - landmarks[:, :1]) / (landmarks[:, -1:] - landmarks[:, :1])


def time_to_phase(t_s: np.ndarray, landmarks_s: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """phi of times ``t_s`` for one lift: piecewise linear through (landmarks_s, fractions), clipped to [0, 1]."""
    return np.interp(t_s, landmarks_s, fractions)


def phase_to_time(phi: np.ndarray, landmarks_s: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`time_to_phase` on [0, 1]."""
    return np.interp(phi, fractions, landmarks_s)


def register_lift(y: np.ndarray, fs_hz: float, landmarks_s: np.ndarray, fractions: np.ndarray,
                  grid: np.ndarray) -> np.ndarray:
    """Resample one lift's target on the phi grid.

    Args:
        y: (3, n) target over the whole lift window, sample j at (j + 1) / fs_hz.
        landmarks_s: (5,) the lift's landmarks (window s). fractions: (5,) fold landmark positions in phi.
        grid: (G,) phi values in [0, 1].

    Returns:
        float64 (3, G).
    """
    t = (np.arange(y.shape[1]) + 1) / fs_hz
    tg = phase_to_time(grid, landmarks_s, fractions)
    return np.stack([np.interp(tg, t, y[a]) for a in range(y.shape[0])])


@dataclass
class LandmarkTemplates:
    """Templates of one fold.

    grid: (G,) phi. values: float32 (n_conditions, 3, G), cm. fractions: (5,) landmark positions in phi.
    landmarks_since_cue_s: (5,) median landmark times after LED-on over the training lifts (for plots only).
    """

    conditions: tuple[int, ...]
    grid: np.ndarray
    values: np.ndarray
    fractions: np.ndarray
    landmarks_since_cue_s: np.ndarray
    counts: dict[int, int]
    subjects: tuple[str, ...]
    n_excluded: int = 0
    info: dict = field(default_factory=dict)

    def __call__(self, phi: np.ndarray, condition: int) -> np.ndarray:
        """T(phi) of one condition, shape (*phi.shape, 3)."""
        v = self.values[self.conditions.index(condition)]
        p = np.clip(np.asarray(phi, np.float64), 0.0, 1.0)
        return np.stack([np.interp(p, self.grid, v[a]) for a in range(v.shape[0])], axis=-1)


def fit_landmark_templates(subjects: Sequence[SubjectSamples], train_subjects: Sequence[str],
                           conditions: Sequence[int] = (165, 330, 660), n_grid: int = 201,
                           condition_column: str = "weight_g") -> LandmarkTemplates:
    """Fit the templates from the lifts of ``train_subjects`` only (other subjects in ``subjects`` are ignored).

    Lifts whose landmarks are not strictly increasing are skipped and counted in ``n_excluded``.
    """
    train = [s for s in subjects if s.subject in set(train_subjects)]
    if not train:
        raise ValueError("no training subjects")
    marks = [lift_landmarks(s.trials) for s in train]
    keep = [ordered(m) for m in marks]
    fractions = np.median(np.concatenate([landmark_fractions(m[k]) for m, k in zip(marks, keep)]), axis=0)
    fractions[0], fractions[-1] = 0.0, 1.0
    grid = np.linspace(0.0, 1.0, int(n_grid))
    sums = {c: np.zeros((3, grid.size)) for c in conditions}
    counts = {c: 0 for c in conditions}
    since_cue = []
    for s, m, k in zip(train, marks, keep):
        tr = s.trials
        since_cue.append(m[k] - tr[CUE_COLUMN].to_numpy(np.float64)[k, None])
        for row in np.flatnonzero(k):
            c = int(tr[condition_column].iloc[row])
            if c not in sums:
                continue
            a, b = int(tr.offset_start.iloc[row]), int(tr.offset_end.iloc[row])
            sums[c] += register_lift(s.target[:, a:b], s.fs_hz, m[row], fractions, grid)
            counts[c] += 1
    missing = [c for c in conditions if counts[c] == 0]
    if missing:
        raise ValueError(f"no training lifts for conditions {missing}")
    values = np.stack([sums[c] / counts[c] for c in conditions]).astype(np.float32)
    return LandmarkTemplates(tuple(int(c) for c in conditions), grid, values, fractions,
                             np.median(np.concatenate(since_cue), axis=0), counts, tuple(s.subject for s in train),
                             int(sum((~k).sum() for k in keep)))


def fold_templates(subjects: Sequence[SubjectSamples], fold, **kwargs) -> LandmarkTemplates:
    """Templates of a ``splits.Fold`` from its training subjects. Raises if a test or validation subject was used."""
    out = fit_landmark_templates(subjects, fold.train_subjects, **kwargs)
    leaked = set(out.subjects) & (set(fold.test_subjects) | set(fold.val_subjects))
    if leaked:
        raise RuntimeError(f"{fold.name}: templates used held-out subjects {sorted(leaked)}")
    return out


class TemplateBank(nn.Module):
    """T(phi) for every condition, by linear interpolation on a uniform phi grid.

    Buffer ``values``: (G, n_conditions, 3). ``forward(phi, weights=None)``: phi (...,) is clipped to [0, 1]; with
    weights (..., n_conditions) it returns the mixture sum_c w_c T_c(phi), shape (..., 3); without, (..., n_conditions, 3).
    Differentiable in phi (almost everywhere) and in the weights.
    """

    def __init__(self, values: torch.Tensor | np.ndarray):
        super().__init__()
        values = torch.as_tensor(values, dtype=torch.float32)  # (n_conditions, 3, G)
        self.register_buffer("values", values.permute(2, 0, 1).contiguous())

    @classmethod
    def from_templates(cls, templates: LandmarkTemplates) -> "TemplateBank":
        return cls(templates.values)

    def forward(self, phi: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        g = self.values.shape[0]
        x = phi.clamp(0.0, 1.0) * (g - 1)
        i0 = x.detach().floor().clamp(0, g - 2).long()
        w = (x - i0.to(x.dtype)).unsqueeze(-1).unsqueeze(-1)
        t = self.values[i0] * (1 - w) + self.values[i0 + 1] * w  # (..., C, 3)
        if weights is None:
            return t
        return (weights.unsqueeze(-1) * t).sum(dim=-2)


# ---------------------------------------------------------------------------------------------------------------------
# Phase prior (SSL-20): distribution of the training lifts' phase at each time since LED-on, among the lifts whose
# scoring segment contains that time.

QUAD_NODES = 32  # Gauss-Hermite nodes over phi; clip kinks limit accuracy: |error| <= 0.05 cm for sigma <= 0.3


def extended_phase(t_s: np.ndarray, landmarks_s: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Unclipped phi: piecewise linear through the landmarks, extended linearly outside them with the slope of the
    first / last landmark interval (phi < 0 before the onset, > 1 after the hand stop)."""
    out = np.interp(t_s, landmarks_s, fractions)
    lo, hi = t_s < landmarks_s[0], t_s > landmarks_s[-1]
    out[lo] = (t_s[lo] - landmarks_s[0]) * (fractions[1] - fractions[0]) / (landmarks_s[1] - landmarks_s[0])
    out[hi] = 1.0 + (t_s[hi] - landmarks_s[-1]) * (fractions[-1] - fractions[-2]) / (landmarks_s[-1] - landmarks_s[-2])
    return out


@dataclass
class PhasePrior:
    """Mean ``mu`` and SD ``sd`` of the training lifts' (unclipped) phase at time ``t0 + k dt`` since LED-on, over the
    lifts whose scoring segment [segment_start_s, segment_end_s] contains that time (the same samples template_cue
    averages). ``mu`` is made non-decreasing (running maximum). Outside the grid the edge values are held."""

    t0: float
    dt: float
    mu: np.ndarray
    sd: np.ndarray
    counts: np.ndarray

    def __call__(self, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grid = self.t0 + self.dt * np.arange(self.mu.size)
        return np.interp(t_s, grid, self.mu), np.interp(t_s, grid, self.sd)


def fit_phase_prior(subjects: Sequence[SubjectSamples], train_subjects: Sequence[str], fractions: np.ndarray,
                    t0: float = -1.5, t1: float = 12.0, dt: float = 0.01, min_count: int = 20,
                    min_sd: float = 1e-3) -> PhasePrior:
    """Phase prior from the lifts of ``train_subjects`` only (lifts with non-increasing landmarks are skipped).
    Grid times with fewer than ``min_count`` lifts take the nearest grid time that has enough."""
    grid = np.arange(t0, t1 + dt / 2, dt)
    s1, s2, cnt = np.zeros(grid.size), np.zeros(grid.size), np.zeros(grid.size)
    for s in subjects:
        if s.subject not in set(train_subjects):
            continue
        tr = s.trials
        marks = lift_landmarks(tr)
        led = tr[CUE_COLUMN].to_numpy(np.float64)
        seg = tr[["segment_start_s", "segment_end_s"]].to_numpy(np.float64)
        for r in np.flatnonzero(ordered(marks)):
            tw = grid + led[r]
            inside = (tw >= seg[r, 0]) & (tw <= seg[r, 1])
            p = extended_phase(tw[inside], marks[r], fractions)
            s1[inside] += p
            s2[inside] += p * p
            cnt[inside] += 1
    ok = cnt >= min_count
    if not ok.any():
        raise ValueError("no grid time with enough training lifts")
    mean = s1[ok] / cnt[ok]
    sd = np.sqrt(np.maximum(s2[ok] / cnt[ok] - mean ** 2, min_sd ** 2))
    mu = np.maximum.accumulate(np.interp(grid, grid[ok], mean))
    return PhasePrior(float(t0), float(dt), mu.astype(np.float32), np.interp(grid, grid[ok], sd).astype(np.float32),
                      cnt.astype(np.int64))


def expected_template(bank: "TemplateBank", phi: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor,
                      n_nodes: int = QUAD_NODES) -> torch.Tensor:
    """E_{x ~ N(phi, sigma^2)} [sum_c w_c T_c(clip(x, 0, 1))] by Gauss-Hermite quadrature. phi, sigma (...,),
    weights (..., C) -> (..., 3). sigma = 0 gives the template at phi."""
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    z = torch.as_tensor(z, dtype=phi.dtype, device=phi.device)
    w = torch.as_tensor(w / w.sum(), dtype=phi.dtype, device=phi.device)
    x = phi.unsqueeze(-1) + sigma.unsqueeze(-1) * z  # (..., K)
    t = bank(x, weights.unsqueeze(-2).expand(*x.shape, weights.shape[-1]))  # (..., K, 3)
    return (w.unsqueeze(-1) * t).sum(dim=-2)
