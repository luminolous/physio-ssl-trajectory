"""Ocular cleaning by ICA per subject, with EOG proxies from scalp channels (no EOG channel exists).

The ICA is fitted once per subject, without labels, on that subject's continuous HS EEG (all series), after
bad-channel interpolation and before CAR. For fitting only, the EEG is band-passed 1-40 Hz (zero phase) and kept at
every ``factor``-th sample. A component is marked ocular when its source correlates with an EOG proxy at
``|r| >= threshold`` (VEOG = mean(Fp1, Fp2), HEOG = F7 - F8), at most ``max_components`` of them, strongest first.

Cleaning is a static spatial filter, x - A_sel W_sel x, applied to the raw interpolated EEG before the causal
band-pass, so the causal chain is unchanged. With no marked component the EEG is returned unchanged.
For the test subject the fit is transductive (its own EEG, no labels), like the per-subject normalization (D9).
These proxies also carry frontal brain activity, so removed components may hold some of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import FastICA

from eegtraj.preprocessing.filters import apply_sos, butter_sos

PROXIES = {"VEOG": (("Fp1", "Fp2"), ()), "HEOG": (("F7",), ("F8",))}


@dataclass
class OcularICA:
    """mixing (C, k) and unmixing (k, C) map channels to sources and back; ``selected`` are the ocular sources."""

    mixing: np.ndarray
    unmixing: np.ndarray
    selected: list[int]
    correlations: dict[str, list[float]]  # proxy -> |r| per source
    rank: int
    removed_variance: float = 0.0  # fraction of fit-data variance removed
    info: dict = field(default_factory=dict)


def eog_proxies(x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """VEOG = mean(Fp1, Fp2) and HEOG = F7 - F8 from (C, T) EEG."""
    out = {}
    for proxy, (plus, minus) in PROXIES.items():
        v = np.mean([x[names.index(c)] for c in plus], axis=0)
        if minus:
            v = v - np.mean([x[names.index(c)] for c in minus], axis=0)
        out[proxy] = v
    return out


def fit_data(x: np.ndarray, fs_hz: float, factor: int, band_hz: tuple[float, float] = (1.0, 40.0)) -> np.ndarray:
    """EEG used only to fit the ICA: zero-phase band-pass, then every ``factor``-th sample."""
    return apply_sos(x, butter_sos(fs_hz, 4, *band_hz), causal=False)[:, ::factor]


def data_rank(x: np.ndarray, rel_tol: float = 1e-6) -> int:
    vals = np.linalg.eigvalsh(np.cov(x))
    return int((vals > rel_tol * vals.max()).sum())


def fit_ocular_ica(x: np.ndarray, names: list[str], threshold: float, max_components: int, random_state: int,
                   max_iter: int = 1000) -> OcularICA:
    """Fit FastICA on fit data (C, T) and mark ocular sources by correlation with the EOG proxies."""
    x = np.asarray(x, dtype=np.float64)
    rank = data_rank(x)
    ica = FastICA(n_components=rank, whiten="unit-variance", random_state=random_state, max_iter=max_iter)
    sources = ica.fit_transform(x.T).T  # (k, T)
    unmixing = ica.components_  # applied to centered data
    mixing = ica.mixing_
    corr = {p: [abs(float(np.corrcoef(s, v)[0, 1])) for s in sources] for p, v in eog_proxies(x, names).items()}
    best = np.max(np.array(list(corr.values())), axis=0)
    order = [int(i) for i in np.argsort(-best) if best[i] >= threshold][:max_components]
    centered = x - x.mean(axis=1, keepdims=True)
    removed = mixing[:, order] @ (unmixing[order] @ centered) if order else np.zeros_like(centered)
    return OcularICA(mixing, unmixing, sorted(order), corr, rank,
                     float((removed ** 2).sum() / (centered ** 2).sum()),
                     {"n_iter": int(ica.n_iter_), "threshold": threshold, "max_components": max_components})


def remove_components(x: np.ndarray, ica: OcularICA | None) -> np.ndarray:
    """x - A_sel W_sel x. Returns ``x`` itself when nothing is selected, so the uncleaned chain is reproduced exactly."""
    if ica is None or not ica.selected:
        return x
    sel = ica.selected
    return x - ica.mixing[:, sel] @ (ica.unmixing[sel] @ x)
