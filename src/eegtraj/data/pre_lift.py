"""EEG before each lift's window, for causal histories longer than the window allows.

The main cache holds only each lift's window, concatenated. Windows of neighbouring lifts overlap and
are not contiguous, so reading across a lift's start in that array would return the wrong samples.
This module cuts the samples before each lift from the continuous HS recording, on the same 100 Hz
grid and through the same processing chain as the main cache.

Arrays in ``<cache_dir>/<subject>/``:
    pre_lift_eeg_<band>_<ref>.npy     float32 (32, n_lifts, n_pre), oldest sample first; the last
                                      sample is lift sample j = -1
    pre_lift_missing_<band>_<ref>.npy int32 (n_lifts,), samples before the start of the series. They
                                      hold the earliest available value on the lift's grid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eegtraj.data.wayeeggal import load_series
from eegtraj.preprocessing.bad_channels import interpolate_bad_channels
from eegtraj.preprocessing.filters import apply_sos, butter_sos
from eegtraj.preprocessing.reference import common_average

logger = logging.getLogger(__name__)

ALIGN_TOL = 1e-3  # microvolts; the first lift sample must match the main cache


def pre_lift_raw_indices(hs_start: int, factor: int, n_pre: int) -> np.ndarray:
    """Raw-rate indices of lift samples j = -n_pre, ..., -1. Lift sample j is at hs_start + factor * j + factor - 1."""
    return hs_start + factor * np.arange(-n_pre, 0) + factor - 1


def cut_pre_lift(stream: np.ndarray, hs_start: int, factor: int, n_pre: int) -> tuple[np.ndarray, int]:
    """Samples j = -n_pre, ..., -1 of one lift from a continuous stream (C, n_raw).

    Returns float32 (C, n_pre), oldest first, and the number of samples before the series start.
    Those hold the earliest value on the lift's grid.
    """
    idx = pre_lift_raw_indices(hs_start, factor, n_pre)
    below = idx < 0
    idx[below] = (hs_start + factor - 1) % factor
    return stream[:, idx].astype(np.float32), int(below.sum())


def eeg_stream(eeg: np.ndarray, names: list[str], fs_hz: float, cfg: dict[str, Any], bad_channels: list[str], band: str,
               reference: str) -> np.ndarray:
    """Continuous EEG through the main cache's chain: interpolate, causal band-pass, reference, causal anti-alias."""
    eeg_cfg = cfg["eeg"]
    x = interpolate_bad_channels(eeg.astype(np.float64), names, fs_hz, bad_channels)
    lo, hi = eeg_cfg["bands_hz"][band]
    x = apply_sos(x, butter_sos(fs_hz, eeg_cfg["filter_order"], lo, hi), causal=True)
    if reference == "car":
        x = common_average(x)
    aa_sos = butter_sos(fs_hz, eeg_cfg["antialias"]["order"],
                        high_hz=eeg_cfg["antialias"]["frac_of_nyquist"] * cfg["fs_out_hz"] / 2)
    return apply_sos(x, aa_sos, causal=True)


def build_pre_lift(subject: str, cfg: dict[str, Any], raw_dir: Path, cache_dir: Path, band: str, reference: str,
                   n_pre: int) -> dict[str, Any]:
    """Build one subject's pre-lift arrays. Raises if the result does not line up with the main cache."""
    subj_dir = Path(cache_dir) / subject
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    trials = pd.read_csv(subj_dir / "trials.csv")
    cached = np.load(subj_dir / f"eeg_{band}_{reference}.npy", mmap_mode="r")
    out = np.empty((cached.shape[0], len(trials), n_pre), dtype=np.float32)
    missing = np.zeros(len(trials), dtype=np.int32)
    for series, rows in trials.groupby("series", sort=False):
        hs = load_series(subject, f"S{series}", raw_dir)
        factor = int(round(hs.eeg_fs_hz / cfg["fs_out_hz"]))
        stream = eeg_stream(hs.eeg, hs.eeg_names, hs.eeg_fs_hz, cfg, manifest["bad_channels"], band, reference)
        for row, tr in zip(rows.index, rows.itertuples(index=False)):
            first = int(tr.hs_start_sample) + factor - 1
            error = np.abs(stream[:, first] - cached[:, int(tr.offset_start)]).max()
            if error > ALIGN_TOL:
                raise ValueError(f"{subject} S{series} lift {tr.lift}: first sample differs from the main cache by {error}")
            out[:, row], missing[row] = cut_pre_lift(stream, int(tr.hs_start_sample), factor, n_pre)
        del stream
    np.save(subj_dir / f"pre_lift_eeg_{band}_{reference}.npy", out)
    np.save(subj_dir / f"pre_lift_missing_{band}_{reference}.npy", missing)
    logger.info("%s: pre-lift %s, %d lifts reach before their series start", subject, out.shape, int((missing > 0).sum()))
    return {"subject": subject, "shape": list(out.shape), "n_lifts_with_missing": int((missing > 0).sum())}


def insert_pre_lift(eeg: np.ndarray, target: np.ndarray, trials: pd.DataFrame, pre_eeg: np.ndarray, n_pre: int
                    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Put ``n_pre`` samples before each lift in the concatenated arrays.

    Args:
        eeg: (C, T), lifts back to back. target: (3, T), same layout.
        pre_eeg: (C, n_lifts, n_stored), oldest first, with n_stored >= n_pre.

    Returns:
        eeg (C, T'), target (3, T') with zeros before each lift, trials with updated offsets, and a
        boolean (T',) that is True on the original lift samples. Sample j of a lift stays at
        offset_start + j, so anything computed from the offsets still works.
    """
    eeg_parts, target_parts, keep_parts, starts, ends = [], [], [], [], []
    pos = 0
    for row, tr in enumerate(trials.itertuples(index=False)):
        a, b = int(tr.offset_start), int(tr.offset_end)
        eeg_parts += [pre_eeg[:, row, pre_eeg.shape[2] - n_pre:], eeg[:, a:b]]
        target_parts += [np.zeros((target.shape[0], n_pre), target.dtype), target[:, a:b]]
        keep_parts += [np.zeros(n_pre, bool), np.ones(b - a, bool)]
        starts.append(pos + n_pre)
        ends.append(pos + n_pre + b - a)
        pos += n_pre + b - a
    return (np.concatenate(eeg_parts, axis=1), np.concatenate(target_parts, axis=1),
            trials.assign(offset_start=starts, offset_end=ends), np.concatenate(keep_parts))


def history_flags(cache_dir: str | Path, subject: str, band: str, reference: str, history: int) -> dict[str, float]:
    """Fractions of a subject's lifts whose first scored sample has a history in the filter start-up
    transient, or reaching before the series start, or leaving the lift window."""
    subj_dir = Path(cache_dir) / subject
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    tr = pd.read_csv(subj_dir / "trials.csv")
    fs = float(manifest["fs_hz"])
    first_used_j = np.ceil(tr.segment_start_s * fs).astype(int) - 1 - (history - 1)
    settle = tr.series.map(lambda k: manifest["filter_settle_time_s"][f"S{k}"][band])
    raw = tr.hs_start_sample + (first_used_j + 1) * 5 - 1
    return {"filter_transient": float((tr.t0_series_s + tr.segment_start_s - (history - 1) / fs < settle).mean()),
            "before_series_start": float((raw < 0).mean()), "outside_lift_window": float((first_used_j < 0).mean())}
