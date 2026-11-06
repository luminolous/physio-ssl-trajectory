"""Extra EEG bands added to an existing cache, including the EEG before each lift.

Adding a band to ``configs/cache/default.yaml`` would change the config hash and start a new cache
folder. This module instead adds arrays to the existing folder and leaves its other files alone.
The processing chain and the 100 Hz grid are those of the main cache.

Arrays in ``<cache_dir>/<subject>/``:
    eeg_<band>_<ref>.npy              float32 (32, T), lifts back to back, as in the main cache
    pre_lift_eeg_<band>_<ref>.npy     float32 (32, n_lifts, n_pre), see ``eegtraj.data.pre_lift``
    pre_lift_missing_<band>_<ref>.npy int32 (n_lifts,)
    variant_<band>_<ref>.json         band edges, settle time per series, provenance

``eegtraj.data.windows.load_subject_samples`` reads these arrays with ``band=<band>``.

An envelope variant (``envelope`` given) stores band power instead of the signal. The chain is causal:
interpolate, band-pass, CAR, rectify, low-pass, keep every factor-th sample. CAR comes before rectifying
because it has no meaning on envelopes. The low-pass replaces the anti-alias filter.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eegtraj.data.pre_lift import cut_pre_lift, eeg_stream
from eegtraj.data.wayeeggal import load_series
from eegtraj.preprocessing.bad_channels import interpolate_bad_channels
from eegtraj.preprocessing.filters import apply_sos, butter_sos, settle_time_s
from eegtraj.preprocessing.reference import common_average

logger = logging.getLogger(__name__)


def with_band(cfg: dict[str, Any], band: str, band_hz: list[float]) -> dict[str, Any]:
    """Copy of a cache config that also defines ``band``. Refuses to redefine a band differently."""
    known = cfg["eeg"]["bands_hz"].get(band)
    if known is not None and list(known) != list(band_hz):
        raise ValueError(f"band {band} is already defined as {known}")
    out = copy.deepcopy(cfg)
    out["eeg"]["bands_hz"][band] = list(band_hz)
    return out


def lift_raw_indices(hs_start: int, factor: int, n_samples: int) -> np.ndarray:
    """Raw-rate indices of lift samples j = 0, ..., n_samples - 1 (same grid as the main cache)."""
    return hs_start + factor * np.arange(n_samples) + factor - 1


def rectified_band(x: np.ndarray, sos: np.ndarray, reference: str) -> np.ndarray:
    """Causal band-pass, then the reference, then rectification.

    Args:
        x: float array (C, n_raw), bad channels already interpolated. sos: band-pass sections.

    Returns:
        float64 array (C, n_raw).
    """
    filtered = apply_sos(x, sos, causal=True)
    return np.abs(common_average(filtered) if reference == "car" else filtered)


def compute_band_variant(subject: str, cfg: dict[str, Any], raw_dir: Path, cache_dir: Path, band: str, reference: str,
                         n_pre: int, envelope: dict[str, Any] | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Compute one subject's arrays for a band defined in ``cfg``. Writes nothing.

    Args:
        envelope: ``lowpass_hz`` and ``order`` for a band-power envelope, or None for the band-passed signal.

    Returns:
        eeg float32 (32, T), pre float32 (32, n_lifts, n_pre), missing int32 (n_lifts,), and the
        filter settle time (s) per series key, for example ``{"S1": 13.1}``.
    """
    subj_dir = Path(cache_dir) / subject
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    trials = pd.read_csv(subj_dir / "trials.csv")
    n_total = int(trials.offset_end.iloc[-1])
    eeg = np.empty((len(manifest["eeg_names"]), n_total), dtype=np.float32)
    pre = np.empty((eeg.shape[0], len(trials), n_pre), dtype=np.float32)
    missing = np.zeros(len(trials), dtype=np.int32)
    settle: dict[str, float] = {}
    lo, hi = cfg["eeg"]["bands_hz"][band]
    tc = cfg["filter_transient"]
    for series, rows in trials.groupby("series", sort=False):
        hs = load_series(subject, f"S{series}", raw_dir)
        fs = hs.eeg_fs_hz
        factor = int(round(fs / cfg["fs_out_hz"]))
        x = interpolate_bad_channels(hs.eeg.astype(np.float64), hs.eeg_names, fs, manifest["bad_channels"])
        probe = int(min(tc["probe_start_s"], x.shape[1] / fs / 2) * fs)
        sos = butter_sos(fs, cfg["eeg"]["filter_order"], lo, hi)
        settle[f"S{series}"] = settle_time_s(x, fs, sos, probe, tc["tolerance"])
        if envelope is None:
            stream = eeg_stream(x, hs.eeg_names, fs, cfg, [], band, reference)
        else:
            lowpass = butter_sos(fs, envelope["order"], high_hz=envelope["lowpass_hz"])
            rectified = rectified_band(x, sos, reference)
            settle[f"S{series}"] += settle_time_s(rectified, fs, lowpass, probe, tc["tolerance"])
            stream = apply_sos(rectified, lowpass, causal=True)
            del rectified
        for row, tr in zip(rows.index, rows.itertuples(index=False)):
            idx = lift_raw_indices(int(tr.hs_start_sample), factor, int(tr.offset_end - tr.offset_start))
            eeg[:, int(tr.offset_start):int(tr.offset_end)] = stream[:, idx].astype(np.float32)
            pre[:, row], missing[row] = cut_pre_lift(stream, int(tr.hs_start_sample), factor, n_pre)
        del stream, x
    return eeg, pre, missing, settle


def build_band_variant(subject: str, cfg: dict[str, Any], raw_dir: Path, cache_dir: Path, band: str, reference: str,
                       n_pre: int, envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute and save one subject's arrays. Raises if any of them already exists."""
    subj_dir = Path(cache_dir) / subject
    paths = {"eeg": subj_dir / f"eeg_{band}_{reference}.npy",
             "pre": subj_dir / f"pre_lift_eeg_{band}_{reference}.npy",
             "missing": subj_dir / f"pre_lift_missing_{band}_{reference}.npy",
             "info": subj_dir / f"variant_{band}_{reference}.json"}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    t_start = time.time()
    eeg, pre, missing, settle = compute_band_variant(subject, cfg, raw_dir, cache_dir, band, reference, n_pre, envelope)
    np.save(paths["eeg"], eeg)
    np.save(paths["pre"], pre)
    np.save(paths["missing"], missing)
    info = {"subject": subject, "band": band, "reference": reference, "band_hz": list(cfg["eeg"]["bands_hz"][band]),
            "envelope": envelope, "n_pre": n_pre, "eeg_shape": list(eeg.shape), "pre_shape": list(pre.shape),
            "n_lifts_with_missing": int((missing > 0).sum()), "filter_settle_time_s": settle,
            "build_seconds": round(time.time() - t_start, 1)}
    paths["info"].write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    logger.info("%s: %s %s, eeg %s, pre %s, %.1f s", subject, band, reference, eeg.shape, pre.shape, info["build_seconds"])
    return info
