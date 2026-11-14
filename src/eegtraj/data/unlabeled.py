"""Continuous EEG streams for self-supervised pretraining (proposed-model stage, decisions SSL-1 option b and SSL-2).

The main cache holds only the lift windows. Pretraining also uses the EEG between lifts (HS S1-S9) and the ST
recording. This module stores each subject's recordings as continuous streams on a 100 Hz grid, through the same
causal chain as the main cache (``eegtraj.data.pre_lift.eeg_stream``: interpolation, causal band-pass, reference,
causal anti-alias), and labels every stream sample with its source:

- ``lift``: the union of the cached lift windows of a series (overlapping neighbours merged);
- ``between``: the rest of HS S1-S9;
- ``st``: the whole HS ST recording.

Grid. Stream sample k of a recording is raw sample ``factor * k + factor - 1`` (``eegtraj.data.cache.keep_indices``
with the recording start as origin). A lift in the main cache uses the same formula with its own window start, so its
grid can be shifted by up to ``factor - 1`` raw samples (8 ms at 500 Hz) against the stream. The builder checks at
the raw rate that the stream reproduces the main cache's first sample of every lift.

A cleaned variant reuses the ocular ICA saved by ``eegtraj.data.band_variant`` (``ica_<band>_<ref>.npz``); it is not
refitted. That ICA was fitted on HS S1-S9 only; applying it to ST is a static spatial filter outside its fit data.

Arrays in ``<cache_dir>/<subject>/``:
    stream_eeg_<band>_<ref>.npy   float32 (32, N), the subject's recordings back to back (S1..S9, then ST)
    stream_segments.csv           one row per contiguous segment: series, source, start, end (stream indices,
                                  half-open), shared by every band
    stream_<band>_<ref>.json      provenance: band, settle times, alignment check, hours per source

The window dataset reads only these files: no kinematics, events, or condition labels.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from eegtraj.data.pre_lift import eeg_stream
from eegtraj.data.wayeeggal import load_series
from eegtraj.diagnostics.inventory import gaps, merge
from eegtraj.preprocessing.bad_channels import interpolate_bad_channels
from eegtraj.preprocessing.filters import butter_sos, settle_time_s
from eegtraj.preprocessing.ocular import OcularICA, remove_components

logger = logging.getLogger(__name__)

SOURCES = ("lift", "between", "st")
ST_SERIES = "ST"
ALIGN_TOL = 1e-3  # microvolts, as in eegtraj.data.pre_lift
EPS = 1e-8


def raw_to_grid(a: int, b: int, factor: int) -> tuple[int, int]:
    """Stream indices [k0, k1) of the grid samples whose raw index ``factor * k + factor - 1`` lies in [a, b)."""
    k0 = max(0, -(-(a - factor + 1) // factor))
    k1 = max(0, -(-(b - factor + 1) // factor))
    return k0, k1


def series_segments(lift_raw: list[tuple[int, int]], n_raw: int, factor: int) -> list[tuple[str, int, int]]:
    """Lift and between-lift segments of one HS series, as (source, k0, k1) on the stream grid, sorted, non-empty.

    Args:
        lift_raw: raw-sample intervals [hs_start, hs_start + factor * n_samples) of the cached lifts.
        n_raw: number of raw samples in the recording.
    """
    lifts = merge(lift_raw)
    if lifts and (lifts[0][0] < 0 or lifts[-1][1] > n_raw):
        raise ValueError("a lift lies outside the recording")
    out = [("lift", *raw_to_grid(a, b, factor)) for a, b in lifts]
    out += [("between", *raw_to_grid(a, b, factor)) for a, b in gaps(lifts, 0, n_raw)]
    return sorted((s for s in out if s[2] > s[1]), key=lambda s: s[1])


def load_ica(path: Path) -> OcularICA:
    """Ocular ICA saved by ``eegtraj.data.band_variant.build_band_variant``."""
    z = np.load(path)
    return OcularICA(mixing=z["mixing"], unmixing=z["unmixing"], selected=[int(i) for i in z["selected"]],
                     correlations={"VEOG": z["corr_veog"].tolist(), "HEOG": z["corr_heog"].tolist()},
                     rank=int(z["unmixing"].shape[0]), info={"names": z["names"].tolist()})


def build_stream(subject: str, cfg: dict[str, Any], raw_dir: Path, cache_dir: Path, band: str, reference: str,
                 ica_from: str | None = None) -> dict[str, Any]:
    """Build and save one subject's stream for ``band`` (defined in ``cfg``). Raises if it already exists.

    Args:
        cfg: main cache config, with ``band`` in ``cfg["eeg"]["bands_hz"]`` (see ``band_variant.with_band``).
        ica_from: name of a saved ICA (``ica_<ica_from>.npz`` in the subject folder) to remove before filtering,
            or None for the uncleaned chain.

    The lift samples are checked against ``eeg_<band>_<reference>.npy`` of the main cache.
    """
    subj_dir = Path(cache_dir) / subject
    out_eeg = subj_dir / f"stream_eeg_{band}_{reference}.npy"
    out_info = subj_dir / f"stream_{band}_{reference}.json"
    out_seg = subj_dir / "stream_segments.csv"
    existing = [str(p) for p in (out_eeg, out_info) if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    t_start = time.time()
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    trials = pd.read_csv(subj_dir / "trials.csv", usecols=["series", "hs_start_sample", "n_samples", "offset_start"])
    cached = np.load(subj_dir / f"eeg_{band}_{reference}.npy", mmap_mode="r")
    ica = load_ica(subj_dir / f"ica_{ica_from}.npz") if ica_from else None
    lo, hi = cfg["eeg"]["bands_hz"][band]
    tc = cfg["filter_transient"]

    chunks, segments, settle, max_err = [], [], {}, 0.0
    offset = 0
    recordings = [f"S{s}" for s in sorted(trials.series.unique())] + [ST_SERIES]
    for series in recordings:
        hs = load_series(subject, series, raw_dir)
        fs = hs.eeg_fs_hz
        factor = int(round(fs / cfg["fs_out_hz"]))
        n_raw = hs.eeg.shape[1]
        x = interpolate_bad_channels(hs.eeg.astype(np.float64), hs.eeg_names, fs, manifest["bad_channels"])
        x = remove_components(x, ica)
        probe = int(min(tc["probe_start_s"], n_raw / fs / 2) * fs)
        settle[series] = settle_time_s(x, fs, butter_sos(fs, cfg["eeg"]["filter_order"], lo, hi), probe, tc["tolerance"])
        stream = eeg_stream(x, hs.eeg_names, fs, cfg, [], band, reference)
        del x
        if series == ST_SERIES:
            segs = [("st", *raw_to_grid(0, n_raw, factor))]
        else:
            rows = trials[trials.series == int(series[1:])]
            for tr in rows.itertuples(index=False):
                err = float(np.abs(stream[:, int(tr.hs_start_sample) + factor - 1] - cached[:, int(tr.offset_start)]).max())
                if err > ALIGN_TOL:
                    raise ValueError(f"{subject} {series}: lift at raw {tr.hs_start_sample} differs from the main cache by {err}")
                max_err = max(max_err, err)
            lift_raw = [(int(t.hs_start_sample), int(t.hs_start_sample) + factor * int(t.n_samples))
                        for t in rows.itertuples(index=False)]
            segs = series_segments(lift_raw, n_raw, factor)
        grid = stream[:, factor - 1::factor].astype(np.float32)
        del stream
        segments += [{"series": series, "source": src, "start": offset + a, "end": offset + b} for src, a, b in segs]
        chunks.append(grid)
        offset += grid.shape[1]
        logger.info("%s %s: %d grid samples, settle %.1f s", subject, series, grid.shape[1], settle[series])

    eeg = np.concatenate(chunks, axis=1)
    seg_df = pd.DataFrame(segments, columns=["series", "source", "start", "end"])
    if out_seg.exists():
        if not pd.read_csv(out_seg).equals(seg_df):
            raise ValueError(f"{out_seg} differs from the segments of this build")
    else:
        seg_df.to_csv(out_seg, index=False)
    np.save(out_eeg, eeg)
    fs_out = float(cfg["fs_out_hz"])
    seconds = (seg_df.end - seg_df.start).groupby(seg_df.source).sum() / fs_out
    info = {"subject": subject, "band": band, "reference": reference, "band_hz": [lo, hi], "fs_hz": fs_out,
            "ica_from": ica_from, "ica_selected": None if ica is None else ica.selected,
            "shape": list(eeg.shape), "hours": {s: round(float(seconds.get(s, 0.0)) / 3600, 4) for s in SOURCES},
            "n_segments": {s: int((seg_df.source == s).sum()) for s in SOURCES},
            "lift_alignment_max_abs_error": max_err, "filter_settle_time_s": settle,
            "build_seconds": round(time.time() - t_start, 1)}
    out_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    logger.info("%s: stream %s %s %s, hours %s, %.0f s", subject, band, reference, eeg.shape, info["hours"],
                info["build_seconds"])
    return info


def window_starts(segments: pd.DataFrame, window: int, stride: int) -> np.ndarray:
    """Start indices of full windows inside each segment (a window never spans two segments)."""
    parts = [np.arange(a, b - window + 1, stride) for a, b in zip(segments.start, segments.end) if b - a >= window]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)


class UnlabeledWindows:
    """torch-style dataset of unlabeled EEG windows from the continuous streams.

    Items: one float32 tensor (32, window) per variant (a tuple when several variants are given), z-scored per
    subject and channel (D9: each subject's own statistics, no labels). Statistics come from the subject's ``lift``
    segments of that variant, i.e. the task EEG, like the main cache.

    Only ``stream_*`` files are read: no kinematics, events, or condition labels. Window positions are the same for
    every variant, so variants are aligned sample by sample.

    Args:
        cache_dir: main cache folder.
        subjects: subjects to draw windows from.
        variants: ``<band>_<ref>`` names, e.g. ``["broadband_car"]``.
        sources: subset of ``SOURCES``.
        window: samples per window (200 = 2 s at 100 Hz). stride: samples between window starts.
        exclude_subjects: subjects to drop (``strict_per_fold``: the fold's test subject).
    """

    def __init__(self, cache_dir: str | Path, subjects: Sequence[str], variants: Sequence[str],
                 sources: Sequence[str] = SOURCES, window: int = 200, stride: int = 50,
                 exclude_subjects: Sequence[str] = ()):
        import torch  # local import keeps numpy-only use free of torch

        unknown = set(sources) - set(SOURCES)
        if unknown:
            raise ValueError(f"unknown sources {sorted(unknown)}")
        self._torch = torch
        self.variants = list(variants)
        self.window = int(window)
        self.subjects = [s for s in subjects if s not in set(exclude_subjects)]
        self.eeg: dict[str, list[np.ndarray]] = {}
        self.mean: dict[str, list[np.ndarray]] = {}
        self.std: dict[str, list[np.ndarray]] = {}
        subj, src, starts = [], [], []
        for i, s in enumerate(self.subjects):
            d = Path(cache_dir) / s
            seg = pd.read_csv(d / "stream_segments.csv")
            self.eeg[s] = [np.load(d / f"stream_eeg_{v}.npy", mmap_mode="r") for v in self.variants]
            n = {a.shape[1] for a in self.eeg[s]}
            if len(n) != 1 or int(seg.end.max()) > n.pop():
                raise ValueError(f"{s}: streams and segments do not match")
            lift = seg[seg.source == "lift"]
            self.mean[s], self.std[s] = [], []
            for a in self.eeg[s]:
                m, sd = lift_stats(a, lift)
                self.mean[s].append(m)
                self.std[s].append(sd)
            for j, source in enumerate(SOURCES):
                if source not in sources:
                    continue
                st = window_starts(seg[seg.source == source], self.window, int(stride))
                starts.append(st)
                subj.append(np.full(st.size, i, np.int16))
                src.append(np.full(st.size, j, np.int8))
        self.subject_index = np.concatenate(subj) if subj else np.zeros(0, np.int16)
        self.source_index = np.concatenate(src) if src else np.zeros(0, np.int8)
        self.start = np.concatenate(starts) if starts else np.zeros(0, np.int64)

    def __len__(self) -> int:
        return len(self.start)

    def __getitem__(self, k: int):
        s = self.subjects[int(self.subject_index[k])]
        a = int(self.start[k])
        out = tuple(self._torch.from_numpy(((x[:, a:a + self.window] - m) / sd).astype(np.float32))
                    for x, m, sd in zip(self.eeg[s], self.mean[s], self.std[s]))
        return out[0] if len(out) == 1 else out


def lift_stats(eeg: np.ndarray, lift_segments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and SD (C, 1) over the lift segments of a stream (C, N)."""
    if lift_segments.empty:
        raise ValueError("no lift segments to compute statistics from")
    bounds = list(zip(lift_segments.start, lift_segments.end))
    total = sum(b - a for a, b in bounds)
    mean = sum(np.asarray(eeg[:, a:b], dtype=np.float64).sum(axis=1) for a, b in bounds) / total
    var = sum(((np.asarray(eeg[:, a:b], dtype=np.float64) - mean[:, None]) ** 2).sum(axis=1) for a, b in bounds) / total
    return mean[:, None].astype(np.float32), np.sqrt(var + EPS)[:, None].astype(np.float32)


def segment_hours(cache_dir: str | Path, subjects: Sequence[str], fs_hz: float = 100.0) -> pd.DataFrame:
    """Hours per subject and source from ``stream_segments.csv`` (subjects x SOURCES, plus ``all``)."""
    rows = {}
    for s in subjects:
        seg = pd.read_csv(Path(cache_dir) / s / "stream_segments.csv")
        sec = (seg.end - seg.start).groupby(seg.source).sum() / fs_hz
        rows[s] = {src: float(sec.get(src, 0.0)) / 3600 for src in SOURCES}
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(SOURCES))
    out["all"] = out.sum(axis=1)
    return out


def windows_from_config(cfg: dict[str, Any]) -> UnlabeledWindows:
    """Dataset from a config like ``configs/pretrain/unlabeled_windows.yaml``.

    ``strict_per_fold: true`` drops ``exclude_subject`` (SSL-1 option a, sensitivity check); it must then be set.
    """
    exclude: tuple[str, ...] = ()
    if cfg.get("strict_per_fold"):
        if not cfg.get("exclude_subject"):
            raise ValueError("strict_per_fold needs exclude_subject")
        exclude = (cfg["exclude_subject"],)
    return UnlabeledWindows(cfg["cache_dir"], cfg["subjects"], cfg["variants"], cfg["sources"], int(cfg["window"]),
                            int(cfg["stride"]), exclude)
