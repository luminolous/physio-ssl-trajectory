"""Preprocessed per-subject cache (protocol D1-D5, D8).

Layout: ``<cache_root>/<config_hash>/``
    config.yaml                 hashed part of the build config
    <subject>/manifest.json     shapes, rates, bad channels, transient times, provenance
    <subject>/trials.csv        one row per lift: metadata, events, onset, flags, offsets
    <subject>/<array>.npy       time-concatenated arrays at fs_out (all lifts back to back)

Arrays (float32 unless noted; T = total samples over lifts, at ``fs_out_hz``):
    eeg_<band>_<ref>   (32, T)   causal band-pass, bad channels interpolated, ref = car | orig
    kin_pos            (4, 3, T) zero-phase smoothed positions of sensors 1..4 (cm, lab frame)
    wrist_vel          (3, T)    velocity of sensor 4 (cm/s)
    emg_env            (5, T)    rectified, causal low-passed EMG
    phase              (T,) int8 phase label, see PHASES

Lift i occupies samples ``offset[i]:offset[i+1]`` (columns ``offset_start``/``offset_end``
of trials.csv). Sample j of a lift is at window time ``(j + 1) / fs_out_hz`` s, i.e. the
500 Hz window samples ``5 j + 4`` are kept, so all arrays share one grid.

Processing order per series (continuous HS data, so filters see no window edges):
interpolate bad channels -> causal band-pass -> optional CAR -> causal anti-alias ->
cut each lift's window -> keep every factor-th sample.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eegtraj.data import raw_io
from eegtraj.data.wayeeggal import Trial, load_series, load_trials
from eegtraj.preprocessing.bad_channels import amplitude_ratio, detect_bad_channels, interpolate_bad_channels
from eegtraj.preprocessing.filters import apply_sos, butter_sos, settle_time_s
from eegtraj.preprocessing.kinematics import smooth_position, velocity
from eegtraj.preprocessing.onset import kinematic_onset
from eegtraj.preprocessing.reference import common_average

logger = logging.getLogger(__name__)

UNHASHED_KEYS = ("paths", "subjects")
WRIST = 4
PHASES = ("pre", "reach", "load", "hold", "replace", "release", "post")
# Event that ends each phase (the first phase ends at the kinematic onset).
PHASE_END_EVENTS = ("onset", "tFirstDigitTouch", "tLiftOff", "tReplace", "tBothReleased", "tHandStop")


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of the config, ignoring paths and the subject list."""
    hashed = {k: v for k, v in cfg.items() if k not in UNHASHED_KEYS}
    return hashlib.sha1(json.dumps(hashed, sort_keys=True).encode()).hexdigest()[:10]


def keep_indices(n_times: int, factor: int) -> np.ndarray:
    """0-based raw-rate indices kept inside a window: factor-1, 2*factor-1, ..."""
    return np.arange(factor - 1, n_times, factor)


def phase_labels(t_s: np.ndarray, events: dict[str, float]) -> np.ndarray:
    """int8 phase index per sample from event times (window time, s). See :data:`PHASES`."""
    bounds = np.array([events[e] for e in PHASE_END_EVENTS])
    return np.searchsorted(bounds, t_s, side="right").astype(np.int8)


def _subject_ratios(subject: str, cfg: dict, raw_dir: Path) -> tuple[np.ndarray, list[str]]:
    bc = cfg["bad_channels"]
    ratios, names = [], None
    for k in cfg["series"]:
        s = load_series(subject, k, raw_dir)
        ratios.append(amplitude_ratio(s.eeg, s.eeg_fs_hz, bc["highpass_hz"], bc["order"]))
        names = s.eeg_names
    return np.array(ratios), names


def _process_series(subject: str, series: str, cfg: dict, raw_dir: Path, bads: list[str],
                    lifts: pd.DataFrame) -> tuple[dict[str, list[np.ndarray]], list[dict], dict[str, float], list[str]]:
    """Process one series. Returns per-array per-lift chunks, trial rows, settle times, EMG names."""
    hs = load_series(subject, series, raw_dir)
    trials: list[Trial] = load_trials(subject, series, raw_dir, lifts=lifts)
    fs = hs.eeg_fs_hz
    factor = int(round(fs / cfg["fs_out_hz"]))
    emg_factor = int(round(hs.emg_fs_hz / fs))
    eeg_cfg = cfg["eeg"]

    eeg = interpolate_bad_channels(hs.eeg.astype(np.float64), hs.eeg_names, fs, bads)
    aa_sos = butter_sos(fs, eeg_cfg["antialias"]["order"],
                        high_hz=eeg_cfg["antialias"]["frac_of_nyquist"] * cfg["fs_out_hz"] / 2)
    probe = int(min(cfg["filter_transient"]["probe_start_s"], eeg.shape[1] / fs / 2) * fs)

    streams: dict[str, np.ndarray] = {}
    settle: dict[str, float] = {}
    for band, (lo, hi) in eeg_cfg["bands_hz"].items():
        sos = butter_sos(fs, eeg_cfg["filter_order"], lo, hi)
        settle[band] = settle_time_s(eeg, fs, sos, probe, cfg["filter_transient"]["tolerance"])
        filtered = apply_sos(eeg, sos, causal=True)
        for ref in eeg_cfg["references"]:
            x = common_average(filtered) if ref == "car" else filtered
            streams[f"eeg_{band}_{ref}"] = apply_sos(x, aa_sos, causal=True)
        del filtered

    kc = cfg["kinematics"]
    if not np.isfinite(hs.kin_pos).all():
        raise ValueError(f"{subject} {series}: non-finite HS positions; zero-phase smoothing would spread them")
    kin = smooth_position(hs.kin_pos.astype(np.float64), fs, kc["smooth_cutoff_hz"], kc["smooth_order"])
    emg_cfg = cfg["emg_envelope"]
    emg_env = apply_sos(np.abs(hs.emg.astype(np.float64)),
                        butter_sos(hs.emg_fs_hz, emg_cfg["order"], high_hz=emg_cfg["lowpass_hz"]), causal=True)

    seg = cfg["segment"]
    oc = cfg["onset"]
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in [*streams, "kin_pos", "wrist_vel", "emg_env", "phase"]}
    rows = []
    for tr in trials:
        s0, n = tr.hs_start_sample, tr.n_times
        keep = keep_indices(n, factor)
        t_keep = (keep + 1) / fs
        for name, x in streams.items():
            chunks[name].append(x[:, s0:s0 + n][:, keep].astype(np.float32))
        pos = kin[:, :, s0:s0 + n]
        chunks["kin_pos"].append(pos[:, :, keep].astype(np.float32))
        chunks["wrist_vel"].append(velocity(pos[WRIST - 1], fs)[:, keep].astype(np.float32))
        chunks["emg_env"].append(emg_env[:, emg_factor * (s0 + keep + 1) - 1].astype(np.float32))

        t_raw = (np.arange(n) + 1) / fs
        onset, t_peak, v_peak = kinematic_onset(
            t_raw, tr.kin_pos[WRIST - 1].T.astype(np.float64), fs, tr.events["LEDOn"], tr.events["tFirstDigitTouch"],
            oc["threshold_frac"], kc["smooth_cutoff_hz"], kc["smooth_order"])
        events = {**tr.events, "onset": onset}
        chunks["phase"].append(phase_labels(t_keep, events))

        first_used_series_s = tr.t0_series_s + onset - seg["pre_onset_margin_s"] - seg["history_s"]
        rows.append({
            "subject": subject, "series": tr.series, "lift": tr.lift, "block_type": tr.block_type,
            "weight_g": tr.weight_g, "surface": tr.surface,
            "prev_weight_g": tr.prev_weight_g, "prev_surface": tr.prev_surface,
            "n_samples": len(keep), "t0_series_s": tr.t0_series_s, "hs_start_sample": s0,
            "onset_s": onset, "onset_peak_s": t_peak, "onset_peak_speed": v_peak,
            "onset_minus_tHandStart_s": onset - tr.events["tHandStart"],
            "onset_mismatch": bool(np.isnan(onset) or abs(onset - tr.events["tHandStart"]) > oc["mismatch_s"]),
            "segment_start_s": onset - seg["pre_onset_margin_s"], "segment_end_s": tr.events["tHandStop"],
            **{f"filter_transient_{b}": bool(first_used_series_s < t) for b, t in settle.items()},
            **{f"ev_{k}": v for k, v in tr.events.items()},
        })
    return chunks, rows, settle, trials[0].emg_names


def build_subject(subject: str, cfg: dict, raw_dir: Path, out_dir: Path, git_info: dict | None = None) -> dict:
    """Build the cache for one subject into ``out_dir / subject``. Returns the manifest."""
    t_start = time.time()
    subj_dir = out_dir / subject
    subj_dir.mkdir(parents=True, exist_ok=False)
    lifts = raw_io.load_alllifts(subject, raw_dir)

    ratios, eeg_names = _subject_ratios(subject, cfg, raw_dir)
    bc = cfg["bad_channels"]
    bads = detect_bad_channels(ratios, eeg_names, bc["flat_ratio"], bc["noisy_ratio"])
    logger.info("%s bad channels: %s", subject, bads)

    all_chunks: dict[str, list[np.ndarray]] = {}
    all_rows: list[dict] = []
    settle_times: dict[str, dict[str, float]] = {}
    emg_names: list[str] = []
    for k in cfg["series"]:
        chunks, rows, settle, emg_names = _process_series(subject, k, cfg, raw_dir, bads, lifts)
        for name, parts in chunks.items():
            all_chunks.setdefault(name, []).extend(parts)
        all_rows.extend(rows)
        settle_times[k] = settle
        logger.info("%s %s: %d lifts, settle times %s s", subject, k, len(rows),
                    {b: round(t, 2) for b, t in settle.items()})

    trials = pd.DataFrame(all_rows)
    offsets = np.concatenate([[0], np.cumsum(trials.n_samples.to_numpy())])
    trials["offset_start"], trials["offset_end"] = offsets[:-1], offsets[1:]
    trials.to_csv(subj_dir / "trials.csv", index=False)

    shapes = {}
    for name, parts in all_chunks.items():
        arr = np.concatenate(parts, axis=-1)
        assert arr.shape[-1] == offsets[-1], name
        np.save(subj_dir / f"{name}.npy", arr)
        shapes[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    manifest = {
        "subject": subject,
        "config_hash": config_hash(cfg),
        "fs_hz": cfg["fs_out_hz"],
        "n_trials": len(trials),
        "n_samples": int(offsets[-1]),
        "eeg_names": eeg_names,
        "emg_names": emg_names,
        "bad_channels": bads,
        "bad_channel_ratio_median": dict(zip(eeg_names, np.round(np.median(ratios, axis=0), 4).tolist())),
        "phases": list(PHASES),
        "filter_settle_time_s": settle_times,
        "arrays": shapes,
        "n_onset_mismatch": int(trials.onset_mismatch.sum()),
        "n_filter_transient": {b: int(trials[f"filter_transient_{b}"].sum()) for b in cfg["eeg"]["bands_hz"]},
        "build_seconds": round(time.time() - t_start, 1),
        **({"git": git_info} if git_info else {}),
    }
    (subj_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_cache_config(cfg: dict, out_dir: Path) -> None:
    """Write (or check) ``out_dir/config.yaml`` with the hashed part of the config."""
    hashed = {k: v for k, v in cfg.items() if k not in UNHASHED_KEYS}
    path = out_dir / "config.yaml"
    if path.exists():
        if yaml.safe_load(path.read_text(encoding="utf-8")) != hashed:
            raise RuntimeError(f"{path} differs from the current config despite the same hash")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(hashed, sort_keys=False), encoding="utf-8")


def load_subject(cache_dir: Path, subject: str, arrays: list[str] | None = None,
                 mmap: bool = True) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Load a subject's cached arrays (memory-mapped by default), trials table, and manifest."""
    subj_dir = Path(cache_dir) / subject
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    names = arrays or list(manifest["arrays"])
    data = {n: np.load(subj_dir / f"{n}.npy", mmap_mode="r" if mmap else None) for n in names}
    trials = pd.read_csv(subj_dir / "trials.csv")
    return data, trials, manifest
