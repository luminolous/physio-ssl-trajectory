"""Event-locked EEG analysis for the neurophysiology sanity check. No training, no labels fitted.

Time convention (as in the cache): lift sample j is at window time (j + 1) / fs s. Negative j are the
samples before the lift, from the pre-lift arrays. An event at t s maps to the nearest sample.

The MRCP uses causal filters, the signal the decoders see, so its timing includes the filter delay
(:func:`filter_delay_ms`). Morlet power is not causal, which is acceptable for analysis only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats

logger = logging.getLogger(__name__)

EVENT_COLUMNS = {"onset": "onset_s", "led": "ev_LEDOn", "tHandStop": "ev_tHandStop"}


def event_sample(t_s: float, fs_hz: float) -> int:
    """Lift sample j nearest to window time ``t_s``. Sample j is at (j + 1) / fs_hz."""
    return int(round(t_s * fs_hz)) - 1


def lift_stream(eeg: np.ndarray, pre: np.ndarray, row: int, start: int, end: int) -> np.ndarray:
    """One lift's EEG with the samples before it.

    Args:
        eeg: (C, T), lifts back to back. pre: (C, n_lifts, n_pre), oldest first.
        row: lift row. start, end: its offsets in ``eeg``.

    Returns:
        float32 (C, n_pre + end - start). Lift sample j is at index n_pre + j.
    """
    return np.concatenate([pre[:, row], eeg[:, start:end]], axis=1).astype(np.float32)


def epoch_stream(stream: np.ndarray, n_pre: int, event_j: int, k0: int, k1: int) -> np.ndarray:
    """Samples event_j + k for k = k0, ..., k1 of a stream from :func:`lift_stream`, NaN where absent.

    Returns:
        float32 (C, k1 - k0 + 1).
    """
    out = np.full((stream.shape[0], k1 - k0 + 1), np.nan, dtype=np.float32)
    idx = n_pre + event_j + np.arange(k0, k1 + 1)
    ok = (idx >= 0) & (idx < stream.shape[1])
    out[:, ok] = stream[:, idx[ok]]
    return out


def sample_range(window_s: tuple[float, float] | list[float], fs_hz: float) -> tuple[int, int]:
    """First and last sample offset k of a window relative to an event."""
    return int(round(window_s[0] * fs_hz)), int(round(window_s[1] * fs_hz))


def relative_times(k0: int, k1: int, fs_hz: float) -> np.ndarray:
    """Times (s) of offsets k0, ..., k1 relative to an event."""
    return np.arange(k0, k1 + 1) / fs_hz


def transient_mask(trials: pd.DataFrame, settle_s: dict[str, float], earliest_s: np.ndarray) -> np.ndarray:
    """True for lifts whose earliest used sample falls inside the filter start-up transient.

    Args:
        trials: needs ``series`` and ``t0_series_s``.
        settle_s: settle time per series key, for example ``{"S1": 13.1}``.
        earliest_s: (n_lifts,) earliest used window time of each lift.
    """
    settle = trials.series.map(lambda k: settle_s[f"S{k}"]).to_numpy()
    return trials.t0_series_s.to_numpy() + earliest_s < settle


def settle_times(cache_dir: str | Path, subject: str, band: str, reference: str) -> dict[str, float]:
    """Settle time per series from the manifest, or from the band variant's file."""
    subj_dir = Path(cache_dir) / subject
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    per_series = manifest["filter_settle_time_s"]
    if band in next(iter(per_series.values())):
        return {k: v[band] for k, v in per_series.items()}
    info = json.loads((subj_dir / f"variant_{band}_{reference}.json").read_text(encoding="utf-8"))
    return info["filter_settle_time_s"]


def load_band(cache_dir: str | Path, subject: str, band: str, reference: str
              ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], float]:
    """Memory-mapped eeg (C, T) and pre (C, n_lifts, n_pre), trials, channel names, and rate."""
    subj_dir = Path(cache_dir) / subject
    manifest = json.loads((subj_dir / "manifest.json").read_text(encoding="utf-8"))
    eeg = np.load(subj_dir / f"eeg_{band}_{reference}.npy", mmap_mode="r")
    pre = np.load(subj_dir / f"pre_lift_eeg_{band}_{reference}.npy", mmap_mode="r")
    return eeg, pre, pd.read_csv(subj_dir / "trials.csv"), manifest["eeg_names"], float(manifest["fs_hz"])


def erp_epochs(eeg: np.ndarray, pre: np.ndarray, trials: pd.DataFrame, fs_hz: float, epochs_s: dict[str, list[float]],
               baseline_led_s: list[float]) -> dict[str, np.ndarray]:
    """Baseline-corrected epochs per event.

    The baseline of a lift is its mean over ``baseline_led_s`` relative to LED-on, for every alignment.
    A baseline relative to onset would move with the reaction time.

    Returns:
        event name -> float32 (n_lifts, C, n_times), NaN where no EEG exists.
    """
    n_pre = pre.shape[2]
    b0, b1 = sample_range(baseline_led_s, fs_hz)
    ranges = {e: sample_range(w, fs_hz) for e, w in epochs_s.items()}
    out = {e: np.full((len(trials), eeg.shape[0], k1 - k0 + 1), np.nan, dtype=np.float32)
           for e, (k0, k1) in ranges.items()}
    for row, tr in enumerate(trials.itertuples(index=False)):
        stream = lift_stream(eeg, pre, row, int(tr.offset_start), int(tr.offset_end))
        base = epoch_stream(stream, n_pre, event_sample(tr.ev_LEDOn, fs_hz), b0, b1).mean(axis=1, keepdims=True)
        for e, (k0, k1) in ranges.items():
            out[e][row] = epoch_stream(stream, n_pre, event_sample(getattr(tr, EVENT_COLUMNS[e]), fs_hz), k0, k1) - base
    return out


def earliest_used_s(trials: pd.DataFrame, epochs_s: dict[str, list[float]], baseline_led_s: list[float]) -> np.ndarray:
    """Earliest window time (s) each lift uses, over all epochs and the baseline."""
    starts = [trials[EVENT_COLUMNS[e]].to_numpy() + w[0] for e, w in epochs_s.items()]
    starts.append(trials.ev_LEDOn.to_numpy() + baseline_led_s[0])
    return np.min(starts, axis=0)


def rt_thirds(rt_s: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the fastest and slowest ``fraction`` of lifts. NaN reaction times are in neither."""
    lo, hi = np.nanquantile(rt_s, [fraction, 1 - fraction])
    return rt_s <= lo, rt_s >= hi


def peak_latency_s(trace: np.ndarray, times_s: np.ndarray, window_s: list[float]) -> float:
    """Time of the most negative value of ``trace`` (n_times,) inside ``window_s``."""
    inside = np.flatnonzero((times_s >= window_s[0]) & (times_s <= window_s[1]))
    return float(times_s[inside[np.nanargmin(trace[inside])]])


def window_mean(x: np.ndarray, times_s: np.ndarray, window_s: list[float]) -> np.ndarray:
    """NaN-aware mean of ``x`` (..., n_times) over ``window_s``."""
    inside = (times_s >= window_s[0]) & (times_s <= window_s[1])
    return np.nanmean(x[..., inside], axis=-1)


def freq_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Frequencies start, start + step, ..., stop (Hz)."""
    return np.arange(start, stop + step / 2, step)


def tf_event_locked(eeg: np.ndarray, pre: np.ndarray, trials: pd.DataFrame, fs_hz: float, freqs_hz: np.ndarray,
                    n_cycles_per_hz: float, edge_s: float, epochs_s: dict[str, list[float]],
                    baseline_led_s: list[float], min_lift_fraction: float) -> dict[str, Any]:
    """Mean Morlet power locked to each event, and the mean power of the LED baseline.

    Power is computed per lift on its available EEG, so variable lift lengths need no padding. Power
    within ``edge_s`` of either end of that EEG is dropped.

    Returns:
        ``power``: event -> float64 (C, F, n_times), NaN where fewer than ``min_lift_fraction`` of the
        lifts cover a time. ``baseline``: (C, F). ``coverage``: event -> (n_times,) lift counts.
    """
    n_pre = pre.shape[2]
    edge = int(round(edge_s * fs_hz))
    b0, b1 = sample_range(baseline_led_s, fs_hz)
    ranges = {e: sample_range(w, fs_hz) for e, w in epochs_s.items()}
    shape = (eeg.shape[0], len(freqs_hz))
    total = {e: np.zeros((*shape, k1 - k0 + 1)) for e, (k0, k1) in ranges.items()}
    count = {e: np.zeros(k1 - k0 + 1) for e, (k0, k1) in ranges.items()}
    base_total, base_n = np.zeros(shape), 0
    for row, tr in zip(trials.index, trials.itertuples(index=False)):  # the index is the row in ``pre``
        stream = lift_stream(eeg, pre, row, int(tr.offset_start), int(tr.offset_end))
        power = mne.time_frequency.tfr_array_morlet(stream[None].astype(np.float64), fs_hz, freqs_hz,
                                                    n_cycles=n_cycles_per_hz * freqs_hz, output="power",
                                                    verbose="error")[0]
        n_times = power.shape[-1]
        idx = n_pre + event_sample(tr.ev_LEDOn, fs_hz) + np.arange(b0, b1 + 1)
        if idx[0] >= edge and idx[-1] < n_times - edge:
            base_total += power[..., idx].mean(axis=-1)
            base_n += 1
        for e, (k0, k1) in ranges.items():
            idx = n_pre + event_sample(getattr(tr, EVENT_COLUMNS[e]), fs_hz) + np.arange(k0, k1 + 1)
            ok = (idx >= edge) & (idx < n_times - edge)
            total[e][..., ok] += power[..., idx[ok]]
            count[e][ok] += 1
    power_mean = {}
    for e in ranges:
        with np.errstate(invalid="ignore", divide="ignore"):
            m = total[e] / count[e]
        m[..., count[e] < min_lift_fraction * len(trials)] = np.nan
        power_mean[e] = m
    return {"power": power_mean, "baseline": base_total / base_n, "coverage": count, "n_baseline_lifts": base_n}


def percent_change(power: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """100 x (power - baseline) / baseline. Negative is ERD, positive is ERS.

    Takes mean power and mean baseline, not a per-lift ratio, whose noisy denominator would bias it upward.

    Args:
        power: (C, F, n_times). baseline: (C, F).
    """
    return 100 * (power - baseline[..., None]) / baseline[..., None]


def band_window_change(power: np.ndarray, baseline: np.ndarray, freqs_hz: np.ndarray, times_s: np.ndarray,
                       band_hz: list[float], window_s: list[float]) -> np.ndarray:
    """Percent power change per channel (C,) in a frequency band and time window."""
    f = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1])
    in_window = window_mean(power[:, f], times_s, window_s).sum(axis=1)
    base = baseline[:, f].sum(axis=1)
    return 100 * (in_window - base) / base


def paired_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Paired summary across subjects of ``left - right``, with an exact two-sided Wilcoxon test."""
    d = np.asarray(left) - np.asarray(right)
    return {"left_mean": float(np.mean(left)), "right_mean": float(np.mean(right)), "diff_mean": float(d.mean()),
            "diff_sd": float(d.std(ddof=1)), "n_negative": int((d < 0).sum()), "n_subjects": int(d.size),
            "wilcoxon_p": float(scipy.stats.wilcoxon(d, method="exact").pvalue)}


def filter_delay_ms(sos_chain: list[np.ndarray], fs_hz: float, freqs_hz: list[float]) -> dict[float, float]:
    """Group delay (ms) of a chain of causal filters at the given frequencies."""
    total = {f: 0.0 for f in freqs_hz}
    for sos in sos_chain:
        w, h = scipy.signal.freqz_sos(sos, worN=2 ** 20, fs=fs_hz)
        delay_s = -np.gradient(np.unwrap(np.angle(h)), 2 * np.pi * w)
        for f in freqs_hz:
            total[f] += 1000 * float(np.interp(f, w, delay_s))
    return total


def lift_pair_correlations(epochs: dict[str, np.ndarray], times: dict[str, np.ndarray], names: list[str], keep: np.ndarray,
                           asym: dict[str, Any]) -> dict[str, float]:
    """Across one subject's lifts, r between the reference-pair difference and the ocular-pair difference, per window."""
    ref, ocular = asym["reference_pair"], asym["ocular_pair"]
    out = {}
    for name, (event, window) in asym["windows_s"].items():
        values = window_mean(epochs[event][keep], times[event], window)  # (n_lifts, C)
        out[name] = float(np.corrcoef(values[:, names.index(ref[0])] - values[:, names.index(ref[1])],
                                      values[:, names.index(ocular[0])] - values[:, names.index(ocular[1])])[0, 1])
    return out


def original_reference_asymmetry(cache_dir: str | Path, subject: str, trials: pd.DataFrame, fs_hz: float, keep: np.ndarray,
                                 names: list[str], asym: dict[str, Any], baseline_led_s: list[float]) -> dict[str, float]:
    """Reference-pair and ocular-pair differences without CAR, which would spread frontal activity to all channels.

    Reads the cached original-reference band in ``asym["orig_band"]``, on lift windows only, since that band has no
    pre-lift samples. Lifts whose window lacks samples are ignored through NaN-aware means.
    """
    eeg = np.load(Path(cache_dir) / subject / f"eeg_{asym['orig_band']}_orig.npy", mmap_mode="r")
    no_pre = np.zeros((eeg.shape[0], len(trials), 0), dtype=np.float32)
    event, window = asym["windows_s"][asym["orig_window"]]
    k0, k1 = sample_range(window, fs_hz)
    x = erp_epochs(eeg, no_pre, trials, fs_hz, {event: window}, baseline_led_s)[event][keep]
    means = np.nanmean(window_mean(x, relative_times(k0, k1, fs_hz), window), axis=0)
    return {f"{a}-{b}": float(means[names.index(a)] - means[names.index(b)])
            for a, b in (asym["reference_pair"], asym["ocular_pair"])}


def asymmetry_table(results: list[dict[str, Any]], asym: dict[str, Any]) -> pd.DataFrame:
    """Left minus right per subject, homologous pair, and window, from the averages in ``results``.

    Returns:
        Columns subject, window, pair, diff_uv.
    """
    rows = []
    for r in results:
        i = r["names"].index
        for name, (event, window) in asym["windows_s"].items():
            means = window_mean(r["erp"][event], r["erp_times"][event], window)
            rows += [{"subject": r["subject"], "window": name, "pair": f"{a}-{b}", "diff_uv": float(means[i(a)] - means[i(b)])}
                     for a, b in asym["pairs"]]
    return pd.DataFrame(rows)


def analyze_subject(cfg: dict[str, Any], subject: str) -> dict[str, Any]:
    """Event-locked averages and power of one subject, on lifts outside the filter transient.

    Args:
        cfg: ``configs/diagnostics/neurophysiology.yaml``.

    Returns:
        A dict with ``erp`` event -> (C, n_times) slow-band average, ``vep`` (C, n_times) fast-band
        LED-locked average, ``split`` event -> {"fast", "slow"} (C, n_times), ``tf_power`` event ->
        (C, F, n_times), ``tf_baseline`` (C, F), and the times, counts, and reaction times behind them.
    """
    erp_cfg, tf_cfg = cfg["erp"], cfg["tf"]
    cache, ref = cfg["cache_dir"], cfg["reference"]
    eeg, pre, trials, names, fs = load_band(cache, subject, erp_cfg["band"], ref)
    fast_eeg, fast_pre, _, _, _ = load_band(cache, subject, erp_cfg["fast_band"], ref)
    tf_eeg, tf_pre, _, _, _ = load_band(cache, subject, tf_cfg["band"], ref)

    earliest = np.minimum(earliest_used_s(trials, erp_cfg["epochs_s"], erp_cfg["baseline_led_s"]),
                          earliest_used_s(trials, tf_cfg["epochs_s"], tf_cfg["baseline_led_s"]))
    flagged = np.zeros(len(trials), dtype=bool)
    for band in {erp_cfg["band"], erp_cfg["fast_band"], tf_cfg["band"]}:
        flagged |= transient_mask(trials, settle_times(cache, subject, band, ref), earliest)
    keep = ~flagged
    rt = (trials.onset_s - trials.ev_LEDOn).to_numpy()

    epochs = erp_epochs(eeg, pre, trials, fs, erp_cfg["epochs_s"], erp_cfg["baseline_led_s"])
    fast, slow = rt_thirds(np.where(keep, rt, np.nan), cfg["rt_split"]["fraction"])
    erp = {e: np.nanmean(x[keep], axis=0) for e, x in epochs.items()}
    split = {e: {"fast": np.nanmean(x[fast], axis=0), "slow": np.nanmean(x[slow], axis=0)} for e, x in epochs.items()}
    times = {e: relative_times(*sample_range(w, fs), fs) for e, w in erp_cfg["epochs_s"].items()}
    within = lift_pair_correlations(epochs, times, names, keep, cfg["asymmetry"])
    del epochs
    orig = original_reference_asymmetry(cache, subject, trials, fs, keep, names, cfg["asymmetry"], erp_cfg["baseline_led_s"])
    led_only = {"led": erp_cfg["epochs_s"]["led"]}
    vep = np.nanmean(erp_epochs(fast_eeg, fast_pre, trials, fs, led_only, erp_cfg["baseline_led_s"])["led"][keep], axis=0)

    freqs = freq_grid(**{k: tf_cfg["freqs_hz"][k] for k in ("start", "stop", "step")})
    tf = tf_event_locked(tf_eeg, tf_pre, trials[keep], fs, freqs, tf_cfg["n_cycles_per_hz"], tf_cfg["edge_s"],
                         tf_cfg["epochs_s"], tf_cfg["baseline_led_s"], tf_cfg["min_lift_fraction"])
    return {"subject": subject, "names": names, "fs_hz": fs, "n_lifts": len(trials), "n_kept": int(keep.sum()),
            "rt_median_s": float(np.median(rt[keep])), "rt_iqr_s": float(np.subtract(*np.percentile(rt[keep], [75, 25]))),
            "rt_fast_median_s": float(np.median(rt[fast])), "rt_slow_median_s": float(np.median(rt[slow])),
            "erp": erp, "vep": vep, "split": split, "freqs_hz": freqs, "lift_pair_r": within, "orig_asymmetry": orig,
            "tf_power": tf["power"], "tf_baseline": tf["baseline"], "tf_coverage": tf["coverage"],
            "erp_times": {e: relative_times(*sample_range(w, fs), fs) for e, w in erp_cfg["epochs_s"].items()},
            "tf_times": {e: relative_times(*sample_range(w, fs), fs) for e, w in tf_cfg["epochs_s"].items()}}


def laterality_table(results: list[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per subject: MRCP amplitude and mu/beta power change on each side.

    Columns are ``<measure>_<side>`` with side ``C3``, ``C4``, ``contra``, or ``ipsi``. "contra" is the left
    hemisphere, which is contralateral only if the right hand moved. MRCP values are in microvolts.
    """
    lat, win = cfg["laterality"], cfg["windows_s"]
    sides = {lat["pair"][0]: [lat["pair"][0]], lat["pair"][1]: [lat["pair"][1]],
             "contra": lat["contralateral"], "ipsi": lat["ipsilateral"]}
    rows = []
    for r in results:
        idx = {side: [r["names"].index(c) for c in chs] for side, chs in sides.items()}
        row: dict[str, Any] = {"subject": r["subject"]}
        measures = {"mrcp": window_mean(r["erp"]["onset"], r["erp_times"]["onset"], win["mrcp"])}
        for band, band_hz in cfg["bands_hz"].items():
            for name, event, window in (("pre", "onset", win["erd_pre"]), ("move", "onset", win["erd_move"]),
                                        ("rebound", "tHandStop", cfg["rebound_s"])):
                measures[f"{band}_{name}"] = band_window_change(r["tf_power"][event], r["tf_baseline"], r["freqs_hz"],
                                                                r["tf_times"][event], band_hz, window)
        for measure, values in measures.items():
            for side, chs in idx.items():
                row[f"{measure}_{side}"] = float(np.mean(values[chs]))
        rows.append(row)
    return pd.DataFrame(rows)


def laterality_stats(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Paired tests across subjects, C3 vs C4 and left vs right cluster, for every measure in ``table``.

    A negative difference means more negative on the left: a larger negativity for the MRCP, a stronger
    power decrease for mu and beta. Both are the direction expected from right-hand movement.
    """
    a, b = cfg["laterality"]["pair"]
    measures = sorted({c.rsplit("_", 1)[0] for c in table.columns if c != "subject"})
    rows = []
    for m in measures:
        for label, left, right in ((f"{a}-{b}", a, b), ("contra-ipsi", "contra", "ipsi")):
            rows.append({"measure": m, "contrast": label,
                         **paired_summary(table[f"{m}_{left}"].to_numpy(), table[f"{m}_{right}"].to_numpy())})
    return pd.DataFrame(rows)


def rt_split_table(results: list[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    """Peak latency of fast and slow lifts per subject, channel, and alignment.

    ``latency_shift_s`` is slow minus fast. Under LED alignment, a component tied to the movement shifts by
    about ``rt_shift_s``, and one tied to the cue shifts by about 0. Under onset alignment it is the reverse.
    """
    rows = []
    for r in results:
        for ch in cfg["rt_split"]["channels"]:
            c = r["names"].index(ch)
            for event, window in cfg["rt_split"]["peak_window_s"].items():
                t = r["erp_times"][event]
                fast = peak_latency_s(r["split"][event]["fast"][c], t, window)
                slow = peak_latency_s(r["split"][event]["slow"][c], t, window)
                rows.append({"subject": r["subject"], "channel": ch, "alignment": event, "latency_fast_s": fast,
                             "latency_slow_s": slow, "latency_shift_s": slow - fast,
                             "rt_shift_s": r["rt_slow_median_s"] - r["rt_fast_median_s"]})
    return pd.DataFrame(rows)
