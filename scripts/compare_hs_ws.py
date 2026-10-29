"""Check how WS windows map onto HS continuous data, and summarize the HS ST file.

For each WS window, search a small range of sample offsets around
``round(trial_start_time * fs)`` in the HS series and report the offset at which
EEG, kinematics, and EMG match, plus the maximum absolute difference there.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_hs_ws.py --subjects P1 P2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

RAW_DIR = Path("dataset/raw")
OUT_DIR = Path("output/data-baselines/data-inspection")
SEARCH_SAMPLES = 10  # offsets searched on each side of the nominal start sample
N_HS_KIN = 36  # HS kin has the first 36 WS kin columns
WELCH_SEGMENT_S = 2.0


def best_offset(ws_sig: np.ndarray, hs_sig: np.ndarray, nominal: int, search: int) -> tuple[int, float]:
    """Return (offset, max_abs_diff) minimizing the max abs difference of ws_sig vs hs_sig[nominal+offset:].

    Windows that run past the end of the HS recording are compared on their overlapping part.
    """
    best = (0, np.inf)
    for off in range(-search, search + 1):
        start = nominal + off
        n = min(ws_sig.shape[0], hs_sig.shape[0] - start)
        if start < 0 or n <= 0:
            continue
        diff = np.nanmax(np.abs(hs_sig[start:start + n] - ws_sig[:n]))
        if diff < best[1]:
            best = (off, float(diff))
    return best


def compare_series(subject: str, series: str) -> list[str]:
    ws = scipy.io.loadmat(RAW_DIR / subject / f"WS_{subject}_{series}.mat", simplify_cells=True)["ws"]
    hs = scipy.io.loadmat(RAW_DIR / subject / f"HS_{subject}_{series}.mat", simplify_cells=True)["hs"]
    fs_eeg, fs_emg = hs["eeg"]["samplingrate"], hs["emg"]["samplingrate"]
    n_hs = hs["eeg"]["sig"].shape[0]

    # Nominal window start in HS samples: StartTime is the LED-on time in series time, and each
    # window begins LEDon (= 2 s) earlier. Found with a wide search during inspection; the narrow
    # search below confirms it per window.
    offsets_eeg, offsets_emg, diffs = [], [], {"eeg": [], "kin": [], "emg": []}
    n_truncated, overrun_s, starts, ends = 0, [], [], []
    for w in ws["win"]:
        t0 = w["trial_start_time"] - w["LEDon"]
        nominal = int(round(t0 * fs_eeg))
        off, d = best_offset(w["eeg"], hs["eeg"]["sig"], nominal, SEARCH_SAMPLES)
        offsets_eeg.append(off)
        diffs["eeg"].append(d)
        start = nominal + off
        n = min(len(w["eeg"]), n_hs - start)
        if n < len(w["eeg"]):
            n_truncated += 1
            overrun_s.append((len(w["eeg"]) - n) / fs_eeg)
        starts.append(start)
        ends.append(start + len(w["eeg"]))
        kin_ws = w["kin"][:n, :N_HS_KIN]
        diffs["kin"].append(float(np.nanmax(np.abs(hs["kin"]["sig"][start:start + n] - kin_ws))))
        nominal_emg = int(round(t0 * fs_emg))
        off_e, d_e = best_offset(w["emg"], hs["emg"]["sig"], nominal_emg, SEARCH_SAMPLES * 8)
        offsets_emg.append(off_e)
        diffs["emg"].append(d_e)

    # Fraction of HS samples covered by at least one window.
    mask = np.zeros(n_hs, dtype=bool)
    for s, e in zip(starts, ends):
        mask[s:min(e, n_hs)] = True
    lines = [
        f"{subject} {series}: HS {n_hs} samples ({n_hs / fs_eeg:.1f} s), {len(ws['win'])} windows",
        f"  window start in HS samples = round((StartTime - LEDon) * fs) + offset, offsets found: "
        f"EEG {sorted(set(offsets_eeg))}, EMG {sorted(set(offsets_emg))}",
        f"  max |WS - HS| at that offset: EEG={max(diffs['eeg']):.3g}, kin(36 cols)={max(diffs['kin']):.3g}, "
        f"EMG={max(diffs['emg']):.3g}",
        f"  first window starts at HS sample {starts[0]} ({starts[0] / fs_eeg:.3f} s); "
        f"last window ends at sample {ends[-1]} ({ends[-1] / fs_eeg:.3f} s)",
        f"  windows running past HS end: {n_truncated} (overrun s: {[round(x, 3) for x in overrun_s]})",
        f"  HS samples covered by >=1 window: {mask.mean():.1%}",
    ]
    return lines


def summarize_st(subject: str) -> list[str]:
    st = scipy.io.loadmat(RAW_DIR / subject / f"HS_{subject}_ST.mat", simplify_cells=True)["hs"]
    hs = scipy.io.loadmat(RAW_DIR / subject / f"HS_{subject}_S1.mat", simplify_cells=True)["hs"]
    names = list(st["eeg"]["names"])
    fs = st["eeg"]["samplingrate"]
    lines = [f"{subject} ST: fields={[k for k in st if k not in ('name',)]}, series={st['series']}, "
             f"EEG {st['eeg']['sig'].shape} at {fs} Hz = {st['eeg']['sig'].shape[0] / fs:.1f} s, "
             f"same channel names as S1: {names == list(hs['eeg']['names'])}"]

    # Compare spectra of ST vs S1 at a few channels: occipital alpha is a resting-state marker.
    nper = int(WELCH_SEGMENT_S * fs)
    for ch in ("Fp1", "C3", "Cz", "O1", "Oz"):
        i = names.index(ch)
        rows = []
        for label, sig in (("ST", st["eeg"]["sig"][:, i]), ("S1", hs["eeg"]["sig"][:, i])):
            f, pxx = scipy.signal.welch(sig - sig.mean(), fs=fs, nperseg=nper)
            band = lambda lo, hi: pxx[(f >= lo) & (f < hi)].mean()
            rows.append(f"{label}: std={sig.std():8.1f} delta(1-4)={band(1, 4):9.3g} "
                        f"alpha(8-13)={band(8, 13):9.3g} beta(13-30)={band(13, 30):9.3g} "
                        f"alpha/beta={band(8, 13) / band(13, 30):5.2f}")
        lines.append(f"  {ch:<4s} " + " | ".join(rows))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", nargs="+", default=["P1"])
    parser.add_argument("--series", nargs="*", default=[f"S{i}" for i in range(1, 10)])
    args = parser.parse_args()

    lines: list[str] = []
    for subject in args.subjects:
        lines.append(f"=== HS vs WS: {subject}")
        for series in args.series:
            lines.extend(compare_series(subject, series))
            print(f"done {subject} {series}", flush=True)
        lines.append(f"=== ST: {subject}")
        lines.extend(summarize_st(subject))
        lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"compare_hs_ws_{'_'.join(args.subjects)}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
