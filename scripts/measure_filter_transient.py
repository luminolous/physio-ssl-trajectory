"""Measure the start-up transient of the causal EEG band-pass filters on HS data.

A causal filter started at sample s deviates from the output of the same filter
that has already been running (started at sample 0) until its state settles.
Transient length = time after s until that deviation stays below a tolerance
(fraction of the steady-state output SD), worst channel. Measured with zero
initial state and with ``sosfilt_zi`` scaled by the first sample.

Then counts, for all subjects (AllLifts), the lifts whose first used EEG sample
(onset - margin - history, onset approximated by tHandStart) falls inside the
transient measured from the HS start.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\measure_filter_transient.py --subject P1 --series S1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal

from eegtraj.data import raw_io

OUT_DIR = Path("output/data-baselines/data-inspection")
BANDS_HZ = {"lowfreq_0.1-4": (0.1, 4.0), "broadband_0.1-40": (0.1, 40.0)}
FILTER_ORDER = 2  # Butterworth order per band edge (sos band-pass has 2*order poles)
START_OFFSET_S = 120.0  # late start used for the measurement; the reference has run this long already
TOLERANCES = (0.01, 0.05, 0.10)  # deviation below this fraction of the steady-state SD counts as settled
FLAG_TOLERANCE = 0.01  # tolerance used for flagging lifts
PRE_ONSET_MARGIN_S = 0.5  # D2
HISTORY_S = 0.5  # D3


def settle_time_s(y_ref: np.ndarray, y_start: np.ndarray, fs: float, tolerance: float) -> float:
    """Last time (s, from the start) where |y_start - y_ref| exceeds tolerance * SD(y_ref), worst channel."""
    tol = tolerance * y_ref.std(axis=0)
    exceed = np.abs(y_start - y_ref) > tol
    last = np.flatnonzero(exceed.any(axis=1))
    return (last[-1] + 1) / fs if last.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="P1")
    parser.add_argument("--series", default="S1")
    args = parser.parse_args()

    hs = raw_io.load_hs(args.subject, args.series)
    x = hs["eeg"]["sig"].astype(np.float64)
    fs = float(hs["eeg"]["samplingrate"])
    s = int(START_OFFSET_S * fs)
    lines = [f"Filter transient on {args.subject} {args.series} HS EEG ({x.shape[0] / fs:.1f} s, {fs:.0f} Hz), "
             f"causal Butterworth order {FILTER_ORDER}, late start at {START_OFFSET_S:.0f} s, tolerances {[f'{t:.0%}' for t in TOLERANCES]} of SD"]

    transients = {}
    for name, (lo, hi) in BANDS_HZ.items():
        sos = scipy.signal.butter(FILTER_ORDER, [lo, hi], btype="bandpass", fs=fs, output="sos")
        zi = scipy.signal.sosfilt_zi(sos)  # (n_sections, 2), steady state for a unit step
        y_ref = scipy.signal.sosfilt(sos, x, axis=0, zi=zi[:, :, None] * x[0][None, None, :])[0][s:]
        y_zero = scipy.signal.sosfilt(sos, x[s:], axis=0)
        y_zi = scipy.signal.sosfilt(sos, x[s:], axis=0, zi=zi[:, :, None] * x[s][None, None, :])[0]
        # Offset removal variant: subtract the first-second mean before a zero-state start.
        x_off = x[s:] - x[s:s + int(fs)].mean(axis=0)
        y_off = scipy.signal.sosfilt(sos, x_off, axis=0)
        for tol in TOLERANCES:
            t_zero, t_zi, t_off = (settle_time_s(y_ref, y, fs, tol) for y in (y_zero, y_zi, y_off))
            if tol == FLAG_TOLERANCE:
                transients[name] = t_zi
            lines.append(f"  {name} @ {tol:.0%}: settle time zero-state = {t_zero:.1f} s, sosfilt_zi = {t_zi:.1f} s, "
                         f"offset-removed zero-state = {t_off:.1f} s")

    lifts = pd.concat([raw_io.load_alllifts(p) for p in raw_io.subjects()], ignore_index=True)
    first_used_s = lifts.StartTime - lifts.LEDOn + lifts.tHandStart - PRE_ONSET_MARGIN_S - HISTORY_S
    lines.append(f"First used EEG sample per lift (series time, onset ~ tHandStart, margin {PRE_ONSET_MARGIN_S} s + "
                 f"history {HISTORY_S} s): min {first_used_s.min():.2f} s")
    for name, t in transients.items():
        hit = lifts[first_used_s < t]
        lines.append(f"  {name}: lifts with first used sample inside the {t:.1f} s transient: {len(hit)} of {len(lifts)} "
                     f"(Lift numbers: {sorted(hit.Lift.astype(int).unique().tolist())})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"filter_transient_{args.subject}_{args.series}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
