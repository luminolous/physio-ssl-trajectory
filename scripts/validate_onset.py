"""Validate the kinematic movement onset (D2) against the labeled tHandStart.

For every lift: kinematic onset from wrist speed, with the peak searched from
LED-on to first digit touch. Lifts with |onset - tHandStart| > mismatch threshold
are plotted (wrist speed and wrist y with event markers) so the onset can be
checked by eye.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\validate_onset.py --subjects P2 P7
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegtraj.data import raw_io
from eegtraj.preprocessing.onset import kinematic_onset, smooth_kinematics

OUT_DIR = Path("output/data-baselines/data-inspection/onset-validation")
WRIST = 4
FS_KIN_HZ = 500.0
ONSET = dict(threshold_frac=0.05, cutoff_hz=10.0, order=2)  # D2
MISMATCH_S = 0.3  # D2 flag threshold
PANELS_PER_FIG = 20
PANEL_COLS = 4


def compute(subject: str) -> tuple[pd.DataFrame, dict]:
    al = raw_io.load_alllifts(subject)
    rows, traces = [], {}
    for k in raw_io.SERIES:
        ws = raw_io.load_ws(subject, k)
        cols = raw_io.position_columns(ws["names"]["kin"], WRIST)
        sub = al[al.Run == int(k[1:])].reset_index(drop=True)
        for i, w in enumerate(ws["win"]):
            r = sub.iloc[i]
            t, pos = w["eeg_t"], w["kin"][:, cols]
            onset, t_peak, v_peak = kinematic_onset(t, pos, FS_KIN_HZ, r.LEDOn, r.tFirstDigitTouch, **ONSET)
            rows.append({"Subject": subject, "Series": k, "Lift": i + 1, "LEDOn": r.LEDOn,
                         "tHandStart": r.tHandStart, "tFirstDigitTouch": r.tFirstDigitTouch,
                         "onset": onset, "t_peak": t_peak, "peak_speed": v_peak})
            traces[(k, i + 1)] = (t, pos)
    df = pd.DataFrame(rows)
    df["onset_minus_tHandStart"] = df.onset - df.tHandStart
    df["onset_mismatch"] = (df.onset_minus_tHandStart.abs() > MISMATCH_S) | df.onset.isna()
    return df, traces


def plot_mismatches(df: pd.DataFrame, traces: dict, subject: str) -> list[Path]:
    bad = df[df.onset_mismatch].reset_index(drop=True)
    paths = []
    for f in range(math.ceil(len(bad) / PANELS_PER_FIG)):
        chunk = bad.iloc[f * PANELS_PER_FIG:(f + 1) * PANELS_PER_FIG]
        rows = math.ceil(len(chunk) / PANEL_COLS)
        fig, axes = plt.subplots(rows, PANEL_COLS, figsize=(4 * PANEL_COLS, 2.6 * rows), squeeze=False)
        for ax, (_, r) in zip(axes.ravel(), chunk.iterrows()):
            t, pos = traces[(r.Series, r.Lift)]
            smooth = smooth_kinematics(pos, FS_KIN_HZ, ONSET["cutoff_hz"], ONSET["order"])
            speed = np.linalg.norm(np.gradient(smooth, t, axis=0), axis=1)
            ax.plot(t, speed, color="k", lw=0.8, label="wrist speed")
            ax2 = ax.twinx()
            ax2.plot(t, pos[:, 1], color="C0", lw=0.8, alpha=0.7, label="wrist y")
            ax2.tick_params(labelsize=6)
            for x, c, lab in ((r.LEDOn, "C2", "LEDOn"), (r.tHandStart, "C1", "tHandStart"),
                              (r.onset, "C3", "kinematic onset"), (r.tFirstDigitTouch, "C4", "tFirstDigitTouch")):
                if np.isfinite(x):
                    ax.axvline(x, color=c, ls="--", lw=1, label=lab)
            ax.set_title(f"{r.Series} lift {r.Lift}: onset-tHandStart={r.onset_minus_tHandStart:+.2f} s", fontsize=8)
            ax.set_xlim(0, min(t[-1], r.tFirstDigitTouch + 2))
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
        fig.suptitle(f"{subject}: lifts with |kinematic onset - tHandStart| > {MISMATCH_S} s "
                     f"(figure {f + 1}); black = wrist speed, blue = wrist y")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        path = OUT_DIR / f"onset_mismatch_{subject}_{f + 1}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", nargs="+", default=["P2", "P7"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [f"Kinematic onset: peak search LEDOn..tFirstDigitTouch, {ONSET}, mismatch > {MISMATCH_S} s"]
    for subject in args.subjects:
        df, traces = compute(subject)
        df.to_csv(OUT_DIR / f"onset_{subject}.csv", index=False)
        good = df[~df.onset_mismatch].onset_minus_tHandStart
        per_series = df.groupby("Series").onset_mismatch.sum().astype(int).to_dict()
        lines.append(f"{subject}: {int(df.onset_mismatch.sum())} mismatch lifts of {len(df)}, per series {per_series}; "
                     f"NaN onsets {int(df.onset.isna().sum())}; non-mismatch onset - tHandStart: median "
                     f"{good.median() * 1e3:.0f} ms, 5-95% [{good.quantile(0.05) * 1e3:.0f}, {good.quantile(0.95) * 1e3:.0f}] ms")
        mm = df[df.onset_mismatch].onset_minus_tHandStart
        if len(mm):
            lines.append(f"  mismatch onset - tHandStart: min {mm.min():.2f} s, median {mm.median():.2f} s, max {mm.max():.2f} s")
        for p in plot_mismatches(df, traces, subject):
            lines.append(f"  figure: {p}")
    (OUT_DIR / f"summary_{'_'.join(args.subjects)}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
