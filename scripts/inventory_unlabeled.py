"""Unlabeled data inventory: hours of EEG and 2 s window counts per subject, source, and LOSO fold. No training.

Sources per subject: lift windows (the cached lifts), between-lift parts of the HS series S1-S9, and the ST
recording. Window counts are on the 100 Hz grid, inside contiguous segments only (a window never spans two
segments). For LOSO, every fold lists the EEG of its other subjects and of its test subject (usable without labels
for transductive pretraining, like D9).

Writes to a new output/diagnostics/pre-design/data-inventory/<run-id>/: segments.csv, per_subject.csv, windows.csv,
folds.csv, summary.md.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\inventory_unlabeled.py [--subjects P1 P2]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from eegtraj.data import raw_io
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.diagnostics.inventory import gaps, merge, n_windows
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/diagnostics/pre-design/data-inventory")
CACHE = Path("dataset/cache/98f5fa6e66")
FS_OUT = 100
WINDOW_S = 2.0
STRIDES_S = (0.1, 0.25, 0.5, 1.0, 2.0)
SOURCES = ("lift_windows", "between_lifts", "st")
logger = logging.getLogger("inventory")


def eeg_info(subject: str, series: str) -> tuple[int, float]:
    hs = raw_io.load_hs(subject, series)
    return int(hs["eeg"]["sig"].shape[0]), float(hs["eeg"]["samplingrate"])


def subject_segments(subject: str) -> list[dict]:
    """Contiguous segments (raw samples) per source; lengths also on the 100 Hz grid."""
    trials = pd.read_csv(CACHE / subject / "trials.csv")
    rows = []
    for series, tr in trials.groupby("series"):
        n_raw, fs = eeg_info(subject, f"S{series}")
        factor = int(round(fs / FS_OUT))
        lifts = merge([(int(t.hs_start_sample), int(t.hs_start_sample) + factor * int(t.n_samples)) for t in tr.itertuples()])
        if lifts[-1][1] > n_raw or lifts[0][0] < 0:
            raise ValueError(f"{subject} S{series}: a lift lies outside the HS recording")
        for source, segs in (("lift_windows", lifts), ("between_lifts", gaps(lifts, 0, n_raw))):
            rows += [{"subject": subject, "series": f"S{series}", "source": source, "start": a, "end": b, "fs_hz": fs,
                      "n_100hz": (b - a) // factor} for a, b in segs]
    n_raw, fs = eeg_info(subject, "ST")
    rows.append({"subject": subject, "series": "ST", "source": "st", "start": 0, "end": n_raw, "fs_hz": fs,
                 "n_100hz": n_raw // int(round(fs / FS_OUT))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    split_cfg = load_split_config("configs/splits/default.yaml")
    subjects = args.subjects or list(split_cfg["subjects"])
    run_dir = setup_run(OUTPUT, {"subjects": subjects, "window_s": WINDOW_S, "strides_s": list(STRIDES_S), "fs_out_hz": FS_OUT,
                                 "cache_dir": str(CACHE)})
    t0 = time.time()
    segs = []
    for s in subjects:
        segs += subject_segments(s)
        logger.info("%s done (%.0f s)", s, time.time() - t0)
    segs = pd.DataFrame(segs)
    segs["seconds"] = (segs.end - segs.start) / segs.fs_hz
    segs.to_csv(run_dir / "segments.csv", index=False)

    hours = segs.pivot_table(index="subject", columns="source", values="seconds", aggfunc="sum").reindex(columns=SOURCES) / 3600
    hours["all"] = hours.sum(axis=1)
    hours = hours.reindex(subjects)
    window = int(WINDOW_S * FS_OUT)
    win = pd.DataFrame([{"subject": s, "source": src, "stride_s": st,
                         "n_windows": n_windows(g.n_100hz.to_numpy(), window, int(round(st * FS_OUT)))}
                        for (s, src), g in segs.groupby(["subject", "source"]) for st in STRIDES_S])
    win.to_csv(run_dir / "windows.csv", index=False)
    wide = win.pivot_table(index="subject", columns=["source", "stride_s"], values="n_windows", aggfunc="sum").reindex(subjects)
    per_subject = hours.round(3).join(wide.xs(0.5, axis=1, level="stride_s").add_prefix("windows_0.5s_"))
    per_subject.to_csv(run_dir / "per_subject.csv")

    folds = []
    for fold in make_folds(split_cfg, "loso"):
        others = [s for s in subjects if s not in fold.test_subjects]
        test = [s for s in fold.test_subjects if s in subjects]
        row = {"fold": fold.name, "test": ",".join(test), "n_other_subjects": len(others)}
        for part, names in (("others", others), ("test", test)):
            row[f"{part}_hours"] = float(hours.loc[names, "all"].sum())
            row[f"{part}_hours_lifts"] = float(hours.loc[names, "lift_windows"].sum())
            for st in STRIDES_S:
                row[f"{part}_windows_{st}s"] = int(win[win.subject.isin(names) & (win.stride_s == st)].n_windows.sum())
        folds.append(row)
    folds = pd.DataFrame(folds)
    folds.to_csv(run_dir / "folds.csv", index=False)

    totals = win.groupby(["source", "stride_s"]).n_windows.sum().unstack()
    (run_dir / "summary.md").write_text(
        "# Unlabeled data inventory\n\nHours of EEG per subject and source (raw HS sample counts). Lift windows are the cached "
        "lifts; between-lift = the rest of HS S1-S9; ST = the HS_ST recording. Windows: 2 s on the 100 Hz grid, inside "
        "contiguous segments only. Causal filters need several seconds to settle at the start of each recording; those "
        "seconds are included here.\n\n## Hours per subject\n\n" + hours.round(3).to_markdown()
        + f"\n\nTotal: {hours['all'].sum():.2f} h; lift windows {hours['lift_windows'].sum():.2f} h.\n\n"
        "## 2 s windows over all subjects, by stride (s)\n\n" + totals.to_markdown()
        + "\n\n## LOSO folds (others = all subjects except the test subject, including P3 and P6)\n\n"
        + folds.round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(hours.round(3).to_string())
    print(totals.to_string())
    print(folds[["fold", "others_hours", "test_hours", "others_windows_0.5s", "test_windows_0.5s"]].round(2).to_string(index=False))
    logger.info("Wall time %.0f s -> %s", time.time() - t0, run_dir)


if __name__ == "__main__":
    main()
