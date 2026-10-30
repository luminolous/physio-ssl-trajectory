"""Compare artifact-control decoders with all-channel EEG using the same model.

Reads every non-smoke run of each artifact control under
output/data-baselines/artifact-controls/<name>/ (and the all-channel CAR reference from
output/data-baselines/baselines/<model>/ when no control run exists for it). Several runs in
one directory are seeds: per-subject values are averaged over seeds first, and the SD of the
aggregate across seeds is reported as ``r_seed_sd``. Writes a CSV and a markdown table to
output/data-baselines/artifact-controls/summary/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_controls.py [--model ridge|eegnet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("output/data-baselines")
CONTROLS = ROOT / "artifact-controls"
OUT_DIR = CONTROLS / "summary"
SETS = ("all", "sensorimotor", "frontal", "temporal")
REFS = ("car", "orig")
BASELINE_FOR_ALL_CAR = {"ridge": "ridge_lag", "eegnet": "eegnet"}


def finished_runs(directory: Path) -> list[Path] | None:
    """All non-smoke runs in ``directory`` (seeds of one configuration), or None."""
    runs = [d for d in sorted(directory.glob("2*")) if (d / "metrics.json").exists()
            and not json.loads((d / "metrics.json").read_text(encoding="utf-8"))["smoke"]]
    return runs or None


def row(name: str, runs: list[Path], all_r: dict[str, float] | None) -> tuple[dict, dict[str, float]]:
    ms = [json.loads((r / "metrics.json").read_text(encoding="utf-8")) for r in runs]
    subjects = list(ms[0]["per_subject"])
    out = {"control": name, "n_runs": len(runs), "run": ", ".join(str(r).replace("\\", "/") for r in runs)}
    for metric in ("r", "r2", "nrmse"):
        per_subject = np.array([[m["per_subject"][s][metric]["mean"] for s in subjects] for m in ms]).mean(axis=0)
        out[f"{metric}_mean"] = float(per_subject.mean())
        out[f"{metric}_sd"] = float(per_subject.std(ddof=1))
    aggs = [m["aggregate"]["r"]["mean"] for m in ms]
    out["r_seed_sd"] = float(np.std(aggs, ddof=1)) if len(aggs) > 1 else float("nan")
    subj_r = {s: float(np.mean([m["per_subject"][s]["r"]["mean"] for m in ms])) for s in subjects}
    if all_r is not None:
        diff = np.array([subj_r[s] - all_r[s] for s in subjects])
        out["delta_r_vs_all"] = float(diff.mean())
        out["r_ratio_vs_all"] = out["r_mean"] / float(np.mean(list(all_r.values())))
        out["n_subjects_above_all"] = int((diff > 0).sum())
    return out, subj_r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ridge")
    args = parser.parse_args()

    rows = []
    for ref in REFS:
        all_runs = finished_runs(CONTROLS / f"{args.model}_all_{ref}")
        if all_runs is None and ref == "car" and args.model in BASELINE_FOR_ALL_CAR:
            all_runs = finished_runs(ROOT / "baselines" / BASELINE_FOR_ALL_CAR[args.model])
        if all_runs is None:
            print(f"missing all-channel run for {ref}; skipping {ref}")
            continue
        all_row, all_r = row(f"all ({ref})", all_runs, None)
        rows.append({**all_row, "reference": ref, "delta_r_vs_all": 0.0, "r_ratio_vs_all": 1.0})
        for ch in SETS[1:]:
            runs = finished_runs(CONTROLS / f"{args.model}_{ch}_{ref}")
            if runs is None:
                print(f"missing {args.model}_{ch}_{ref}")
                continue
            r, _ = row(f"{ch} ({ref})", runs, all_r)
            rows.append({**r, "reference": ref})
        emg = finished_runs(CONTROLS / f"{args.model}_emg")
        if emg is not None:
            r, _ = row("emg", emg, all_r)
            rows.append({**r, "reference": f"n/a (vs all {ref})"})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{args.model}_controls.csv", index=False)
    cols = ["control", "reference", "n_runs", "r_mean", "r_sd", "r_seed_sd", "r2_mean", "nrmse_mean",
            "delta_r_vs_all", "r_ratio_vs_all", "n_subjects_above_all", "run"]
    md = df[[c for c in cols if c in df]].to_markdown(index=False, floatfmt=".3f")
    (OUT_DIR / f"{args.model}_controls.md").write_text(
        f"# Artifact controls ({args.model})\n\nPer-trial Pearson r (D7), mean +- SD across the 10 LOSO test subjects "
        "(per-subject values averaged over seeds when n_runs > 1; r_seed_sd = SD of the aggregate across seeds). "
        "delta_r_vs_all: mean over subjects of (control r - all-channel r) under the same reference.\n\n" + md + "\n",
        encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
