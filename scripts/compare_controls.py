"""Compare artifact-control decoders with all-channel EEG using the same model.

Reads the latest non-smoke run of every artifact control under
output/data-baselines/artifact-controls/<name>/ (and the all-channel CAR reference from
output/data-baselines/baselines/<model>/ when no control run exists for it), then writes
a CSV and a markdown table to output/data-baselines/artifact-controls/summary/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_controls.py [--model ridge]
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
BASELINE_FOR_ALL_CAR = {"ridge": "ridge_lag"}


def latest_run(directory: Path) -> Path | None:
    runs = [d for d in sorted(directory.glob("2*")) if (d / "metrics.json").exists()
            and not json.loads((d / "metrics.json").read_text(encoding="utf-8"))["smoke"]]
    return runs[-1] if runs else None


def row(name: str, run: Path, all_r: dict[str, float] | None) -> dict:
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    subj_r = {s: v["r"]["mean"] for s, v in m["per_subject"].items()}
    out = {"control": name, "run": str(run).replace("\\", "/")}
    for metric in ("r", "r2", "nrmse"):
        out[f"{metric}_mean"] = m["aggregate"][metric]["mean"]
        out[f"{metric}_sd"] = m["aggregate"][metric]["sd"]
    if all_r is not None:
        diff = np.array([subj_r[s] - all_r[s] for s in subj_r])
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
        all_dir = CONTROLS / f"{args.model}_all_{ref}"
        all_run = latest_run(all_dir)
        if all_run is None and ref == "car" and args.model in BASELINE_FOR_ALL_CAR:
            all_run = latest_run(ROOT / "baselines" / BASELINE_FOR_ALL_CAR[args.model])
        if all_run is None:
            print(f"missing all-channel run for {ref}; skipping {ref}")
            continue
        all_row, all_r = row(f"all ({ref})", all_run, None)
        rows.append({**all_row, "reference": ref, "delta_r_vs_all": 0.0, "r_ratio_vs_all": 1.0})
        for ch in SETS[1:]:
            run = latest_run(CONTROLS / f"{args.model}_{ch}_{ref}")
            if run is None:
                print(f"missing {args.model}_{ch}_{ref}")
                continue
            r, _ = row(f"{ch} ({ref})", run, all_r)
            rows.append({**r, "reference": ref})
        emg = latest_run(CONTROLS / f"{args.model}_emg")
        if emg is not None:
            r, _ = row("emg", emg, all_r)
            rows.append({**r, "reference": f"n/a (vs all {ref})"})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{args.model}_controls.csv", index=False)
    cols = ["control", "reference", "r_mean", "r_sd", "r2_mean", "nrmse_mean", "delta_r_vs_all",
            "r_ratio_vs_all", "n_subjects_above_all", "run"]
    md = df[[c for c in cols if c in df]].to_markdown(index=False, floatfmt=".3f")
    (OUT_DIR / f"{args.model}_controls.md").write_text(
        f"# Artifact controls ({args.model})\n\nPer-trial Pearson r (D7), mean +- SD across the 10 LOSO test subjects. "
        "delta_r_vs_all: mean over subjects of (control r - all-channel r) under the same reference.\n\n" + md + "\n",
        encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
