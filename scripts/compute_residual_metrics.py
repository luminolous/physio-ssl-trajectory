"""Compute the residual metrics (residual.py) for every saved predictions.npz.

Scans output/diagnostics/ and output/data-baselines/ (proposed-model runs: scripts/rescore_matched.py) and leaves
the runs unchanged. residual_r_matched added 2026-11-26 (design.md SSL-27). Writes residual.csv (one row per run
and test subject) to output/diagnostics/residual/<run-id>/. scripts/summarize_runs.py reads the newest one.
Runs already in the newest table keep their rows, unless --all is given.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compute_residual_metrics.py [--all]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eegtraj.diagnostics.residual import load_templates, score_predictions
from eegtraj.utils.run import setup_run

OUTPUT = Path("output")
OUT_DIR = OUTPUT / "diagnostics" / "residual"
SEED = 20261106
N_PERMUTATIONS = 5
REPORTED = ("residual_r", "residual_r_oracle", "residual_r_other_trial", "residual_r_specific", "residual_r_matched")
ROOTS = (OUTPUT / "diagnostics", OUTPUT / "data-baselines")  # proposed-model runs: scripts/rescore_matched.py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="also recompute runs already in the newest table")
    args = parser.parse_args()
    tables = sorted(OUT_DIR.glob("2*/residual.csv"))
    kept = pd.read_csv(tables[-1]) if tables and not args.all else None
    done = set(kept.run) if kept is not None else set()
    runs = sorted(p.parent for root in ROOTS for p in root.rglob("predictions.npz") if p.parent.as_posix() not in done)
    run_dir = setup_run(OUT_DIR, {"seed": SEED, "n_permutations": N_PERMUTATIONS, "n_runs": len(runs),
                                  "runs": [r.as_posix() for r in runs]})
    print(f"Run directory: {run_dir}")
    logging.getLogger("eegtraj.baselines").setLevel(logging.WARNING)
    cache: dict[tuple, dict] = {}
    rows, skipped = [], []
    for path in runs:
        cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        if "data" not in cfg or "splits" not in cfg:
            skipped.append(path.as_posix())  # not a runner layout (no data.windows / splits)
            print(path.as_posix(), "skipped: config has no data.windows / splits")
            continue
        w, flags = cfg["data"]["windows"], cfg.get("exclude_flags", [])
        key = (w["cache_dir"], cfg["splits"]["config"], cfg["splits"]["mode"], w["target"], tuple(flags))
        if key not in cache:
            cache[key] = load_templates(*key[:4], flags)
        with np.load(path / "predictions.npz") as predictions:
            scored = score_predictions(predictions, cache[key], SEED, N_PERMUTATIONS)
        rows += [{"run": path.as_posix(), **row} for row in scored]
        print(path.as_posix(), *(f"{k} {pd.Series([r[k] for r in scored]).mean():.3f}" for k in REPORTED))
    table = pd.concat([kept, pd.DataFrame(rows)], ignore_index=True) if kept is not None else pd.DataFrame(rows)
    table.to_csv(run_dir / "residual.csv", index=False)
    pd.Series(skipped, name="run").to_csv(run_dir / "skipped.csv", index=False)
    print(f"{len(skipped)} runs skipped (see skipped.csv)")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
