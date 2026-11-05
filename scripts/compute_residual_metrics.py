"""Compute the residual metrics (residual.py) for every saved predictions.npz.

Scans output/ except output/smoke/ and leaves the runs unchanged. Writes residual.csv (one row per run
and test subject) to output/diagnostics/residual/<run-id>/. scripts/summarize_runs.py reads the newest one.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compute_residual_metrics.py
"""

from __future__ import annotations

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
REPORTED = ("residual_r", "residual_r_oracle", "residual_r_other_trial", "residual_r_specific")


def main() -> None:
    runs = sorted(p.parent for p in OUTPUT.rglob("predictions.npz") if OUTPUT / "smoke" not in p.parents)
    run_dir = setup_run(OUT_DIR, {"seed": SEED, "n_permutations": N_PERMUTATIONS, "n_runs": len(runs),
                                  "runs": [r.as_posix() for r in runs]})
    print(f"Run directory: {run_dir}")
    logging.getLogger("eegtraj.baselines").setLevel(logging.WARNING)
    cache: dict[tuple, dict] = {}
    rows = []
    for path in runs:
        cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        w, flags = cfg["data"]["windows"], cfg.get("exclude_flags", [])
        key = (w["cache_dir"], cfg["splits"]["config"], cfg["splits"]["mode"], w["target"], tuple(flags))
        if key not in cache:
            cache[key] = load_templates(*key[:4], flags)
        with np.load(path / "predictions.npz") as predictions:
            scored = score_predictions(predictions, cache[key], SEED, N_PERMUTATIONS)
        rows += [{"run": path.as_posix(), **row} for row in scored]
        print(path.as_posix(), *(f"{k} {pd.Series([r[k] for r in scored]).mean():.3f}" for k in REPORTED))
    pd.DataFrame(rows).to_csv(run_dir / "residual.csv", index=False)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
