"""Explain the Kalman decoder's negative R^2: analyze earlier predictions, then refit selected folds.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\diagnose_kalman.py --config configs\\diagnostics\\kalman_check.yaml
        [--smoke] [--skip-refit] [--subjects P1] [--seed 1] [--tag name]

Normal runs write to output/diagnostics/fixes/kalman/<run-id>/. Smoke runs refit 1 fold with
20 trials per subject and write to output/smoke/diagnostics/fixes/kalman/<run-id>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegtraj.baselines.runner import resolve_config
from eegtraj.diagnostics.kalman import run
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/diagnostics/fixes/kalman")
SMOKE_OUTPUT = Path("output/smoke/diagnostics/fixes/kalman")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-refit", action="store_true", help="only analyze predictions of earlier runs")
    parser.add_argument("--subjects", nargs="+", help="only folds whose test subject is listed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tag")
    args = parser.parse_args()

    cfg = resolve_config(args.config, {"smoke": args.smoke or None, "subjects": args.subjects, "seed": args.seed})
    run_dir = setup_run(SMOKE_OUTPUT if cfg.get("smoke") else OUTPUT, cfg, tag=args.tag)
    print(f"Run directory: {run_dir}")
    run(cfg, run_dir, skip_refit=args.skip_refit)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
