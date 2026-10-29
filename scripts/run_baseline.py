"""Run one baseline over the configured folds.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_baseline.py --config configs\\baselines\\template.yaml [--smoke]
        [--subjects P1 P2] [--seed 1] [--tag name]

Normal runs write to output/data-baselines/baselines/<baseline>/<run-id>/;
smoke runs (1 fold, 20 trials per subject) write to output/smoke/baselines/<baseline>/<run-id>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegtraj.baselines.runner import resolve_config, run
from eegtraj.utils.run import setup_run

OUTPUT_ROOT = Path("output/data-baselines/baselines")
SMOKE_ROOT = Path("output/smoke/baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--subjects", nargs="+", help="only folds whose test subject is listed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tag")
    args = parser.parse_args()

    cfg = resolve_config(args.config, {"smoke": args.smoke or None, "subjects": args.subjects, "seed": args.seed})
    root = SMOKE_ROOT if cfg.get("smoke") else OUTPUT_ROOT
    run_dir = setup_run(root / cfg["baseline"]["name"], cfg, tag=args.tag)
    print(f"Run directory: {run_dir}")
    run(cfg, run_dir)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
