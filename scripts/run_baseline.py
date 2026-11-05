"""Run one baseline over the configured folds.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_baseline.py --config configs\\baselines\\template.yaml [--smoke]
        [--subjects P1 P2] [--seed 1] [--tag name] [--selection-run <grid run dir>]
        [--history-samples 200]

Normal runs write to <root>/<group>/<name>/<run-id>/, where the optional config block
``output: {root, group, name}`` defaults to root output/data-baselines, group "baselines", and the
baseline name (artifact controls use group "artifact-controls"). Smoke runs (1 fold, 20 trials per
subject) write to output/smoke/<group>/<name>/<run-id>/.

--selection-run makes a deep model reuse the per-fold configurations chosen by that grid run.
--history-samples overrides the EEG history and adds history_<n> to the output path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegtraj.baselines.runner import resolve_config, run
from eegtraj.utils.run import setup_run

OUTPUT_ROOT = Path("output/data-baselines")
SMOKE_ROOT = Path("output/smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--subjects", nargs="+", help="only folds whose test subject is listed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tag")
    parser.add_argument("--selection-run", help="grid run directory whose per-fold choices this seed reuses")
    parser.add_argument("--history-samples", type=int, help="EEG window length in samples")
    args = parser.parse_args()

    cfg = resolve_config(args.config, {"smoke": args.smoke or None, "subjects": args.subjects, "seed": args.seed})
    if args.selection_run:
        cfg["baseline"]["params"]["grid"] = None
        cfg["baseline"]["params"]["selection_run"] = args.selection_run
    if args.history_samples:
        cfg["data"]["windows"]["history_samples"] = args.history_samples
    out = cfg.get("output") or {}
    root = SMOKE_ROOT if cfg.get("smoke") else Path(out.get("root", OUTPUT_ROOT))
    group, name = out.get("group", "baselines"), out.get("name", cfg["baseline"]["name"])
    if args.history_samples:
        name = f"{name}/history_{args.history_samples}"
    run_dir = setup_run(root / group / name, cfg, tag=args.tag)
    print(f"Run directory: {run_dir}")
    run(cfg, run_dir)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
