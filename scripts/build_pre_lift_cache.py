"""Build the EEG before each lift's window, needed for causal histories longer than 0.5 s.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\build_pre_lift_cache.py --config configs\\cache\\pre_lift.yaml [--subjects P1]

Needs the main cache. Adds pre_lift_* arrays to each subject's folder and leaves the other files alone.
A subject that already has them is skipped. Logs go to output/diagnostics/history/cache-build/<run-id>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eegtraj.data.pre_lift import build_pre_lift
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/diagnostics/history/cache-build")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--tag")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(cfg["cache_config"]).read_text(encoding="utf-8"))
    subjects = args.subjects or base["subjects"]
    run_dir = setup_run(OUTPUT, {**cfg, "subjects": subjects}, tag=args.tag)
    print(f"Run directory: {run_dir}")
    cache_dir, band, ref = Path(cfg["cache_dir"]), cfg["band"], cfg["reference"]
    built = []
    for subject in subjects:
        if (cache_dir / subject / f"pre_lift_eeg_{band}_{ref}.npy").exists():
            print(f"{subject}: already built, skipped")
            continue
        built.append(build_pre_lift(subject, base, Path(base["paths"]["raw_dir"]), cache_dir, band, ref,
                                    int(cfg["max_history_samples"])))
    (run_dir / "metrics.json").write_text(json.dumps({"built": built}, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
