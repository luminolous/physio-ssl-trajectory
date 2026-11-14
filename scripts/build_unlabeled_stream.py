"""Build continuous EEG streams (lift windows, between lifts, ST) for self-supervised pretraining.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\build_unlabeled_stream.py --config configs\\cache\\unlabeled_stream.yaml [--subjects P1]

Needs the main cache (and, for a cleaned variant, the saved ICA). A subject that already has the stream is skipped.
Logs go to output/proposed-model/data/cache-build/<run-id>/; metrics.json holds hours per subject and source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eegtraj.data.band_variant import with_band
from eegtraj.data.unlabeled import build_stream, segment_hours
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/proposed-model/data/cache-build")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--tag")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    band, ref = cfg["band"], cfg["reference"]
    base = with_band(yaml.safe_load(Path(cfg["cache_config"]).read_text(encoding="utf-8")), band, cfg["band_hz"])
    subjects = args.subjects or base["subjects"]
    run_dir = setup_run(OUTPUT, {**cfg, "subjects": subjects}, tag=args.tag)
    print(f"Run directory: {run_dir}")
    cache_dir = Path(cfg["cache_dir"])
    built = []
    for subject in subjects:
        if (cache_dir / subject / f"stream_{band}_{ref}.json").exists():
            print(f"{subject}: already built, skipped")
            continue
        built.append(build_stream(subject, base, Path(base["paths"]["raw_dir"]), cache_dir, band, ref, cfg.get("ica_from")))
    hours = segment_hours(cache_dir, subjects, float(base["fs_out_hz"]))
    totals = hours.sum().round(3).to_dict()
    print(hours.round(3).to_string())
    print("Total hours:", totals)
    (run_dir / "metrics.json").write_text(json.dumps({"built": built, "hours_per_subject": hours.round(4).to_dict(orient="index"),
                                                      "hours_total": totals}, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
