"""Add an extra EEG band to the existing cache.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\build_band_variant.py --config configs\\cache\\mrcp.yaml [--subjects P1]

Needs the main cache. A subject that already has the arrays is skipped.
Logs go to output/diagnostics/pre-design/cache-build/<run-id>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eegtraj.data.band_variant import build_band_variant, with_band
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/diagnostics/pre-design/cache-build")


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
        if (cache_dir / subject / f"variant_{band}_{ref}.json").exists():
            print(f"{subject}: already built, skipped")
            continue
        built.append(build_band_variant(subject, base, Path(base["paths"]["raw_dir"]), cache_dir, band, ref,
                                        int(cfg["n_pre"]), cfg.get("envelope")))
    (run_dir / "metrics.json").write_text(json.dumps({"built": built}, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
