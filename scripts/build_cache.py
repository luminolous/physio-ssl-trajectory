"""Build the preprocessed per-subject cache.

Writes ``<cache_root>/<config_hash>/<subject>/`` (see ``eegtraj.data.cache``) and a
run directory with the resolved config, env info, log, and a copy of each manifest.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\build_cache.py --config configs\\cache\\default.yaml --subjects P1
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import yaml

from eegtraj.data.cache import build_subject, config_hash, write_cache_config
from eegtraj.utils.run import setup_run

logger = logging.getLogger("build_cache")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", help="override the subject list")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild subjects that already exist in the cache")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.subjects:
        cfg["subjects"] = args.subjects
    h = config_hash(cfg)
    cache_dir = Path(cfg["paths"]["cache_root"]) / h

    run_dir = setup_run(cfg["paths"]["output_dir"], {**cfg, "config_hash": h}, tag=args.tag)
    print(f"Run directory: {run_dir}")
    logger.info("Config hash %s, cache dir %s, subjects %s", h, cache_dir, cfg["subjects"])
    write_cache_config(cfg, cache_dir)

    env_lines = (run_dir / "env.txt").read_text(encoding="utf-8").splitlines()
    git = {k: v.strip() for k, v in (line.split(":", 1) for line in env_lines if line.startswith("git_"))}

    t0 = time.time()
    for subject in cfg["subjects"]:
        subj_dir = cache_dir / subject
        if subj_dir.exists():
            if not args.force:
                logger.info("%s already cached, skipping (use --force to rebuild)", subject)
                continue
            shutil.rmtree(subj_dir)
        manifest = build_subject(subject, cfg, Path(cfg["paths"]["raw_dir"]), cache_dir, git_info=git)
        (run_dir / f"manifest_{subject}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("%s done in %.1f s: %d trials, %d samples, bad channels %s, onset mismatch %d, "
                    "filter transient %s", subject, manifest["build_seconds"], manifest["n_trials"],
                    manifest["n_samples"], manifest["bad_channels"], manifest["n_onset_mismatch"],
                    manifest["n_filter_transient"])
    logger.info("All done in %.1f s", time.time() - t0)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
