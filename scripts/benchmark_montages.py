"""Compute cost per montage for the electrode reduction curve: parameters and batch-1 inference latency.

Uses the model settings of one electrode reduction config and random input (no data, no training). Writes
compute.csv to a new output/diagnostics/final-checks/electrode-reduction/compute/<run-id>/. Training cost (epoch
time, peak GPU memory in training) comes from the run logs; see scripts/compare_electrode_reduction.py.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\benchmark_montages.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

from eegtraj.diagnostics.compute_cost import montage_cost
from eegtraj.preprocessing.reference import CHANNEL_SETS
from eegtraj.utils.run import setup_run, teardown_logging

OUTPUT = Path("output/diagnostics/final-checks/electrode-reduction/compute")
MODEL_CONFIG = Path("configs/diagnostics/electrode_reduction/ica__cz__car__direct.yaml")
MONTAGES = {"all": 32, **{m: len(CHANNEL_SETS[m]) for m in ("headband_ring", "central9", "contralateral", "minimal3", "cz")}}
logger = logging.getLogger("benchmark_montages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-repeat", type=int, default=500)
    parser.add_argument("--cpu-threads", type=int, default=1, help="torch CPU threads (1 = a single small core)")
    args = parser.parse_args()
    model_cfg = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    windows = yaml.safe_load(Path(model_cfg["data"]["windows_config"]).read_text(encoding="utf-8"))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    cfg = {"model_config": MODEL_CONFIG.as_posix(), "architecture": model_cfg["baseline"]["name"],
           "model": model_cfg["baseline"]["params"]["model"], "n_times": windows["history_samples"], "montages": MONTAGES,
           "devices": devices, "n_warmup": args.n_warmup, "n_repeat": args.n_repeat, "cpu_threads": args.cpu_threads,
           "seed": 0, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    run_dir = setup_run(OUTPUT, cfg)
    print(f"Run directory: {run_dir}")
    torch.set_num_threads(args.cpu_threads)
    rows = []
    for montage, n in MONTAGES.items():
        row = {"montage": montage} | montage_cost(cfg["architecture"], cfg["model"], n, cfg["n_times"], devices,
                                                  args.n_warmup, args.n_repeat, cfg["seed"])
        logger.info("%s", row)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "compute.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"Run directory: {run_dir}")
    teardown_logging()


if __name__ == "__main__":
    main()
