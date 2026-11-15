"""Linear-probe diagnostic of pretraining checkpoints on the screening subjects (P3, P6). Inference only.

For every checkpoint of the given pretraining runs, and for random-init encoders with several seeds (probe noise
reference), runs the health probe (``eegtraj.models.probe.probe_report``: within-subject leave-one-series-out and
cross-subject P3 <-> P6) on two inputs:
- ``raw``: the broadband CAR cache, the encoder's input in training;
- ``ica``: the ICA-cleaned cache (``broadband_ica``, its own per-subject z-score), i.e. without the removed ocular
  components. Targets, scored samples and template_cue are the same for both inputs.

Writes to a new output/proposed-model/diagnostics/probe-checkpoints/<run-id>/: table.csv, summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\probe_checkpoints.py --runs <run dir> <run dir> --random-seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

from eegtraj.data.target_view import prior_vector
from eegtraj.data.windows import input_channel_names
from eegtraj.models.encoder import CausalEncoder
from eegtraj.models.probe import load_probe_subject, probe_report
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/proposed-model/diagnostics/probe-checkpoints")
INPUTS = {"raw": "broadband", "ica": "broadband_ica"}
SHOW = ["within_rrs", "within_rrs_P3", "within_rrs_P6", "within_r", "cross_rrs", "cross_r", "emb_sd_min", "eff_rank"]


def checkpoints(run: Path, kinds: tuple[str, ...]) -> list[tuple[str, int, Path]]:
    """(kind, step, path) of the run's encoder checkpoints, sorted by step."""
    out = []
    for f in (run / "checkpoints").glob("encoder*.pt"):
        kind = "ema" if f.name.startswith("encoder_ema") else "online"
        if kind not in kinds:
            continue
        m = re.search(r"step(\d+)", f.name)
        step = int(m.group(1)) if m else int(torch.load(f, map_location="cpu", weights_only=False)["step"])
        out.append((kind, step, f))
    return sorted(out, key=lambda t: (t[0], t[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--random-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain/pretrain.yaml"))
    parser.add_argument("--kinds", nargs="*", default=["online"], help="online and/or ema")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pcfg = cfg["probe"]
    run_dir = setup_run(OUTPUT, {"runs": [str(r) for r in args.runs], "random_seeds": args.random_seeds,
                                 "kinds": args.kinds, "probe": pcfg, "inputs": INPUTS})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = yaml.safe_load(Path(cfg["encoder_config"]).read_text(encoding="utf-8"))
    kw = {k: v for k, v in enc_cfg.items() if k != "prior_config"}
    names = input_channel_names(pcfg["cache_dir"], pcfg["subjects"][0], "broadband", "all")
    prior = torch.from_numpy(prior_vector(yaml.safe_load(Path(enc_cfg["prior_config"]).read_text(encoding="utf-8")),
                                          names))
    pairs = [(list(a), list(b)) for a, b in pcfg["pairs"]]

    encoders: list[tuple[str, str, str, int, object]] = []  # (encoder, run, kind, step, path or seed)
    for seed in args.random_seeds:
        encoders.append((f"random_seed{seed}", "-", "init", 0, seed))
    for run in args.runs:
        variant = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["variant"]["name"]
        for kind, step, path in checkpoints(run, tuple(args.kinds)):
            encoders.append((variant, run.name, kind, step, path))

    rows = []
    for input_name, band in INPUTS.items():
        probes = [load_probe_subject(pcfg["cache_dir"], s, int(kw["window"]), int(pcfg["stride"]), band)
                  for s in pcfg["subjects"]]
        for name, run_name, kind, step, src in encoders:
            if isinstance(src, int):
                torch.manual_seed(src)  # same construction order as scripts/pretrain.py
                enc = CausalEncoder(prior, **kw)
            else:
                enc = CausalEncoder(prior, **kw)
                enc.load_state_dict(torch.load(src, map_location="cpu", weights_only=False)["state_dict"])
            rep = probe_report(enc.to(device), probes, pairs, device, float(pcfg["alpha"]), int(cfg["seed"]),
                               int(pcfg["n_permutations"]))
            rows.append({"input": input_name, "encoder": name, "run": run_name, "kind": kind, "step": step, **rep})
            print(f"{input_name:3s} {name:18s} {run_name[-14:]:14s} {kind:6s} step {step:5d} within_rrs {rep['within_rrs']:.4f} "
                  f"within_r {rep['within_r']:.3f} cross_rrs {rep['cross_rrs']:+.4f}")
        del probes
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "table.csv", index=False)
    rnd = table[table.kind == "init"].groupby("input")[SHOW].agg(["mean", "std"])
    rnd.columns = [f"{a}_{b}" for a, b in rnd.columns]
    pivot = table[table.kind != "init"].pivot_table(index=["encoder", "run", "kind", "step"], columns="input",
                                                    values=["within_rrs", "within_r", "cross_rrs"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    (run_dir / "summary.md").write_text(
        "# Probe of pretraining checkpoints on P3/P6\n\n`raw`: broadband CAR input (training input); `ica`: ICA-cleaned "
        "cache. Selection score `within_rrs` = within-subject (leave-one-series-out) probe `residual_r_specific`, "
        "mean of P3 and P6.\n\n## Random-init encoders (mean and SD over seeds "
        f"{args.random_seeds})\n\n" + rnd.round(4).to_markdown() + "\n\n## Checkpoints\n\n"
        + pivot.round(4).to_markdown() + "\n\n## All rows\n\n" + table[["input", "encoder", "run", "kind", "step"] + SHOW]
        .round(4).to_markdown(index=False) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"rows": table.to_dict(orient="records")}, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
