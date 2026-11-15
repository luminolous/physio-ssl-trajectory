"""Train and evaluate the stage 2 decoder over the LOSO folds (``eegtraj.models.decoder_runner``).

Writes to a new output/proposed-model/decoder/<name>__<head>_<branches>_<setting>_<encoder>/<run-id>/: config.yaml (resolved), log,
metrics.json (per fold, per subject, aggregate, template_cue floor, lr grid and training history), predictions.npz
(per test subject: y_true, y, sigma, phi, weights, amp, sample_idx, trial_row).

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_decoder.py --config configs\\decoder\\proposed.yaml --smoke
    .venv\\Scripts\\python scripts\\run_decoder.py --config configs\\decoder\\proposed.yaml --setting finetune
    .venv\\Scripts\\python scripts\\run_decoder.py --config configs\\decoder\\proposed.yaml --branches ocular
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eegtraj.models.decoder_runner import run
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/proposed-model/decoder")
SWEEP_OUTPUT = Path("output/proposed-model/decoder-sweep")
BRANCHES = {"hybrid": (True, True), "neural": (True, False), "ocular": (False, True)}


def encoder_label(m: dict) -> str:
    """Encoder in the folder name: the pretraining variant of the checkpoint, random<seed>, or none."""
    enc = m["encoder"]
    if not m["neural"] or enc["source"] == "none":
        return "noenc"
    if enc["source"] == "random":
        return f"random{enc['random_seed']}"
    return Path(enc["checkpoint"]).parts[-4]  # output/proposed-model/pretrain/<variant>/<run-id>/checkpoints/<file>


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="2 folds, 20 lifts per subject, 2 epochs")
    parser.add_argument("--setting", choices=["frozen", "finetune"])
    parser.add_argument("--branches", choices=sorted(BRANCHES))
    parser.add_argument("--head", choices=["direct", "residual", "global_warp", "proposed"])
    parser.add_argument("--encoder-source", choices=["checkpoint", "random", "none"])
    parser.add_argument("--checkpoint", help="stage 1 encoder checkpoint")
    parser.add_argument("--seed", type=int, help="decoder seed")
    parser.add_argument("--name", help="run name (output folder)")
    parser.add_argument("--folds", choices=["loso", "screening"], help="screening: test P3 / P6 (tuning only)")
    parser.add_argument("--loss", default="", help="loss weight overrides, e.g. aux=0.1,delta=0.01")
    parser.add_argument("--sweep", action="store_true", help="tuning run: writes under decoder-sweep/")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m = cfg["model"]
    if args.setting:
        m["setting"] = args.setting
    if args.branches:
        m["neural"], m["ocular"] = BRANCHES[args.branches]
    if args.head:
        m["head"] = args.head
    if args.encoder_source:
        m["encoder"]["source"] = args.encoder_source
    if args.checkpoint:
        m["encoder"]["checkpoint"] = args.checkpoint
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.name:
        cfg["name"] = args.name
    if not m["neural"]:
        m["encoder"]["source"] = "none"
    if args.folds:
        cfg["folds"] = args.folds
    for item in filter(None, args.loss.split(",")):
        key, value = item.split("=")
        if key not in cfg["loss"]:
            raise SystemExit(f"unknown loss weight {key!r}; choose from {sorted(cfg['loss'])}")
        cfg["loss"][key] = float(value)
    cfg["smoke_run"] = args.smoke
    branches = next(k for k, v in BRANCHES.items() if v == (m["neural"], m["ocular"]))
    folder = f"{cfg['name']}__{m['head']}_{branches}_{m['setting']}_{encoder_label(m)}"
    tag = f"seed{cfg['seed']}"
    if args.sweep:
        tag += "_" + "_".join(f"{k}{v:g}" for k, v in cfg["loss"].items())
    if cfg.get("folds", "loso") != "loso":
        tag += f"_{cfg['folds']}"
    run_dir = setup_run((SWEEP_OUTPUT if args.sweep else OUTPUT) / folder, cfg, tag=tag + ("_smoke" if args.smoke else ""))
    print(f"Run directory: {run_dir}")
    run(cfg, run_dir)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
