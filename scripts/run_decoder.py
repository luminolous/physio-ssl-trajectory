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
    cfg["smoke_run"] = args.smoke
    branches = next(k for k, v in BRANCHES.items() if v == (m["neural"], m["ocular"]))
    folder = f"{cfg['name']}__{m['head']}_{branches}_{m['setting']}_{encoder_label(m)}"
    run_dir = setup_run(OUTPUT / folder, cfg, tag=f"seed{cfg['seed']}" + ("_smoke" if args.smoke else ""))
    print(f"Run directory: {run_dir}")
    run(cfg, run_dir)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
