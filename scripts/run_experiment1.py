"""Run the Experiment 1 grid (RQ1, design.md SSL-21) sequentially through ``scripts/run_decoder.py``. Resumable.

Stage 1 (decided 2026-11-15): the frozen grid (5 pretrained encoders x {neural, hybrid}, random-init frozen control
x {neural, hybrid}, ocular-only once) and fine-tune for the proposed encoder (lejepa_guided) and scratch (random init,
fine-tune) x {neural, hybrid}; 3 decoder seeds each (51 runs). Stage 2 (fine-tune for the other 4 pretrained encoders x
{neural, hybrid}, 24 runs) only after a separate decision on the stage 1 results.

A configuration and seed is skipped when a finished (non-smoke) run directory with metrics.json already exists, so an
interrupted grid can be restarted with the same command. Every run writes its own new directory under
output/proposed-model/decoder/<name>__<head>_<branches>_<setting>_<encoder>/<run-id>_seed<k>/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_experiment1.py --stage 1 --dry-run
    .venv\\Scripts\\python scripts\\run_experiment1.py --stage 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PRETRAIN = Path("output/proposed-model/pretrain")
CHECKPOINTS = {  # one pretraining run per variant (design.md "Seeds"), online encoder with gate (SSL-10)
    "lejepa_guided": PRETRAIN / "lejepa_guided/20261115-123041/checkpoints/encoder_final.pt",
    "lejepa_unguided": PRETRAIN / "lejepa_unguided/20261115-124031/checkpoints/encoder_final.pt",
    "jepa_guided": PRETRAIN / "jepa_guided/20261115-124934/checkpoints/encoder_final.pt",
    "jepa_unguided": PRETRAIN / "jepa_unguided/20261115-125922/checkpoints/encoder_final.pt",
    "mae": PRETRAIN / "mae/20261115-130726/checkpoints/encoder_final.pt",
}
SEEDS = (1, 2, 3)
OUTPUT = Path("output/proposed-model/decoder")
CONFIG = Path("configs/decoder/proposed.yaml")
NAME = "exp1"


def grid(stage: int) -> list[dict]:
    """Configurations: setting, branches, encoder (variant name, 'random' or 'none')."""
    out = []
    if stage == 1:
        for enc in list(CHECKPOINTS) + ["random"]:
            for br in ("neural", "hybrid"):
                out.append({"setting": "frozen", "branches": br, "encoder": enc})
        out.append({"setting": "frozen", "branches": "ocular", "encoder": "none"})
        for enc in ("lejepa_guided", "random"):  # random + fine-tune = scratch
            for br in ("neural", "hybrid"):
                out.append({"setting": "finetune", "branches": br, "encoder": enc})
    elif stage == 2:
        for enc in ("lejepa_unguided", "jepa_guided", "jepa_unguided", "mae"):
            for br in ("neural", "hybrid"):
                out.append({"setting": "finetune", "branches": br, "encoder": enc})
    else:
        raise SystemExit("stage must be 1 or 2")
    return out


def folder(c: dict) -> str:
    label = {"none": "noenc", "random": "random0"}.get(c["encoder"], c["encoder"])
    return f"{NAME}__proposed_{c['branches']}_{c['setting']}_{label}"


def done(c: dict, seed: int) -> bool:
    return any(not p.parent.name.endswith("_smoke") for p in (OUTPUT / folder(c)).glob(f"*_seed{seed}/metrics.json"))


def command(c: dict, seed: int) -> list[str]:
    cmd = [sys.executable, "scripts/run_decoder.py", "--config", str(CONFIG), "--name", NAME, "--head", "proposed",
           "--setting", c["setting"], "--branches", c["branches"], "--seed", str(seed), "--folds", "loso"]
    if c["encoder"] == "random":
        cmd += ["--encoder-source", "random"]
    elif c["encoder"] != "none":
        cmd += ["--encoder-source", "checkpoint", "--checkpoint", CHECKPOINTS[c["encoder"]].as_posix()]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--dry-run", action="store_true", help="list the runs and their status, run nothing")
    args = parser.parse_args()
    missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    todo = [(c, s) for c in grid(args.stage) for s in SEEDS]
    pending = [(c, s) for c, s in todo if not done(c, s)]
    print(f"stage {args.stage}: {len(todo)} runs, {len(todo) - len(pending)} finished, {len(pending)} to run")
    for i, (c, s) in enumerate(todo, 1):
        print(f"  [{'done' if (c, s) not in pending else 'todo'}] {folder(c)} seed {s}")
    if args.dry_run:
        return
    for i, (c, s) in enumerate(pending, 1):
        print(f"\n=== {i}/{len(pending)}: {folder(c)} seed {s} ===", flush=True)
        result = subprocess.run(command(c, s))
        if result.returncode != 0:
            raise SystemExit(f"run failed ({result.returncode}): {' '.join(command(c, s))}")
    print("stage complete")


if __name__ == "__main__":
    main()
