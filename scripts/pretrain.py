"""Stage 1 pretraining of the neural-branch encoder for one variant (LeJEPA / classic JEPA, guided / unguided, MAE).

Data: unlabeled 2 s windows from the continuous streams (``eegtraj.data.unlabeled``), no labels. Guided variants
also load the ICA-cleaned stream for the target view. Health metrics (embedding SD, effective rank, linear-probe r and
``residual_r_specific`` on the screening subjects, ``eegtraj.models.probe``) are logged every ``health_every`` steps;
classic JEPA logs them for the online and the EMA encoder (SSL-10).

Writes to a new output/proposed-model/pretrain/<variant>/<run-id>/: config.yaml, log, train_log.csv, health.csv,
metrics.json, checkpoints/encoder_step<k>.pt and encoder_final.pt (online encoder with gate; classic JEPA also
encoder_ema_*.pt). ``--sweep`` (budget sweep, with ``--lam`` / ``--steps``) writes to
output/proposed-model/pretrain-sweep/<variant>/<run-id>_lam<lam>_s<steps>/ instead.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\pretrain.py --config configs\\pretrain\\pretrain.yaml --variant lejepa_guided
    .venv\\Scripts\\python scripts\\pretrain.py --config configs\\pretrain\\pretrain.yaml --variant mae --smoke
    .venv\\Scripts\\python scripts\\pretrain.py --config configs\\pretrain\\pretrain.yaml --variant lejepa_guided --sweep --lam 0.05 --steps 6000
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eegtraj.data.target_view import prior_vector
from eegtraj.data.unlabeled import UnlabeledWindows
from eegtraj.data.windows import input_channel_names
from eegtraj.models.encoder import CausalEncoder, n_parameters
from eegtraj.models.pretraining import Pretrainer
from eegtraj.models.probe import load_probe_subject, probe_report
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/proposed-model/pretrain")
SWEEP_OUTPUT = Path("output/proposed-model/pretrain-sweep")
logger = logging.getLogger("pretrain")


def resolve_config(cfg: dict, variant: str, smoke: bool, lam: float | None = None, steps: int | None = None) -> dict:
    """Config with the variant, the --smoke overrides and the sweep overrides (lam, steps) applied; nested configs
    inlined."""
    if variant not in cfg["variants"]:
        raise SystemExit(f"unknown variant {variant!r}; choose from {sorted(cfg['variants'])}")
    out = copy.deepcopy(cfg)
    out["variant"] = {"name": variant, **cfg["variants"][variant]}
    out["smoke"] = smoke
    out["encoder"] = yaml.safe_load(Path(cfg["encoder_config"]).read_text(encoding="utf-8"))
    out["windows"] = yaml.safe_load(Path(cfg["windows_config"]).read_text(encoding="utf-8"))
    out["prior"] = yaml.safe_load(Path(out["encoder"]["prior_config"]).read_text(encoding="utf-8"))
    if smoke:
        s = cfg["smoke"]
        out["windows"]["subjects"] = list(s["subjects"])
        out["train"].update({k: v for k, v in s.items() if k != "subjects"})
    if lam is not None:
        out["objective"]["lam"] = float(lam)
    if steps is not None:
        out["train"]["steps"] = int(steps)
    return out


def lr_at(step: int, base: float, warmup_frac: float, total: int) -> float:
    """Linear warm-up over ``warmup_frac`` of the steps, then cosine to 0."""
    warmup = max(1, round(warmup_frac * total))
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


class BatchSampler:
    """Uniform random windows (with replacement across epochs), fetched in one pass over the memory maps."""

    def __init__(self, data: UnlabeledWindows, seed: int):
        self.data, self.rng = data, np.random.default_rng(seed)
        self.order, self.pos = self.rng.permutation(len(data)), 0

    def __call__(self, batch: int) -> list[torch.Tensor]:
        if self.pos + batch > len(self.order):
            self.order, self.pos = self.rng.permutation(len(self.data)), 0
        idx = self.order[self.pos:self.pos + batch]
        self.pos += batch
        items = [self.data[int(k)] for k in idx]
        if isinstance(items[0], torch.Tensor):
            return [torch.stack(items)]
        return [torch.stack(v) for v in zip(*items)]


def save_encoder(path: Path, enc: torch.nn.Module, cfg: dict, step: int, kind: str) -> None:
    torch.save({"state_dict": enc.state_dict(), "encoder": cfg["encoder"], "variant": cfg["variant"], "step": step,
                "kind": kind}, path)


def health(pre: Pretrainer, probes, cfg: dict, device, step: int) -> list[dict]:
    p = cfg["probe"]
    pairs = [(list(a), list(b)) for a, b in p["pairs"]]
    encoders = [("online", pre.encoder)] + ([("ema", pre.target_encoder)] if pre.target_encoder is not None else [])
    rows = []
    for name, enc in encoders:
        rep = probe_report(enc, probes, pairs, device, float(p["alpha"]), int(cfg["seed"]), int(p["n_permutations"]))
        rows.append({"step": step, "encoder": name, **rep})
        logger.info("health step %d %s: sd_min %.3f sd_mean %.3f eff_rank %.1f within r %.3f rrs %.4f | cross r %.3f "
                    "rrs %.4f", step, name, rep["emb_sd_min"], rep["emb_sd_mean"], rep["eff_rank"], rep["within_r"],
                    rep["within_rrs"], rep["cross_r"], rep["cross_rrs"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--smoke", action="store_true", help="2 subjects, a few hundred steps")
    parser.add_argument("--sweep", action="store_true", help="budget sweep run: writes under pretrain-sweep/")
    parser.add_argument("--lam", type=float, default=None, help="override objective.lam (sweep)")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps (sweep)")
    args = parser.parse_args()
    cfg = resolve_config(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.variant, args.smoke, args.lam,
                         args.steps)
    cfg["sweep"] = args.sweep
    tag = "_".join(t for t in ["smoke" if args.smoke else "",
                               f"lam{cfg['objective']['lam']:g}_s{cfg['train']['steps']}" if args.sweep else ""] if t)
    run_dir = setup_run((SWEEP_OUTPUT if args.sweep else OUTPUT) / args.variant, cfg, tag=tag or None)
    (run_dir / "checkpoints").mkdir()
    tr, ob, var = cfg["train"], cfg["objective"], cfg["variant"]
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("variant %s (objective %s, guided %s), device %s, seed %d", var["name"], var["objective"],
                var["guided"], device, seed)

    wcfg = cfg["windows"]
    variants = [cfg["raw_variant"]] + ([cfg["cleaned_variant"]] if var["guided"] else [])
    data = UnlabeledWindows(wcfg["cache_dir"], wcfg["subjects"], variants, wcfg["sources"], int(wcfg["window"]),
                            int(wcfg["stride"]))
    logger.info("unlabeled windows: %d from %s (sources %s, variants %s)", len(data), data.subjects, wcfg["sources"],
                variants)
    names = input_channel_names(wcfg["cache_dir"], data.subjects[0], "broadband", "all")
    prior = torch.from_numpy(prior_vector(cfg["prior"], names))
    enc_kw = {k: v for k, v in cfg["encoder"].items() if k != "prior_config"}
    encoder = CausalEncoder(prior, **enc_kw)
    pre = Pretrainer(encoder, var["objective"], bool(var["guided"]), prior, lam=float(ob["lam"]),
                     lambda_gate=float(ob["lambda_gate"]), mask_ratio=tuple(ob["mask_ratio"]),
                     total_steps=int(tr["steps"]), predictor=ob["predictor"], sigreg=ob["sigreg"],
                     ema=tuple(ob["ema"]), seed=seed).to(device)
    logger.info("encoder parameters %d, predictor parameters %d", n_parameters(encoder), n_parameters(pre.predictor))
    params = [q for q in pre.parameters() if q.requires_grad]
    decay = [q for q in params if q.dim() >= 2]
    no_decay = [q for q in params if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(tr["weight_decay"])},
                             {"params": no_decay, "weight_decay": 0.0}], lr=float(tr["lr"]))
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tr["amp"])
    use_amp = amp is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == torch.float16)

    probes = [load_probe_subject(cfg["probe"]["cache_dir"], s, int(wcfg["window"]), int(cfg["probe"]["stride"]))
              for s in cfg["probe"]["subjects"]]
    logger.info("probe windows: %s", {p.subject: len(p.windows) for p in probes})

    sampler = BatchSampler(data, seed)
    steps, batch = int(tr["steps"]), int(tr["batch_size"])
    log_rows, health_rows = [], health(pre, probes, cfg, device, 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, acc = time.time(), {}
    pre.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, float(tr["lr"]), float(tr["warmup_frac"]), steps)
        views = [v.to(device, non_blocking=True) for v in sampler(batch)]
        raw, cleaned = views[0], (views[1] if len(views) > 1 else None)
        with torch.autocast(device_type=device.type, dtype=amp or torch.float32, enabled=use_amp):
            out = pre(raw, cleaned, step)
        if not torch.isfinite(out["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: v.item() for k, v in out.items()} }")
        opt.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(tr["grad_clip"]))
        scaler.step(opt)
        scaler.update()
        momentum = pre.after_step(step)
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + v.item()
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + float(grad_norm)
        acc["n"] = acc.get("n", 0) + 1
        done = step + 1
        if done % int(tr["log_every"]) == 0 or done == steps:
            n = acc.pop("n")
            row = {"step": done, "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1),
                   **{k: v / n for k, v in acc.items()}}
            if momentum is not None:
                row["ema_momentum"] = momentum
            log_rows.append(row)
            logger.info("step %d " + " ".join(f"{k} {v:.4g}" for k, v in row.items() if k != "step"), done)
            acc = {}
            pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
        if done % int(tr["health_every"]) == 0 or done == steps:
            health_rows += health(pre, probes, cfg, device, done)
            pd.DataFrame(health_rows).to_csv(run_dir / "health.csv", index=False)
        if done % int(tr["checkpoint_every"]) == 0 and done != steps:
            save_encoder(run_dir / "checkpoints" / f"encoder_step{done}.pt", pre.encoder, cfg, done, "online")
            if pre.target_encoder is not None:
                save_encoder(run_dir / "checkpoints" / f"encoder_ema_step{done}.pt", pre.target_encoder, cfg, done, "ema")

    save_encoder(run_dir / "checkpoints" / "encoder_final.pt", pre.encoder, cfg, steps, "online")
    if pre.target_encoder is not None:
        save_encoder(run_dir / "checkpoints" / "encoder_ema_final.pt", pre.target_encoder, cfg, steps, "ema")
    elapsed = time.time() - t0
    final = {r["encoder"]: {k: v for k, v in r.items() if k not in ("step", "encoder")}
             for r in health_rows if r["step"] == steps}
    metrics = {"variant": var, "steps": steps, "batch_size": batch, "n_windows": len(data),
               "train_seconds": round(elapsed, 1), "sec_per_step": round(elapsed / steps, 4),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None,
               "encoder_parameters": n_parameters(encoder), "final_loss": log_rows[-1], "final_health": final,
               "gate_weights": dict(zip(names, encoder.gate.weights.detach().cpu().numpy().round(4).tolist()))}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("done in %.1f s (%.3f s/step); final health %s", elapsed, elapsed / steps, final)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
