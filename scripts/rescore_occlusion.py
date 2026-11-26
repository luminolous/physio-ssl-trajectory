"""Neural-branch occlusion scored with ``residual_r_matched`` (design.md SSL-27), from saved fold models. No training.

Occlusion runs (SSL-22) saved each fold's selected model (``models/<fold>.pt``) but not the occluded predictions, so
their occlusion drops exist only for ``residual_r_specific``. For every finished run under
output/proposed-model/decoder/<name>__*/<run>/ with occlusion groups and saved models, each fold is rebuilt as in
``eegtraj.models.decoder_runner.run`` (fold references from the training subjects only), the saved state dict is
loaded, and the test subject is predicted without occlusion and with each channel group zeroed in the neural input
(ocular proxies intact; frozen tokens recomputed from the occluded EEG), then scored with ``score_subject``.

Checks per fold: unoccluded predictions against predictions.npz (max |difference|), and the rescored r /
residual_r_specific (unoccluded and occluded) against metrics.json.

Writes to a new <out>/<run-id>/: drops.csv (one row per run, test subject, group), checks.csv, summary.md,
metrics.json. Default out: output/proposed-model/rescoring/occlusion.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\rescore_occlusion.py
    .venv\\Scripts\\python scripts\\rescore_occlusion.py --names exp1-occl exp1
    smoke: --runs <run dir> --max-folds 1 --out output/smoke/rescore-occlusion
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.scoring import fit_cue_template
from eegtraj.data.lift_streams import STREAM_START_BEFORE_LED_S, build_subject_streams
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import input_channel_names
from eegtraj.models.decoder import StreamDecoder
from eegtraj.models.decoder_runner import (OCCLUSION_GROUPS, build_encoder, cue_phase_params, occluded_lifts,
                                           precompute_tokens, predict, score_subject)
from eegtraj.models.landmark_template import TemplateBank, fit_phase_prior, fold_templates
from eegtraj.utils.run import setup_run

logger = logging.getLogger("rescore_occlusion")
DECODER = Path("output/proposed-model/decoder")
SCORES = ("r", "residual_r_specific", "residual_r_matched")


def occlusion_runs(names: list[str]) -> list[Path]:
    out = []
    for name in names:
        for run in sorted(DECODER.glob(f"{name}__*/*")):
            if not (run / "metrics.json").exists() or "smoke" in run.name or not (run / "models").is_dir():
                continue
            if json.loads((run / "metrics.json").read_text(encoding="utf-8")).get("occlusion_groups"):
                out.append(run)
    return out


STREAMS: dict = {}
SETUPS: dict = {}  # fold references, identical for every run with the same data, split and templates


def fold_setup(cfg: dict, fold, names: list[str]) -> tuple:
    key = (cfg["cache_dir"], cfg["split_config"], fold.name, tuple(cfg["templates"]["conditions"]),
           int(cfg["templates"]["n_grid"]))
    if key not in SETUPS:
        subj = {s: STREAMS.setdefault((cfg["cache_dir"], s), build_subject_streams(cfg["cache_dir"], s)) for s in names}
        train_samples = [subj[s].samples for s in fold.train_subjects]
        masks_all = {s: np.ones(subj[s].samples.sample_idx.size, bool) for s in names}
        ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0,
                         train_masks={s: masks_all[s] for s in fold.train_subjects}, val_masks={}, logger=logger)
        template = fit_cue_template(train_samples, ctx)
        tmpl = fold_templates([subj[s].samples for s in names], fold, conditions=key[3], n_grid=key[4])
        tau0, s0 = cue_phase_params(train_samples)
        prior = fit_phase_prior([subj[s].samples for s in names], fold.train_subjects, tmpl.fractions)
        cue = {}
        for s in fold.test_subjects:
            pred = template.predict(subj[s].samples, masks_all[s], ctx)
            cue |= {(s, l.trial_row): pred[l.sample_pos].astype(np.float32) for l in subj[s].lifts}
        SETUPS[key] = (subj, ctx, template, tmpl, tau0, s0, prior, cue)
    return SETUPS[key]


def rescore_run(run: Path, device, max_folds: int | None) -> tuple[list[dict], list[dict]]:
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    m = cfg["model"]
    enc_cfg = m["encoder"] if m["neural"] else {"source": "none"}
    groups = list(met["occlusion_groups"])
    channel_names = input_channel_names(cfg["cache_dir"], "P1", "broadband", "all")
    folds = [f for f in make_folds(load_split_config(cfg["split_config"]), cfg.get("folds", "loso"))
             if (run / "models" / f"{f.name}.pt").exists()][:max_folds]
    drops, checks = [], []
    saved = np.load(run / "predictions.npz")
    for fold in folds:
        t0 = time.time()
        names = sorted(set(fold.train_subjects) | set(fold.val_subjects) | set(fold.test_subjects),
                       key=lambda s: int(s[1:]))
        subj, ctx, template, tmpl, tau0, s0, prior, cue = fold_setup(cfg, fold, names)
        prior_dict = {"t0": prior.t0, "dt": prior.dt, "mu": prior.mu, "sd": prior.sd} if m.get("phase_prior", True) else None
        ckpt = torch.load(run / "models" / f"{fold.name}.pt", map_location="cpu", weights_only=False)
        if abs(ckpt["tau0"] - tau0) > 1e-9 or abs(ckpt["s0"] - s0) > 1e-9:
            raise RuntimeError(f"{run} {fold.name}: rebuilt tau0 / s0 differ from the saved model")
        enc, _ = build_encoder(enc_cfg, cfg["cache_dir"], names[0])
        model = StreamDecoder(m["head"], enc, bool(m["ocular"]), TemplateBank.from_templates(tmpl),
                              freeze_encoder=m["setting"] == "frozen", d_fused=int(m["d_fused"]),
                              d_ocular=int(m["d_ocular"]), ocular_channels=int(m["ocular_channels"]),
                              a_max=float(m["a_max"]), tau0=tau0, s0=s0, clock_log_range=float(m["clock_log_range"]),
                              stream_start_s=-STREAM_START_BEFORE_LED_S, phase_prior=prior_dict,
                              sigma_log_range=float(m["sigma_log_range"]), dropout=float(m["dropout"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        frozen = model.encoder is not None and m["setting"] == "frozen"
        for s in fold.test_subjects:
            lifts = subj[s].lifts
            tokens = precompute_tokens(model, lifts, device) if frozen else None
            full, _, arrays = score_subject(subj[s], predict(model, lifts, cue, tokens, device, cfg["train"]),
                                            template, ctx, cfg)
            diff = float(np.abs(arrays["y"] - saved[f"{s}__y"]).max())
            checks.append({"run": run.as_posix(), "subject": s, "group": "none", "max_abs_diff_y": diff,
                           "r_diff": full["r"] - met["per_subject"][s]["r"],
                           "rrs_diff": full["residual_r_specific"] - met["per_subject"][s]["residual_r_specific"]})
            for g in groups:
                rows = [channel_names.index(c) for c in OCCLUSION_GROUPS[g]]
                occ = occluded_lifts(lifts, rows)
                occ_tokens = precompute_tokens(model, occ, device) if frozen else None
                o, _, _ = score_subject(subj[s], predict(model, occ, cue, occ_tokens, device, cfg["train"]),
                                        template, ctx, cfg)
                saved_occ = met["occlusion"][s][g]["scores"]
                checks.append({"run": run.as_posix(), "subject": s, "group": g, "max_abs_diff_y": np.nan,
                               "r_diff": o["r"] - saved_occ["r"],
                               "rrs_diff": o["residual_r_specific"] - saved_occ["residual_r_specific"]})
                drops.append({"run": run.as_posix(), "name": run.parent.name, "seed": met["decoder_seed"],
                              "subject": s, "group": g, **{f"full_{k}": full[k] for k in SCORES},
                              **{f"occluded_{k}": o[k] for k in SCORES},
                              **{f"drop_{k}": full[k] - o[k] for k in SCORES}})
            logger.info("%s %s test %s: max |y diff| %.2e, %.1f s", run.parent.name, fold.name, s, diff,
                        time.time() - t0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    saved.close()
    return drops, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", nargs="*", default=["exp1-occl", "exp1"], help="run name prefixes (before __)")
    parser.add_argument("--runs", type=Path, nargs="*", default=None, help="explicit run directories")
    parser.add_argument("--max-folds", type=int, default=None, help="smoke: first N folds per run")
    parser.add_argument("--out", type=Path, default=Path("output/proposed-model/rescoring/occlusion"))
    args = parser.parse_args()
    runs = args.runs if args.runs else occlusion_runs(args.names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = setup_run(args.out, {"runs": [r.as_posix() for r in runs], "max_folds": args.max_folds,
                                   "device": str(device)})
    logging.getLogger("eegtraj.baselines").setLevel(logging.WARNING)
    drops, checks = [], []
    for run in runs:
        d, c = rescore_run(run, device, args.max_folds)
        drops += d
        checks += c
    drops, checks = pd.DataFrame(drops), pd.DataFrame(checks)
    drops.to_csv(run_dir / "drops.csv", index=False)
    checks.to_csv(run_dir / "checks.csv", index=False)
    worst = {k: float(checks[k].abs().max()) for k in ("max_abs_diff_y", "r_diff", "rrs_diff")}
    table = drops.groupby(["name", "group"])[[f"drop_{k}" for k in SCORES]].mean().round(4)
    md = (f"# Occlusion drops with residual_r_matched\n\n{len(runs)} runs, device {device}. Checks against the saved "
          f"run (max |difference|): unoccluded y {worst['max_abs_diff_y']:.2e}, r {worst['r_diff']:.2e}, rrs "
          f"{worst['rrs_diff']:.2e}.\n\nMean drop (full - occluded) over seeds and subjects:\n\n"
          + table.to_markdown() + "\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"n_runs": len(runs), "checks_max_abs": worst}, indent=2) + "\n",
                                          encoding="utf-8")
    print(md)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
