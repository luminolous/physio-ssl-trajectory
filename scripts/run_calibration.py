"""Calibration within subject: subject cue template, LOSO model with and without fine-tuning, scratch on k lifts.

For each LOSO test subject, k in ``ks`` and each draw: calibration lifts from S1-S5, early stopping on S6, scores on
S7-S9 against the subject cue template of the same k lifts (see eegtraj.diagnostics.calibration). Needs a LOSO
run in residual form with saved models (configs/diagnostics/calibration/loso/).

Writes results.csv (one row per subject, condition, k, draw; rewritten after every fit) and fits.jsonl (training
histories) to <root>/<group>/<name>/<run-id>/. Smoke: 1 subject, 1 draw per k, 2 epochs, under output/smoke/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_calibration.py --config configs\\diagnostics\\calibration\\uncleaned.yaml
        [--smoke] [--subjects P1 P2] [--tag name] [--pretrained-run <run dir>]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eegtraj.baselines.runner import set_seeds
from eegtraj.baselines.scoring import subject_scores
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import Standardizer, load_subject_samples
from eegtraj.diagnostics.calibration import (
    Series,
    calibration_rows,
    context,
    delta_scale,
    fit_seed,
    load_pretrained,
    predict_residual,
    rows_mask,
    scratch_model,
    series_mask,
    subject_template,
    train_residual,
)
from eegtraj.models.trainer import TrainConfig
from eegtraj.utils.run import setup_run

logger = logging.getLogger("calibration")
SMOKE_ROOT = Path("output/smoke")
SMOKE_TRAIN = dict(max_epochs=2)


def newest_run(folder: Path, smoke: bool) -> Path:
    for d in sorted(folder.glob("2*"), reverse=True):
        m = d / "metrics.json"
        if (d / "models").is_dir() and m.exists() and bool(json.loads(m.read_text(encoding="utf-8"))["smoke"]) == smoke:
            return d
    raise FileNotFoundError(f"no {'smoke ' if smoke else ''}run with saved models in {folder}")


def check_no_calibration(pre: Path, subject: str, s, test: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Max |difference| to the LOSO run's own saved predictions on S7-S9 (they must agree), or None."""
    f = pre / "predictions.npz"
    if not f.exists():
        return None
    z = np.load(f)
    if f"{subject}__sample_idx" not in z:
        return None
    pos = {int(i): j for j, i in enumerate(z[f"{subject}__sample_idx"])}
    j = [pos.get(int(i)) for i in s.sample_idx[test]]
    if any(v is None for v in j):
        return None
    return float(np.abs(z[f"{subject}__y_pred"][j] - y_pred).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--tag")
    parser.add_argument("--pretrained-run", type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    w = yaml.safe_load(Path(cfg["data"]["windows_config"]).read_text(encoding="utf-8"))
    folder = Path(cfg["pretrained"])
    if args.smoke:  # smoke LOSO runs live under output/smoke/<group>/<name>/
        folder = SMOKE_ROOT / folder.relative_to(cfg["output"]["root"])
    pre = args.pretrained_run or newest_run(folder, args.smoke)
    cfg.update(smoke=args.smoke, pretrained_run=pre.as_posix(), windows=w)
    subjects = [f.test_subjects[0] for f in make_folds(load_split_config(cfg["splits"]["config"]), "loso")]
    subjects = [s for s in subjects if (pre / "models" / f"loso_{s}.pt").exists()]
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    if args.smoke:
        subjects = subjects[:1]
    cfg["subjects"] = subjects
    out = cfg["output"]
    root = SMOKE_ROOT if args.smoke else Path(out["root"])
    run_dir = setup_run(root / out["group"] / out["name"], cfg, tag=args.tag)
    print(f"Run directory: {run_dir}")

    seed, history = int(cfg["seed"]), int(w["history_samples"])
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = {**cfg["train"], **(SMOKE_TRAIN if args.smoke else {})}
    train_cfg = {c: TrainConfig(lr=float(cfg["lr"][c]), **base) for c in ("finetune", "scratch")}
    n_draws = {str(k): int(v) for k, v in (cfg.get("n_draws") or {}).items()}
    series = Series()
    sc = cfg["scoring"]
    logger.info("Pretrained run %s; subjects %s; conditions %s; ks %s; draws %s; device %s", pre, subjects,
                cfg["conditions"], cfg["ks"], n_draws, device)

    rows: list[dict] = []
    t_start = time.time()
    for subject in subjects:
        s = load_subject_samples(w["cache_dir"], subject, cfg["data"]["band"], cfg["data"]["reference"],
                                 cfg["data"]["channel_set"], w["target"], history, pre_lift=bool(w.get("pre_lift")))
        eeg_norm = Standardizer.fit([s.eeg_for_stats()])  # D9: per subject, label-free
        val, test = series_mask(s, series.validation), series_mask(s, series.test)
        model0, norm0, cross_t, spec = load_pretrained(pre / "models" / f"loso_{subject}.pt", s.eeg.shape[0], history,
                                                       device)
        logger.info("%s: %d lifts, %d val samples (S6), %d test samples (S7-S9)", subject, len(s.trials),
                    int(val.sum()), int(test.sum()))
        no_cal = None
        for k in cfg["ks"]:
            for draw in range(1 if args.smoke else n_draws.get(str(k), 1)):
                cal_rows = calibration_rows(s, k, draw, int(cfg["draw_seed"]), series)
                cal = rows_mask(s, cal_rows)
                ctx = context(s, eeg_norm, cal, val, history, seed, logger)
                tmpl = subject_template(s, cal, ctx)
                key = {"subject": subject, "k": str(k), "draw": draw, "n_cal_lifts": len(cal_rows),
                       "n_cal_samples": int(cal.sum())}

                def record(condition: str, y_pred: np.ndarray, extra: dict | None = None) -> None:
                    scores = subject_scores(s, test, y_pred, tmpl, ctx, int(sc["seed"]), int(sc["n_permutations"]))
                    rows.append({**key, "condition": condition, **scores, **(extra or {})})
                    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
                    logger.info("%s k=%s draw %d %s: r %.3f, R2 %.3f, NRMSE %.3f, specific %.3f", subject, k, draw,
                                condition, scores["r"], scores["r2"], scores["nrmse"], scores["residual_r_specific"])

                if "template" in cfg["conditions"]:
                    record("template", tmpl.predict(s, test, ctx))
                if "no_calibration" in cfg["conditions"]:
                    if no_cal is None:
                        no_cal = predict_residual(model0, s, test, cross_t, norm0, ctx, train_cfg["finetune"], device)
                        diff = check_no_calibration(pre, subject, s, test, no_cal)
                        logger.info("%s: no-calibration vs saved LOSO predictions on S7-S9, max |diff| %s", subject,
                                    "n/a" if diff is None else f"{diff:.2e}")
                    record("no_calibration", no_cal)
                for condition in ("finetune", "scratch"):
                    if condition not in cfg["conditions"]:
                        continue
                    t0 = time.time()
                    fit_ctx = replace(ctx, seed=fit_seed(seed, subject, k, draw, condition))
                    if condition == "finetune":
                        model, norm = copy.deepcopy(model0), norm0
                    else:
                        norm = delta_scale(s, tmpl, ctx)
                        model = scratch_model(spec["architecture"], s.eeg.shape[0], history, spec["model_kwargs"],
                                              fit_ctx.seed).to(device)
                    model, hist = train_residual(model, s, tmpl, norm, fit_ctx, train_cfg[condition], device)
                    y_pred = predict_residual(model, s, test, tmpl, norm, ctx, train_cfg[condition], device)
                    best = min(hist, key=lambda h: h["val_loss"])
                    seconds = round(time.time() - t0, 1)
                    record(condition, y_pred, {"best_epoch": best["epoch"], "epochs_run": len(hist), "seconds": seconds})
                    with (run_dir / "fits.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({**key, "condition": condition, "seed": fit_ctx.seed,
                                            "lr": train_cfg[condition].lr, "history": hist}) + "\n")
                    del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("%s done, %.1f min since start", subject, (time.time() - t_start) / 60)

    df = pd.DataFrame(rows)
    summary = df.groupby(["condition", "k"])[["r", "r2", "nrmse", "residual_r_specific"]].mean().round(3)
    logger.info("Mean over subjects and draws:\n%s", summary.to_string())
    logger.info("Wall time %.1f min", (time.time() - t_start) / 60)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
