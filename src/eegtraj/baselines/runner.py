"""Fold loop shared by all baselines: load, normalize (train only), fit, predict, score.

Writes into the run directory:
- ``metrics.json``: metric definitions, protocol facts, per fold, per subject, aggregate
  (mean +- SD across test subjects), pooled r as a labeled secondary metric;
- ``predictions.npz`` (optional): per test subject y_true, y_pred, sample_idx, trial_row.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eegtraj.baselines.base import FitContext, make_baseline
from eegtraj.data.splits import Fold, load_split_config, make_folds
from eegtraj.data.windows import (
    SubjectSamples,
    fit_eeg_normalizers,
    fit_target_normalizer,
    load_subject_samples,
    sample_masks,
)
from eegtraj.metrics import METRIC_DEFINITIONS, aggregate_across_subjects, aggregate_subject, per_trial_metrics, pooled_r

logger = logging.getLogger(__name__)

SMOKE_N_FOLDS = 1
SMOKE_TRIALS_PER_SUBJECT = 20


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def restrict(s: SubjectSamples, keep_trial: np.ndarray) -> SubjectSamples:
    """Drop target samples whose trial row is not in ``keep_trial`` (boolean over trials)."""
    m = keep_trial[s.trial_row]
    return replace(s, sample_idx=s.sample_idx[m], trial_row=s.trial_row[m])


def trial_filter(s: SubjectSamples, exclude_flags: list[str], max_trials: int | None) -> SubjectSamples:
    """Apply flag-based exclusion (sensitivity analyses) and the smoke-run trial limit."""
    keep = np.ones(len(s.trials), dtype=bool)
    for flag in exclude_flags:
        keep &= ~s.trials[flag].astype(bool).to_numpy()
    if max_trials is not None:
        keep &= np.arange(len(s.trials)) < max_trials
    return restrict(s, keep)


def average_repeats(reps: list[dict]) -> dict:
    """Average subject-level results over random repeats (identity for a single repeat)."""
    if len(reps) == 1:
        return reps[0]
    out = {}
    for metric, first in reps[0].items():
        out[metric] = {
            "per_axis": {a: float(np.nanmean([r[metric]["per_axis"][a] for r in reps])) for a in first["per_axis"]},
            "mean": float(np.nanmean([r[metric]["mean"] for r in reps])),
            "n_nan": int(sum(r[metric]["n_nan"] for r in reps)),
            "sd_over_repeats": float(np.std([r[metric]["mean"] for r in reps], ddof=1)),
        }
    return out


def resolve_config(path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Baseline config with the windows and split configs inlined, plus CLI overrides."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["data"]["windows"] = yaml.safe_load(Path(cfg["data"]["windows_config"]).read_text(encoding="utf-8"))
    cfg["splits"]["resolved"] = load_split_config(cfg["splits"]["config"])
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def select_folds(cfg: dict[str, Any]) -> list[Fold]:
    folds = make_folds(cfg["splits"]["resolved"], cfg["splits"]["mode"])
    if cfg.get("subjects"):
        folds = [f for f in folds if set(f.test_subjects) & set(cfg["subjects"])]
        if not folds:
            raise ValueError(f"no fold tests any of {cfg['subjects']} (screening subjects are never test folds)")
    if cfg.get("smoke"):
        folds = folds[:SMOKE_N_FOLDS]
    return folds


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run every fold and write metrics.json (and predictions.npz). Returns the metrics dict."""
    t_start = time.time()
    seed = int(cfg["seed"])
    set_seeds(seed)
    w = cfg["data"]["windows"]
    history, lag_step = int(w["history_samples"]), int(w["ridge_lag_step_samples"])
    max_trials = SMOKE_TRIALS_PER_SUBJECT if cfg.get("smoke") else None
    folds = select_folds(cfg)
    logger.info("Baseline %s, mode %s, %d folds, seed %d, smoke %s", cfg["baseline"]["name"],
                cfg["splits"]["mode"], len(folds), seed, bool(cfg.get("smoke")))

    loaded: dict[str, SubjectSamples] = {}

    def get(subject: str) -> SubjectSamples:
        if subject not in loaded:
            s = load_subject_samples(w["cache_dir"], subject, cfg["data"]["band"], cfg["data"]["reference"],
                                     cfg["data"]["channel_set"], w["target"], history)
            loaded[subject] = trial_filter(s, cfg.get("exclude_flags", []), max_trials)
        return loaded[subject]

    fold_results, subject_results, pooled, predictions, info = [], {}, {}, {}, {}
    per_repeat_subject: dict[str, list[dict]] = {}
    n_repeats = 1
    for fold in folds:
        t_fold = time.time()
        names = sorted(set(fold.train_subjects) | set(fold.val_subjects) | set(fold.test_subjects),
                       key=lambda s: int(s[1:]))
        subjects = [get(s) for s in names]
        masks = sample_masks(fold, subjects)
        train = [s for s in subjects if s.subject in fold.train_subjects]
        val = [s for s in subjects if s.subject in fold.val_subjects]
        test = [s for s in subjects if s.subject in fold.test_subjects]
        n = {part: {s.subject: int(masks[part][s.subject].sum()) for s in subjects if masks[part][s.subject].any()}
             for part in ("train", "val", "test")}
        logger.info("Fold %s: train %s, val %s, test %s; target samples train %d, val %d, test %d",
                    fold.name, fold.train_subjects, fold.val_subjects, fold.test_subjects,
                    sum(n["train"].values()), sum(n["val"].values()), sum(n["test"].values()))

        eeg_norm = fit_eeg_normalizers(subjects, fold.train_subjects, w["eeg_normalization"])
        target_norm = fit_target_normalizer(train, fold.train_subjects, train_masks=masks["train"])
        ctx = FitContext(eeg_norm=eeg_norm, target_norm=target_norm, history=history, lag_step=lag_step,
                         seed=seed, train_masks=masks["train"], val_masks=masks["val"],
                         smoke=bool(cfg.get("smoke")), run_dir=run_dir, logger=logger)
        model = make_baseline(cfg["baseline"]["name"], cfg["baseline"].get("params"))
        model.fit(train, val, ctx)
        n_repeats = model.n_repeats

        fold_subjects = {}
        for s in test:
            m = masks["test"][s.subject]
            y_true = s.target[:, s.sample_idx[m]].T
            reps, pooled_reps = [], []
            for k in range(model.n_repeats):
                y_pred = model.predict(s, m, ctx, repeat=k)
                trial_df = per_trial_metrics(y_true, y_pred, s.trial_row[m])
                reps.append(aggregate_subject(trial_df))
                pooled_reps.append(pooled_r(y_true, y_pred))
                if k == 0:
                    predictions[s.subject] = dict(y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32),
                                                  sample_idx=s.sample_idx[m], trial_row=s.trial_row[m])
            res = average_repeats(reps)
            res["n_trials"] = int(trial_df.trial.nunique())
            res["n_samples"] = int(m.sum())
            subject_results[s.subject] = res
            per_repeat_subject.setdefault(s.subject, reps)
            pooled[s.subject] = {key: float(np.mean([p[key] for p in pooled_reps])) for key in pooled_reps[0]}
            fold_subjects[s.subject] = res
            logger.info("Fold %s test %s: r = %.3f (x %.3f, y %.3f, z %.3f), R2 = %.3f, NRMSE = %.3f, %d trials",
                        fold.name, s.subject, res["r"]["mean"], *res["r"]["per_axis"].values(),
                        res["r2"]["mean"], res["nrmse"]["mean"], res["n_trials"])
        info[fold.name] = model.info()
        fold_results.append({"fold": fold.name, "train_subjects": list(fold.train_subjects),
                             "val_subjects": list(fold.val_subjects), "test_subjects": list(fold.test_subjects),
                             "train_series": list(fold.train_series), "val_series": list(fold.val_series),
                             "test_series": list(fold.test_series), "n_target_samples": n,
                             "baseline_info": info[fold.name], "test": fold_subjects,
                             "seconds": round(time.time() - t_fold, 2)})

    aggregate = aggregate_across_subjects(subject_results)
    repeat_spread = None
    if n_repeats > 1:
        # Aggregate across subjects separately for each random repeat, then summarize the spread.
        per_rep = [aggregate_across_subjects({s: reps[k] for s, reps in per_repeat_subject.items()})
                   for k in range(n_repeats)]
        repeat_spread = {metric: {"n_repeats": n_repeats,
                                  "mean": float(np.mean([p[metric]["mean"] for p in per_rep])),
                                  "sd": float(np.std([p[metric]["mean"] for p in per_rep], ddof=1)),
                                  "min": float(np.min([p[metric]["mean"] for p in per_rep])),
                                  "max": float(np.max([p[metric]["mean"] for p in per_rep]))}
                         for metric in ("r", "r2", "rmse", "nrmse")}
    metrics = {
        "baseline": cfg["baseline"]["name"],
        "split_mode": cfg["splits"]["mode"],
        "smoke": bool(cfg.get("smoke")),
        "seed": seed,
        "target": w["target"],
        "eeg_normalization": w["eeg_normalization"],
        "eeg_normalization_note": ("transductive: each subject's own label-free EEG statistics"
                                   if w["eeg_normalization"] == "per_subject" else "training-subject statistics"),
        "exclude_flags": cfg.get("exclude_flags", []),
        "metric_definitions": METRIC_DEFINITIONS,
        "aggregate": aggregate,
        "n_repeats": n_repeats,
        "repeat_spread": repeat_spread,
        "per_subject": subject_results,
        "pooled_r_secondary": pooled,
        "per_fold": fold_results,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if cfg.get("save_predictions"):
        np.savez_compressed(run_dir / "predictions.npz",
                            **{f"{s}__{k}": v for s, d in predictions.items() for k, v in d.items()})
    logger.info("Aggregate over %d test subjects: r = %.3f +- %.3f, R2 = %.3f +- %.3f, NRMSE = %.3f +- %.3f",
                aggregate["r"]["n_subjects"], aggregate["r"]["mean"], aggregate["r"]["sd"],
                aggregate["r2"]["mean"], aggregate["r2"]["sd"], aggregate["nrmse"]["mean"], aggregate["nrmse"]["sd"])
    if repeat_spread:
        logger.info("Across %d repeats: aggregate r mean %.3f, SD %.4f, range [%.3f, %.3f]", n_repeats,
                    repeat_spread["r"]["mean"], repeat_spread["r"]["sd"], repeat_spread["r"]["min"],
                    repeat_spread["r"]["max"])
    logger.info("Wall time %.1f s", metrics["wall_seconds"])
    return metrics
