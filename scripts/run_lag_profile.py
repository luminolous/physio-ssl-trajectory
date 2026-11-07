"""Lag profile: single-lag ridge on 0.1-2 Hz EEG, LOSO, per filter, channel set, and lag. Analysis only.

Positive lag = EEG before the kinematic sample; negative = EEG after it. These are not decoders.

Writes to output/diagnostics/pre-design/lag-profile/<run-id>/:
    per_subject.csv   test scores per filter, channel set, lag, and subject (r, R2, residual_r_specific, ...)
    lag_profile.csv   mean and SEM over subjects, plus the chosen alpha per fold
    lag_profile.png   r and residual_r_specific against lag
    alphas.csv, dropped_samples.csv

Scored samples whose EEG at -500 ms would fall after their lift's end are dropped (the same samples for every lag).

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_lag_profile.py --config configs\\diagnostics\\lag_profile.yaml [--smoke]

--smoke uses 2 folds and 4 lags.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from eegtraj.baselines import FitContext
from eegtraj.baselines.runner import trial_filter
from eegtraj.baselines.scoring import fit_cue_template
from eegtraj.baselines.template import time_since_event
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import Standardizer, load_subject_samples, sample_masks
from eegtraj.diagnostics.lag_profile import check_lag_range, fit_ridge, keep_lag_range, lag_features, mean_trial_r, subject_stats
from eegtraj.metrics import aggregate_subject, per_trial_metrics, residual_r, residual_r_other_trial
from eegtraj.preprocessing.reference import channel_indices
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/diagnostics/pre-design/lag-profile")
logger = logging.getLogger("lag_profile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tag")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["smoke"] = args.smoke
    run_dir = setup_run(OUTPUT if not args.smoke else Path("output/smoke/lag-profile"), cfg, tag=args.tag)
    t_start = time.time()

    folds = make_folds(load_split_config(cfg["splits_config"]), "loso")
    lm = cfg["lags_ms"]
    lags = np.arange(lm["min"], lm["max"] + 1, lm["step"]) // 10  # samples at 100 Hz
    if args.smoke:
        folds, lags = folds[:2], np.array([-50, 0, 30, 100])
    n_pre = int(lags.max())
    alphas = np.asarray(cfg["alphas"], dtype=np.float64)
    seed, n_perm = int(cfg["scoring"]["seed"]), int(cfg["scoring"]["n_permutations"])
    names = sorted({s for f in folds for s in (*f.train_subjects, *f.val_subjects, *f.test_subjects)}, key=lambda s: int(s[1:]))
    logger.info("%d folds, %d lags (%d to %d samples), subjects %s", len(folds), len(lags), lags.min(), lags.max(), names)

    rows, alpha_rows, dropped_rows = [], [], []
    for filt, band in cfg["bands"].items():
        subjects = {}
        for name in names:
            s = load_subject_samples(cfg["cache_dir"], name, band, cfg["reference"], "all", cfg["target"], n_pre + 1, pre_lift=True)
            s = trial_filter(s, cfg["exclude_flags"], 20 if args.smoke else None)
            n_all = len(s.sample_idx)
            s, dropped = keep_lag_range(s, n_pre, int(lags.min()), int(lags.max()))
            check_lag_range(s, n_pre, int(lags.min()), int(lags.max()))
            dropped_rows.append({"filter": filt, "subject": name, "n_scored": n_all, "n_dropped": dropped})
            subjects[name] = s
        if cfg["eeg_normalization"] != "per_subject":
            raise ValueError("only per_subject EEG normalization is implemented")
        z_all = {n: Standardizer.fit([s.eeg_for_stats()]).transform(s.eeg).astype(np.float32) for n, s in subjects.items()}
        eeg_names = json.loads((Path(cfg["cache_dir"]) / names[0] / "manifest.json").read_text(encoding="utf-8"))["eeg_names"]
        for chans in cfg["channel_sets"]:
            ch = channel_indices(eeg_names, chans)
            z = {n: v[ch] for n, v in z_all.items()}
            t_set = time.time()
            stats = {(n, int(l)): subject_stats(z[n], s.sample_idx, s.target[:, s.sample_idx].T, int(l))
                     for n, s in subjects.items() for l in lags}
            for fold in folds:
                part = [subjects[n] for n in names]
                masks = sample_masks(fold, part)
                if any(not m.all() for d in masks.values() for m in d.values() if m.any()):
                    raise ValueError("LOSO masks must cover whole subjects")
                ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0, train_masks=masks["train"],
                                 val_masks=masks["val"], fold_name=fold.name)
                template = fit_cue_template([subjects[n] for n in fold.train_subjects], ctx)
                test = {}
                for n in fold.test_subjects:
                    s, m = subjects[n], masks["test"][n]
                    test[n] = (s, s.target[:, s.sample_idx].T, template.predict(s, m, ctx),
                               np.round(time_since_event(s, "ev_LEDOn", m) * s.fs_hz).astype(np.int64))
                for l in lags:
                    l = int(l)
                    weights = fit_ridge([stats[(n, l)] for n in fold.train_subjects], alphas)
                    val = {a: np.mean([mean_trial_r(subjects[n].target[:, subjects[n].sample_idx].T,
                                                    lag_features(z[n], subjects[n].sample_idx, l) @ w + b, subjects[n].trial_row)
                                       for n in fold.val_subjects]) for a, (w, b) in weights.items()}
                    alpha = max(val, key=val.get)
                    w, b = weights[alpha]
                    alpha_rows.append({"filter": filt, "channels": chans, "lag_ms": l * 10, "fold": fold.name, "alpha": alpha,
                                       "val_r": val[alpha]})
                    for n, (s, y_true, cue, bins) in test.items():
                        y_pred = lag_features(z[n], s.sample_idx, l) @ w + b
                        sc = aggregate_subject(per_trial_metrics(y_true, y_pred, s.trial_row))
                        res = residual_r(y_true, y_pred, cue, s.trial_row)["mean"]
                        rng = np.random.default_rng([seed, int(n[1:])])
                        other = residual_r_other_trial(y_true, y_pred, cue, s.trial_row, bins, rng, n_perm)["mean"]
                        rows.append({"filter": filt, "channels": chans, "lag_ms": l * 10, "subject": n,
                                     "r": sc["r"]["mean"], "r2": sc["r2"]["mean"], "nrmse": sc["nrmse"]["mean"],
                                     "residual_r": res, "residual_r_other_trial": other, "residual_r_specific": res - other})
            logger.info("%s %s done in %.0f s", filt, chans, time.time() - t_set)

    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(run_dir / "per_subject.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(run_dir / "alphas.csv", index=False)
    dropped = pd.DataFrame(dropped_rows)
    dropped.to_csv(run_dir / "dropped_samples.csv", index=False)
    logger.info("Samples dropped so every lag stays inside its lift: %s",
                dropped.groupby("subject").apply(lambda d: f"{d.n_dropped.max()}/{d.n_scored.max()}", include_groups=False).to_dict())
    g = per_subject.groupby(["filter", "channels", "lag_ms"])
    profile = g[["r", "r2", "residual_r_specific"]].mean().add_suffix("_mean").join(
        g[["r", "r2", "residual_r_specific"]].sem().add_suffix("_sem")).join(g.size().rename("n_subjects")).reset_index()
    profile.to_csv(run_dir / "lag_profile.csv", index=False)

    fig, axes = plt.subplots(2, len(cfg["channel_sets"]), figsize=(5 * len(cfg["channel_sets"]), 7), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for j, chans in enumerate(cfg["channel_sets"]):
        for i, metric in enumerate(("r", "residual_r_specific")):
            ax = axes[i, j]
            for filt, style in zip(cfg["bands"], ("-", "--")):
                d = profile[(profile["filter"] == filt) & (profile.channels == chans)]
                ax.plot(d.lag_ms, d[f"{metric}_mean"], style, label=filt)
                ax.fill_between(d.lag_ms, d[f"{metric}_mean"] - d[f"{metric}_sem"], d[f"{metric}_mean"] + d[f"{metric}_sem"], alpha=0.2)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{chans}: {metric}")
            ax.set_xlabel("lag (ms), > 0 = EEG before the kinematic sample")
        axes[0, j].legend()
    fig.suptitle("Single-lag ridge, 0.1-2 Hz EEG, LOSO test, mean +- SEM over subjects (analysis only, not a decoder)")
    fig.tight_layout()
    fig.savefig(run_dir / "lag_profile.png", dpi=150)
    peaks = profile.loc[profile.groupby(["filter", "channels"]).residual_r_specific_mean.idxmax(),
                        ["filter", "channels", "lag_ms", "residual_r_specific_mean", "r_mean"]]
    (run_dir / "metrics.json").write_text(json.dumps({"smoke": args.smoke, "peaks_by_specific": peaks.to_dict("records"),
                                                      "wall_seconds": round(time.time() - t_start, 1)}, indent=2) + "\n",
                                          encoding="utf-8")
    logger.info("Peaks by residual_r_specific:\n%s", peaks.round(3).to_string(index=False))
    logger.info("Wall time %.1f s -> %s", time.time() - t_start, run_dir)


if __name__ == "__main__":
    main()
