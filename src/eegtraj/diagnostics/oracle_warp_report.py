"""Run and report the oracle-warp ceiling (see ``eegtraj.diagnostics.oracle_warp``). ANALYSIS ONLY.

For every LOSO fold: fit the cue-locked and the oracle template on the fold's training subjects (same masks and
code as the template baselines), then fit every warp variant to each trial of the test subject. Scoring uses the
D7 metrics and the residual metrics with the same seed as ``scripts/compute_residual_metrics.py``. Saved EEG
decoder predictions are scored the same way, as references.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.runner import trial_filter
from eegtraj.baselines.template import TemplateBaseline, TemplateCueBaseline, time_since_event
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import SubjectSamples, load_subject_samples, sample_masks
from eegtraj.diagnostics.oracle_warp import (
    N_PARAMS,
    PHASE_EVENTS,
    VARIANTS,
    Bounds,
    FitSettings,
    Template,
    TrialData,
    fit_all,
)
from eegtraj.diagnostics.residual import SubjectTemplates, score_predictions
from eegtraj.metrics import METRIC_DEFINITIONS, aggregate_subject, per_trial_metrics

logger = logging.getLogger(__name__)
IDENTITY_TOL_CM = 1e-4  # float32 template vs float64 interpolation
META = ("series", "lift", "block_type", "weight_g", "surface", "onset_mismatch")


def settings_from_config(cfg: dict) -> FitSettings:
    b = cfg.get("bounds", {})
    bounds = Bounds(**{k: tuple(v) if isinstance(v, list) else v for k, v in b.items()})
    fit = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("fit", {}).items()}
    return FitSettings(bounds=bounds, **fit)


def phase_landmarks(trials: pd.DataFrame) -> np.ndarray:
    """(n_trials, 5) seconds since LED-on of ``PHASE_EVENTS``."""
    return trials[list(PHASE_EVENTS)].to_numpy(np.float64) - trials["ev_LEDOn"].to_numpy(np.float64)[:, None]


def fold_models(subjects: list[SubjectSamples], fold) -> tuple[Template, TemplateCueBaseline, TemplateBaseline, FitContext, dict]:
    """Cue and oracle templates fitted on the fold's training subjects, and the cue template as a ``Template``."""
    masks = sample_masks(fold, subjects)
    ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0, train_masks=masks["train"],
                     val_masks=masks["val"], fold_name=fold.name, logger=logger)
    train = [s for s in subjects if s.subject in fold.train_subjects]
    cue, oracle = TemplateCueBaseline(), TemplateBaseline()
    cue.fit(train, [], ctx)
    oracle.fit(train, [], ctx)
    marks = np.concatenate([phase_landmarks(s.trials.iloc[np.unique(s.trial_row[masks["train"][s.subject]])])
                            for s in train])
    template = Template(cue.template, cue.bin0, cue.fs_hz, np.median(marks, axis=0))
    return template, cue, oracle, ctx, masks


def subject_trials(s: SubjectSamples, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scored samples of a test subject: t (s since LED-on, bin grid), bins, y_true (n, 3), trial_row."""
    bins = np.round(time_since_event(s, "ev_LEDOn", mask) * s.fs_hz).astype(np.int64)
    return bins / s.fs_hz, bins, s.target[:, s.sample_idx[mask]].T.astype(np.float64), s.trial_row[mask]


def fit_subject(s: SubjectSamples, mask: np.ndarray, template: Template, settings: FitSettings,
                cue_pred: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Fit every variant to every trial of one test subject. Returns predictions per variant and parameter rows."""
    t, _, y, rows = subject_trials(s, mask)
    identity = template(t)
    diff = float(np.max(np.abs(identity - cue_pred)))
    if diff > IDENTITY_TOL_CM:
        raise AssertionError(f"{s.subject}: identity warp differs from template_cue by {diff} cm")
    preds = {v: np.empty_like(y) for v in VARIANTS}
    marks = phase_landmarks(s.trials)
    rt_template = template.onset_s
    dur_template = template.landmarks_s[-1] - template.landmarks_s[0]
    out_rows = []
    for row in pd.unique(rows):
        m = rows == row
        trial = TrialData(t[m], y[m], marks[row])
        tr = s.trials.iloc[row]
        meta = {"subject": s.subject, **{k: tr[k] for k in META}, "rt_s": marks[row][0],
                "rt_dev_s": marks[row][0] - rt_template, "duration_s": marks[row][-1] - marks[row][0],
                "duration_ratio": (marks[row][-1] - marks[row][0]) / dur_template, "n_samples": int(m.sum())}
        for name, fit in fit_all(trial, template, settings).items():
            preds[name][m] = fit.pred
            out_rows.append(meta | {"variant": name, "cost": fit.cost, "rmse_cm": float(np.sqrt(2 * fit.cost / fit.pred.size)),
                                    "success": fit.success, "at_bound": ";".join(fit.at_bound)} | fit.params)
    return preds, out_rows


def _fit_job(s: SubjectSamples, mask: np.ndarray, template: Template, cue_pred: np.ndarray,
             settings: FitSettings) -> tuple[dict, list[dict], float]:
    t0 = time.time()
    preds, rows = fit_subject(s, mask, template, settings, cue_pred)
    return preds, rows, time.time() - t0


def score_method(predictions: dict[str, dict[str, np.ndarray]], templates: dict[str, SubjectTemplates],
                 residual_seed: int, n_permutations: int) -> dict[str, dict]:
    """Per-subject D7 metrics plus residual metrics. ``predictions[subject]`` holds y_true, y_pred, trial_row."""
    flat = {f"{s}__{k}": v for s, d in predictions.items() for k, v in d.items()}
    residual = {r["subject"]: r for r in score_predictions(flat, templates, residual_seed, n_permutations)}
    out = {}
    for s, d in predictions.items():
        agg = aggregate_subject(per_trial_metrics(d["y_true"], d["y_pred"], d["trial_row"]))
        out[s] = {m: agg[m]["mean"] for m in ("r", "r2", "nrmse")} | {
            m: residual[s][m] for m in ("residual_r", "residual_r_other_trial", "residual_r_specific")}
        out[s] |= {f"r_{a}": v for a, v in agg["r"]["per_axis"].items()}
    return out


def reference_predictions(run_dirs: list[str], templates: dict[str, SubjectTemplates]) -> list[dict[str, dict]]:
    """Saved decoder predictions, one dict per seed, checked against the template samples."""
    out = []
    for d in run_dirs:
        with np.load(Path(d) / "predictions.npz") as z:
            subj = sorted({k.split("__")[0] for k in z.keys()}, key=lambda x: int(x[1:]))
            seed = {}
            for s in subj:
                if s not in templates:
                    continue
                keep = np.isin(z[f"{s}__trial_row"], templates[s].trial_row)  # only differs with --max-trials
                seed[s] = {k: z[f"{s}__{k}"][keep] for k in ("y_true", "y_pred", "trial_row")}
            out.append(seed)
    return out


def summary_table(per_subject: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over subjects per method, with the paired difference to template_cue."""
    metrics = ["r", "r2", "nrmse", "residual_r_specific"]
    cue = per_subject[per_subject.method == "template_cue"].set_index("subject")
    rows = []
    for method, g in per_subject.groupby("method", sort=False):
        g = g.set_index("subject")
        row = {"method": method, "kind": g["kind"].iloc[0], "n_params": g["n_params"].iloc[0], "n_subjects": len(g)}
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1)
        for m in ("r", "r2", "nrmse"):
            d = (g[m] - cue.loc[g.index, m])
            row[f"d_{m}_vs_cue"] = d.mean()
            row[f"n_better_{m}"] = int((d < 0).sum() if m == "nrmse" else (d > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_summary(params: pd.DataFrame) -> pd.DataFrame:
    """Per variant and subject: medians/IQR of the timing parameters and their Spearman rho with RT and duration."""
    rows = []
    for (variant, subject), g in params.groupby(["variant", "subject"], sort=False):
        row = {"variant": variant, "subject": subject, "n_trials": len(g),
               "frac_at_bound": float((g.at_bound.fillna("") != "").mean()), "frac_failed": float((~g.success).mean())}
        for col in ("tau_s", "scale", "amp_x", "amp_y", "amp_z", "offset_x", "offset_y", "offset_z", "bp_shift_onset",
                    "reach_duration_s", "rmse_cm"):
            if col in g and g[col].notna().any():
                q = g[col].quantile([0.25, 0.5, 0.75])
                row[f"{col}_median"], row[f"{col}_iqr"] = q[0.5], q[0.75] - q[0.25]
        if "tau_s" in g and g.tau_s.notna().any():
            row["rho_tau_rt"] = g.tau_s.corr(g.rt_dev_s, method="spearman")
        if "scale" in g and g.scale.notna().any():
            row["rho_scale_duration"] = g.scale.corr(g.duration_ratio, method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def bound_hits(params: pd.DataFrame) -> pd.DataFrame:
    """Number of trials per variant with each parameter at a bound."""
    rows = []
    for variant, g in params.groupby("variant", sort=False):
        names = pd.Series([n for s in g.at_bound.fillna("") for n in s.split(";") if n])
        for name, n in names.value_counts().items():
            rows.append({"variant": variant, "parameter": name, "n_trials": int(n), "frac": n / len(g)})
    return pd.DataFrame(rows, columns=["variant", "parameter", "n_trials", "frac"])


def run(cfg: dict, run_dir: Path) -> dict:
    t_start = time.time()
    settings = settings_from_config(cfg)
    folds = make_folds(load_split_config(cfg["splits"]["config"]), cfg["splits"]["mode"])
    if cfg.get("subjects"):
        folds = [f for f in folds if set(f.test_subjects) & set(cfg["subjects"])]
    names = sorted({s for f in folds for s in (*f.train_subjects, *f.val_subjects, *f.test_subjects)}, key=lambda x: int(x[1:]))
    subjects = [trial_filter(load_subject_samples(cfg["cache_dir"], s, "broadband", "car", "all", cfg["target"], 1),
                             cfg.get("exclude_flags", []), None) for s in names]
    logging.getLogger("eegtraj.baselines").setLevel(logging.WARNING)

    methods: dict[str, dict[str, dict]] = {m: {} for m in ("template_cue", "template", *VARIANTS)}
    templates: dict[str, SubjectTemplates] = {}
    param_rows: list[dict] = []
    landmark_rows = []
    jobs = []
    for fold in folds:
        template, cue, oracle, ctx, masks = fold_models(subjects, fold)
        landmark_rows.append({"fold": fold.name, **{e: v for e, v in zip(PHASE_EVENTS, template.landmarks_s)}})
        for s in subjects:
            if s.subject not in fold.test_subjects or (cfg.get("subjects") and s.subject not in cfg["subjects"]):
                continue
            mask = masks["test"][s.subject].copy()
            if cfg.get("max_trials"):
                keep_rows = np.unique(s.trial_row[mask])[: cfg["max_trials"]]
                mask &= np.isin(s.trial_row, keep_rows)
            cue_pred, oracle_pred = cue.predict(s, mask, ctx), oracle.predict(s, mask, ctx)
            _, bins, y, rows = subject_trials(s, mask)
            templates[s.subject] = SubjectTemplates(y.astype(np.float32), rows, cue_pred, oracle_pred, bins)
            base = {"y_true": y, "trial_row": rows}
            methods["template_cue"][s.subject] = base | {"y_pred": cue_pred}
            methods["template"][s.subject] = base | {"y_pred": oracle_pred}
            jobs.append((replace(s, eeg=np.empty((0, 0), np.float32)), mask, template, cue_pred))  # workers need no EEG

    t0 = time.time()
    results = Parallel(n_jobs=cfg.get("n_jobs", 1))(delayed(_fit_job)(*job, settings) for job in jobs)
    for (s, *_), (preds, prow, seconds) in zip(jobs, results):
        param_rows += prow
        for v, p in preds.items():
            methods[v][s.subject] = {"y_true": methods["template_cue"][s.subject]["y_true"],
                                     "trial_row": methods["template_cue"][s.subject]["trial_row"], "y_pred": p}
        logger.info("%s: %d trials fitted in %.0f s (worker time)", s.subject, len({r["lift"] + 100 * r["series"] for r in prow}), seconds)
    logger.info("All fits: %.0f s with n_jobs=%s", time.time() - t0, cfg.get("n_jobs", 1))

    res_cfg = cfg["residual"]
    kinds = {"template_cue": "floor", "template": "oracle_reference"} | {v: "oracle_warp" for v in VARIANTS}
    long_rows = []
    for method, preds in methods.items():
        for s, row in score_method(preds, templates, res_cfg["seed"], res_cfg["n_permutations"]).items():
            long_rows.append({"method": method, "kind": kinds[method], "n_params": N_PARAMS.get(method, np.nan),
                              "subject": s, **row})
    for name, dirs in (cfg.get("reference_runs") or {}).items():
        seeds = [score_method(p, templates, res_cfg["seed"], res_cfg["n_permutations"])
                 for p in reference_predictions(dirs, templates)]
        for s in seeds[0]:
            vals = pd.DataFrame([sd[s] for sd in seeds]).mean()
            long_rows.append({"method": name, "kind": "eeg_decoder", "n_params": np.nan, "subject": s, **vals.to_dict(),
                              "n_seeds": len(seeds)})
    per_subject = pd.DataFrame(long_rows)
    table = summary_table(per_subject)
    params = pd.DataFrame(param_rows)
    psum, hits = parameter_summary(params), bound_hits(params)

    per_subject.to_csv(run_dir / "per_subject.csv", index=False)
    table.to_csv(run_dir / "summary.csv", index=False)
    params.to_csv(run_dir / "params.csv", index=False)
    psum.to_csv(run_dir / "params_summary.csv", index=False)
    hits.to_csv(run_dir / "bound_hits.csv", index=False)
    pd.DataFrame(landmark_rows).to_csv(run_dir / "template_landmarks.csv", index=False)
    np.savez_compressed(run_dir / "oracle_predictions.npz",  # not predictions.npz: compute_residual_metrics scans those
                        **{f"{m}__{s}__{k}": v for m, d in methods.items() for s, sd in d.items() for k, v in sd.items()})
    metrics = {
        "analysis_only": True,
        "note": "every warp parameter is fitted to the test trial's own trajectory (oracle); not a decoder",
        "metric_definitions": METRIC_DEFINITIONS,
        "n_params": N_PARAMS,
        "aggregate": {r["method"]: {k: r[k] for k in r if k != "method"} for r in table.to_dict("records")},
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    logger.info("Wall time %.0f s", metrics["wall_seconds"])
    return {"per_subject": per_subject, "summary": table, "params": params, "params_summary": psum,
            "bound_hits": hits, "methods": methods, "templates": templates}


# ---------------------------------------------------------------- figures

OVERLAY_METHODS = ("template_cue", "template", "tau_scale", "tau_scale_amp_offset", "piecewise_fit", "min_jerk")


def pick_overlay_trials(per_trial_cue: pd.DataFrame, quantiles: tuple[float, ...]) -> list[tuple[str, int]]:
    """(subject, trial_row) of the trials at the given quantiles of the template_cue per-trial r."""
    d = per_trial_cue.sort_values("r").reset_index(drop=True)
    idx = sorted({int(round(q * (len(d) - 1))) for q in quantiles})
    return [(d.subject[i], int(d.trial[i])) for i in idx]


def per_trial_r(methods: dict, method: str) -> pd.DataFrame:
    """Per-trial r (mean over axes) of one method, all subjects."""
    parts = []
    for s, d in methods[method].items():
        df = per_trial_metrics(d["y_true"], d["y_pred"], d["trial_row"]).groupby("trial", as_index=False)["r"].mean()
        parts.append(df.assign(subject=s))
    return pd.concat(parts, ignore_index=True)


def plot_overlays(methods: dict, templates: dict, trials: list[tuple[str, int]], fs_hz: float, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"template_cue": dict(color="0.5", ls="--"), "template": dict(color="0.3", ls=":"),
              "tau_scale": dict(color="tab:blue"), "tau_scale_amp_offset": dict(color="tab:cyan"),
              "piecewise_fit": dict(color="tab:orange"), "min_jerk": dict(color="tab:green")}
    for s, row in trials:
        tpl = templates[s]
        m = tpl.trial_row == row
        t = tpl.time_bin[m] / fs_hz
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for a, ax in enumerate(axes):
            ax.plot(t, tpl.y_true[m, a], color="k", lw=2, label="ground truth")
            for name in OVERLAY_METHODS:
                ax.plot(t, methods[name][s]["y_pred"][m, a], lw=1.2, label=name, **styles[name])
            ax.set_ylabel(f"{'xyz'[a]} (cm)")
        axes[-1].set_xlabel("time since LED-on (s)")
        axes[0].legend(fontsize=7, ncol=4, loc="upper left")
        fig.suptitle(f"{s}, trial row {row}: oracle-warp fits (ANALYSIS ONLY, parameters fitted to this trial)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{s}_row{row}.png", dpi=110)
        plt.close(fig)


def plot_parameters(params: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for s, g in params[params.variant == "tau"].groupby("subject", sort=False):
        axes[0].scatter(g.rt_dev_s, g.tau_s, s=6, alpha=0.6, label=s)
    lim = [-1, 2]
    axes[0].plot(lim, lim, color="k", lw=0.8)
    axes[0].set(xlim=lim, ylim=lim, xlabel="RT - template median RT (s)", ylabel="fitted tau (s), variant tau",
                title="time shift vs reaction time")
    axes[0].legend(fontsize=6, ncol=2, markerscale=2)
    for s, g in params[params.variant == "tau_scale"].groupby("subject", sort=False):
        axes[1].scatter(g.duration_ratio, g.scale, s=6, alpha=0.6, label=s)
    axes[1].plot([0.5, 2], [0.5, 2], color="k", lw=0.8)
    axes[1].set(xlabel="(hand stop - onset) / template", ylabel="fitted scale, variant tau_scale",
                title="time scale vs movement duration")
    fig.tight_layout()
    fig.savefig(out_dir / "parameters_vs_behavior.png", dpi=110)
    plt.close(fig)


def plot_summary(per_subject: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(dict.fromkeys(per_subject.method))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, metric in zip(axes, ("r", "r2", "residual_r_specific")):
        for i, m in enumerate(order):
            v = per_subject.loc[per_subject.method == m, metric]
            ax.barh(i, v.mean(), color="0.8")
            ax.scatter(v, np.full(len(v), i), s=8, color="k")
        cue = per_subject.loc[per_subject.method == "template_cue", metric].mean()
        if np.isfinite(cue):
            ax.axvline(cue, color="tab:red", lw=0.8)
        ax.set_title(metric)
    axes[0].set_yticks(range(len(order)), order, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Oracle-warp ceiling (oracle variants are ANALYSIS ONLY); red line: template_cue", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_by_method.png", dpi=110)
    plt.close(fig)
