"""Why does the Kalman decoder have a strongly negative R^2 (diagnostics stage)?

Two parts:

- Prediction analysis (read-only on earlier runs). For every trial and axis,
  R^2 = 2 r a - a^2 - b^2, with a = SD(pred) / SD(true) and b = mean(pred - true) / SD(true).
  This separates amplitude errors (a) and offset errors (b) from correlation (r).
- Refit analysis. Refits the decoder on selected folds, then checks the filter:
  stability of A, whether the claimed error matches the real error, whiteness of the
  observation noise and innovations, and counterfactuals with inflated R or other k.

The filter in :func:`run_filter` is written separately from ``KalmanBaseline._filter`` and
is compared against it, which also cross-checks the decoder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegtraj.baselines.base import FitContext, make_baseline  # noqa: E402
from eegtraj.baselines.kalman import KalmanBaseline, _state  # noqa: E402
from eegtraj.baselines.runner import SMOKE_TRIALS_PER_SUBJECT, select_folds, set_seeds, trial_filter  # noqa: E402
from eegtraj.data.windows import (  # noqa: E402
    SubjectSamples,
    fit_eeg_normalizers,
    fit_target_normalizer,
    load_subject_samples,
    sample_masks,
)
from eegtraj.metrics import aggregate_subject, per_trial_metrics  # noqa: E402
from eegtraj.metrics.trajectory import AXES  # noqa: E402

logger = logging.getLogger(__name__)

FS_HZ = 100.0
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3


def _trial_slices(trial_row: np.ndarray) -> list[slice]:
    """Slices of the contiguous runs of equal ``trial_row`` values."""
    edges = np.r_[0, np.flatnonzero(np.diff(trial_row)) + 1, len(trial_row)]
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def r2_decomposition(y_true: np.ndarray, y_pred: np.ndarray, trial_row: np.ndarray) -> pd.DataFrame:
    """Per trial and axis: r, a, b, r2, with r2 = 2 r a - a^2 - b^2.

    Args:
        y_true, y_pred: (n_samples, n_axes), same units.
        trial_row: (n_samples,) trial index; each trial must be contiguous.

    Returns:
        DataFrame with columns trial, axis, r, a, b, r2, term_corr (2ra), term_amp (a^2), term_offset (b^2).
        A constant prediction has r set to 0, since a = 0 makes its term vanish anyway.
    """
    rows = []
    for sl in _trial_slices(trial_row):
        t, p = np.asarray(y_true[sl], np.float64), np.asarray(y_pred[sl], np.float64)
        sd_t, sd_p = t.std(0), p.std(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = ((t - t.mean(0)) * (p - p.mean(0))).mean(0)
            r = np.where(sd_p > 0, cov / (sd_t * sd_p), 0.0)
            a, b = sd_p / sd_t, (p.mean(0) - t.mean(0)) / sd_t
        for i in range(t.shape[1]):
            rows.append({"trial": int(trial_row[sl.start]), "axis": AXES[i], "r": r[i], "a": a[i], "b": b[i],
                         "r2": 2 * r[i] * a[i] - a[i] ** 2 - b[i] ** 2, "term_corr": 2 * r[i] * a[i],
                         "term_amp": a[i] ** 2, "term_offset": b[i] ** 2})
    return pd.DataFrame(rows)


def rmse_by_time(y_true: np.ndarray, y_pred: np.ndarray, trial_row: np.ndarray, bin_s: float, max_s: float,
                 fs_hz: float = FS_HZ) -> tuple[np.ndarray, np.ndarray]:
    """RMSE over axes vs time since the segment start. Returns bin starts (s) and RMSE (NaN if empty)."""
    n_bins = int(round(max_s / bin_s))
    sq, count = np.zeros(n_bins), np.zeros(n_bins)
    for sl in _trial_slices(trial_row):
        e = ((np.asarray(y_true[sl], np.float64) - y_pred[sl]) ** 2).mean(1)
        bins = (np.arange(len(e)) / fs_hz / bin_s + 1e-9).astype(int)
        keep = bins < n_bins
        np.add.at(sq, bins[keep], e[keep])
        np.add.at(count, bins[keep], 1)
    with np.errstate(invalid="ignore"):
        return np.arange(n_bins) * bin_s, np.sqrt(sq / count)


def lag_autocorr(e: np.ndarray, trial_row: np.ndarray, lags: list[int]) -> dict[int, float]:
    """Mean over columns of the within-trial autocorrelation of ``e`` (n, d) around zero, at each lag."""
    den = (e ** 2).sum(0)
    out = {}
    for lag in lags:
        same = trial_row[lag:] == trial_row[:-lag]
        out[lag] = float(np.mean((e[lag:] * e[:-lag])[same].sum(0) / den))
    return out


@dataclass
class FilterTrace:
    """Output of :func:`run_filter`. n = selected samples, T = longest trial."""

    pred: np.ndarray  # (n, 3) position in target units
    state: np.ndarray  # (n, 6) normalized posterior state
    white_innov: np.ndarray  # (n, k) innovations whitened with the predicted S_t; N(0, I) if consistent
    claimed_var: np.ndarray  # (T, 6) posterior variance the filter claims, normalized units
    gain_norm: np.ndarray  # (T,) Frobenius norm of K_t


def run_filter(model: KalmanBaseline, s: SubjectSamples, mask: np.ndarray, ctx: FitContext, k: int,
               r_scale: float = 1.0, prior_only: bool = False) -> FilterTrace:
    """Causal filter over one subject with R scaled by ``r_scale``; ``prior_only`` sets all gains to 0."""
    H, c, R = model.models[k]
    R = R * r_scale
    z = model._z(s, mask, ctx)[:, :k] - c
    rows = s.trial_row[mask]
    starts = np.array([sl.start for sl in _trial_slices(rows)])
    lengths = np.diff(np.r_[starts, len(rows)])
    n_steps = int(lengths.max())
    x = np.tile(model.x0, (len(starts), 1))
    P = model.P0
    states, white = np.empty((len(rows), x.shape[1])), np.empty((len(rows), k))
    claimed, gain_norm = np.empty((n_steps, x.shape[1])), np.empty(n_steps)
    eye = np.eye(x.shape[1])
    for t in range(n_steps):
        live = lengths > t
        pos = starts[live] + t
        if t > 0:
            P = model.A @ P @ model.A.T + model.Q
            x[live] = x[live] @ model.A.T
        S = H @ P @ H.T + R
        K = np.zeros((x.shape[1], k)) if prior_only else np.linalg.solve(S, H @ P).T
        innov = z[pos] - x[live] @ H.T
        white[pos] = np.linalg.solve(np.linalg.cholesky(S), innov.T).T
        x[live] = x[live] + innov @ K.T
        P = (eye - K @ H) @ P
        states[pos] = x[live]
        claimed[t], gain_norm[t] = np.diag(P), np.linalg.norm(K)
    pred = ctx.target_norm.inverse(states[:, :3].T.astype(np.float32)).T
    return FilterTrace(pred, states, white, claimed, gain_norm)


def filter_health(model: KalmanBaseline, k: int, trace: FilterTrace) -> dict[str, Any]:
    """Stability of A, sizes of Q and R, and convergence of the covariance and the gain."""
    H, _, R = model.models[k]
    mags = np.sort(np.abs(np.linalg.eigvals(model.A)))[::-1]
    with np.errstate(divide="ignore"):
        tau_s = -1.0 / (FS_HZ * np.log(mags))  # negative: |eigenvalue| > 1, the mode grows
    t_last = len(trace.gain_norm) - 1
    return {"k": k, "A_eig_magnitude": mags.tolist(), "A_time_constant_s": tau_s.tolist(),
            "x0": model.x0.tolist(), "P0_diag": np.diag(model.P0).tolist(), "Q_diag": np.diag(model.Q).tolist(),
            "R_eig_min_max": [float(np.linalg.eigvalsh(R).min()), float(np.linalg.eigvalsh(R).max())],
            "H_frobenius": float(np.linalg.norm(H)),
            "claimed_pos_var_t0": trace.claimed_var[0, :3].tolist(),
            "claimed_pos_var_t100": trace.claimed_var[min(100, t_last), :3].tolist(),
            "claimed_pos_var_last": trace.claimed_var[t_last, :3].tolist(),
            "gain_norm_t0_t100_last": [float(trace.gain_norm[0]), float(trace.gain_norm[min(100, t_last)]),
                                       float(trace.gain_norm[t_last])]}


def consistency(model: KalmanBaseline, s: SubjectSamples, mask: np.ndarray, ctx: FitContext, k: int,
                trace: FilterTrace, lags: list[int]) -> dict[str, Any]:
    """Does the filter's noise model match this subject's data?

    Returns the normalized innovation squared per dimension (1 if consistent), autocorrelation of
    observation noise and innovations (about 0 if white), and real / claimed position error variance.
    """
    rows, idx = s.trial_row[mask], s.sample_idx[mask]
    H, c, _ = model.models[k]
    x_true = _state(s, idx, ctx)
    obs_noise = model._z(s, mask, ctx)[:, :k] - c - x_true @ H.T
    err = trace.state[:, :3] - x_true[:, :3]
    t_in_trial = np.concatenate([np.arange(sl.stop - sl.start) for sl in _trial_slices(rows)])
    claimed_mean = trace.claimed_var[t_in_trial, :3].mean(0)
    return {"nis_per_dim": float((trace.white_innov ** 2).mean()),
            "obs_noise_autocorr": {str(a): v for a, v in lag_autocorr(obs_noise, rows, lags).items()},
            "innovation_autocorr": {str(a): v for a, v in lag_autocorr(trace.white_innov, rows, lags).items()},
            "pos_error_var_real": (err ** 2).mean(0).tolist(), "pos_error_var_claimed": claimed_mean.tolist(),
            "pos_error_var_ratio": ((err ** 2).mean(0) / claimed_mean).tolist()}


def _score(s: SubjectSamples, mask: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = s.target[:, s.sample_idx[mask]].T
    agg = aggregate_subject(per_trial_metrics(true, pred, s.trial_row[mask]))
    dec = r2_decomposition(true, pred, s.trial_row[mask])
    return {"r": agg["r"]["mean"], "r2": agg["r2"]["mean"], "nrmse": agg["nrmse"]["mean"],
            "term_amp": float(dec.term_amp.mean()), "term_offset": float(dec.term_offset.mean())}


def analyze_fold(model: KalmanBaseline, test: SubjectSamples, train: list[SubjectSamples], masks: dict,
                 ctx: FitContext, d: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Health and consistency at the chosen k, plus scores for each (k, r_scale) on test and train subjects."""
    lags = [int(v) for v in d["autocorr_lags_samples"]]
    m_test = masks["test"][test.subject]
    trace = run_filter(model, test, m_test, ctx, model.k)
    max_diff = float(np.abs(trace.pred - model.predict(test, m_test, ctx)).max())
    logger.info("Independent filter vs decoder on %s: max |difference| %.2e target units", test.subject, max_diff)
    report = {"test_subject": test.subject, "chosen_k": model.k, "val_r_per_k": model.val_scores,
              "independent_filter_max_diff": max_diff, "health": filter_health(model, model.k, trace),
              "consistency_test": consistency(model, test, m_test, ctx, model.k, trace, lags), "consistency_train": {}}
    scored = train[: int(d["n_train_subjects_scored"])]
    for s in scored:
        m = masks["train"][s.subject]
        report["consistency_train"][s.subject] = consistency(model, s, m, ctx, model.k,
                                                             run_filter(model, s, m, ctx, model.k), lags)
    table = []
    for part, subjects, part_masks in (("test", [test], masks["test"]), ("train", scored, masks["train"])):
        for s in subjects:
            m = part_masks[s.subject]
            prior = run_filter(model, s, m, ctx, model.k, prior_only=True)
            table.append({"fold_test": test.subject, "part": part, "subject": s.subject, "k": model.k,
                          "r_scale": float("inf"), **_score(s, m, prior.pred)})
            for k in model.grid:
                for scale in d["r_scales"]:
                    res = run_filter(model, s, m, ctx, k, r_scale=float(scale))
                    table.append({"fold_test": test.subject, "part": part, "subject": s.subject, "k": k,
                                  "r_scale": float(scale), **_score(s, m, res.pred)})
    return report, table


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_traces(data: dict[str, Any], subject: str, n_trials: int, path: Path) -> None:
    """Target, Kalman, and ridge per axis for evenly spaced trials of one subject."""
    rows = data["kalman"][f"{subject}__trial_row"]
    unique = np.unique(rows)
    chosen = unique[np.linspace(0, len(unique) - 1, n_trials).astype(int)]
    fig, axes = plt.subplots(3, n_trials, figsize=(3.6 * n_trials, 6), sharex="col", sharey="row", facecolor=SURFACE)
    for col, trial in enumerate(chosen):
        m = rows == trial
        t = np.arange(m.sum()) / FS_HZ
        for a, name in enumerate(AXES):
            ax = axes[a, col]
            _style(ax)
            ax.plot(t, data["kalman"][f"{subject}__y_true"][m, a], color=TEXT, lw=2, label="Target")
            ax.plot(t, data["kalman"][f"{subject}__y_pred"][m, a], color=SERIES[0], lw=1.5, label="Kalman")
            ax.plot(t, data["ridge_lag"][f"{subject}__y_pred"][m, a], color=SERIES[1], lw=1.5, label="Ridge")
            if col == 0:
                ax.set_ylabel(f"{name} (cm)", color=MUTED, fontsize=9)
            if a == 0:
                ax.set_title(f"{subject}, trial row {int(trial)}", color=TEXT, fontsize=9, loc="left")
            if a == 2:
                ax.set_xlabel("Time since segment start (s)", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    _save(fig, path)


def plot_rmse_by_time(curves: dict[str, dict[str, tuple]], path: Path) -> None:
    """Across-subject mean RMSE vs time, with Kalman's per-subject curves behind it."""
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    _style(ax)
    for t, e in curves["kalman"].values():
        ax.plot(t, e, color=SERIES[0], lw=0.8, alpha=0.3)
    for i, name in enumerate(("kalman", "ridge_lag")):
        t = next(iter(curves[name].values()))[0]
        mean = np.nanmean([e for _, e in curves[name].values()], axis=0)
        ax.plot(t, mean, color=SERIES[i], lw=2, label=name)
    ax.set_xlabel("Time since segment start (s)", color=MUTED, fontsize=9)
    ax.set_ylabel("RMSE (cm)", color=MUTED, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title("Prediction error by time in trial; thin lines are Kalman per subject", color=TEXT, fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def plot_decomposition(summary: pd.DataFrame, path: Path) -> None:
    """Stacked bars of R^2 = 2 r a - a^2 - b^2 per model (mean over subjects)."""
    order = summary.sort_values("r2").index
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(order) + 1.6), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(order))
    s = summary.loc[order]
    ax.barh(y, s.term_corr, height=0.6, color=SERIES[0], label="+2ra (correlation)")
    ax.barh(y, -s.term_amp, height=0.6, color=SERIES[1], edgecolor=SURFACE, linewidth=2, label="-a² (amplitude)")
    ax.barh(y, -s.term_offset, left=-s.term_amp, height=0.6, color=SERIES[2], edgecolor=SURFACE, linewidth=2,
            label="-b² (offset)")
    for yi, (_, row) in zip(y, s.iterrows()):
        ax.text(max(row.term_corr, 0) + 0.1, yi, f"R² {row.r2:.2f}", va="center", color=TEXT, fontsize=8)
    ax.set_yticks(y, order)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(False, axis="y")
    ax.set_xlabel("Contribution to R²", color=MUTED, fontsize=9)
    ax.set_title("What makes R² negative", color=TEXT, fontsize=10, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0)
    _save(fig, path)


def plot_counterfactuals(df: pd.DataFrame, k: int, path: Path) -> None:
    """R^2 and r vs the R multiplier on test subjects at one k; dashed lines are prior only (no EEG)."""
    test = df[(df.part == "test") & (df.k == k)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, facecolor=SURFACE)
    for ax, metric, label in zip(axes, ("r2", "r"), ("R²", "r")):
        _style(ax)
        for i, (subject, g) in enumerate(test.groupby("subject", sort=False)):
            curve = g[np.isfinite(g.r_scale)].sort_values("r_scale")
            ax.plot(curve.r_scale, curve[metric], color=SERIES[i], lw=2, marker="o", ms=5, label=subject)
            ax.axhline(g[~np.isfinite(g.r_scale)][metric].item(), color=SERIES[i], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("R multiplier (diagnostic only)", color=MUTED, fontsize=9)
        ax.set_title(f"{label} on the test subject, k = {k}; dashed = prior only", color=TEXT, fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    _save(fig, path)


def analyze_predictions(d: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Read-only analysis of earlier runs. Writes CSVs and figures; returns the summary."""
    data = {name: np.load(Path(p) / "predictions.npz") for name, p in d["source_runs"].items()}
    rows, curves = [], {}
    for name, npz in data.items():
        for s in sorted({key.split("__")[0] for key in npz.files}, key=lambda v: int(v[1:])):
            y_true, y_pred, trial_row = npz[f"{s}__y_true"], npz[f"{s}__y_pred"], npz[f"{s}__trial_row"]
            dec = r2_decomposition(y_true, y_pred, trial_row)
            for axis, g in dec.groupby("axis", sort=False):
                means = g[["r", "a", "b", "r2", "term_corr", "term_amp", "term_offset"]].mean().to_dict()
                rows.append({"model": name, "subject": s, "axis": axis, **means, "r2_median": g.r2.median(),
                             "a_median": g.a.median()})
            curves.setdefault(name, {})[s] = rmse_by_time(y_true, y_pred, trial_row, d["time_bin_s"], d["time_max_s"])
    per_axis = pd.DataFrame(rows)
    per_axis.to_csv(run_dir / "r2_decomposition.csv", index=False, float_format="%.4f")
    summary = per_axis.drop(columns=["subject", "axis"]).groupby("model").mean()
    summary.to_csv(run_dir / "r2_decomposition_summary.csv", float_format="%.4f")
    logger.info("R2 decomposition, mean over subjects and axes:\n%s", summary.round(3).to_string())
    logger.info("Kalman per axis:\n%s", per_axis[per_axis.model == "kalman"].drop(columns=["model", "subject"])
                .groupby("axis").mean().round(3).to_string())
    figures = run_dir / "figures"
    for s in d["trace_subjects"]:
        plot_traces(data, s, int(d["trace_trials_per_subject"]), figures / f"traces_{s}.png")
    plot_rmse_by_time(curves, figures / "rmse_by_time.png")
    plot_decomposition(summary, figures / "r2_decomposition.png")
    return summary.to_dict("index")


def refit_folds(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Refit the decoder on the selected folds and run :func:`analyze_fold`. Writes counterfactuals.csv."""
    set_seeds(int(cfg["seed"]))
    w = cfg["data"]["windows"]
    history, lag_step = int(w["history_samples"]), int(w["ridge_lag_step_samples"])
    max_trials = SMOKE_TRIALS_PER_SUBJECT if cfg.get("smoke") else None
    loaded: dict[str, SubjectSamples] = {}

    def get(subject: str) -> SubjectSamples:
        if subject not in loaded:
            s = load_subject_samples(w["cache_dir"], subject, cfg["data"]["band"], cfg["data"]["reference"],
                                     cfg["data"]["channel_set"], w["target"], history)
            loaded[subject] = trial_filter(s, cfg.get("exclude_flags", []), max_trials)
        return loaded[subject]

    reports, table = [], []
    for fold in select_folds(cfg):
        names = sorted(set(fold.train_subjects) | set(fold.val_subjects) | set(fold.test_subjects),
                       key=lambda s: int(s[1:]))
        subjects = [get(s) for s in names]
        masks = sample_masks(fold, subjects)
        train = [s for s in subjects if s.subject in fold.train_subjects]
        val = [s for s in subjects if s.subject in fold.val_subjects]
        test = [s for s in subjects if s.subject in fold.test_subjects]
        ctx = FitContext(eeg_norm=fit_eeg_normalizers(subjects, fold.train_subjects, w["eeg_normalization"]),
                         target_norm=fit_target_normalizer(train, fold.train_subjects, train_masks=masks["train"]),
                         history=history, lag_step=lag_step, seed=int(cfg["seed"]), train_masks=masks["train"],
                         val_masks=masks["val"], smoke=bool(cfg.get("smoke")), run_dir=run_dir, logger=logger)
        model = make_baseline(cfg["baseline"]["name"], cfg["baseline"].get("params"))
        model.fit(train, val, ctx)
        for s in test:
            report, rows = analyze_fold(model, s, train, masks, ctx, cfg["diagnostics"])
            reports.append(report)
            table.extend(rows)
            logger.info("Fold %s: %s", fold.name, json.dumps(report, indent=1))
    df = pd.DataFrame(table)
    df.to_csv(run_dir / "counterfactuals.csv", index=False, float_format="%.4f")
    logger.info("Counterfactuals (r_scale inf = prior only):\n%s", df.round(3).to_string())
    plot_counterfactuals(df, max(cfg["baseline"]["params"]["n_components"]), run_dir / "figures" / "counterfactuals.png")
    return {"folds": reports}


def run(cfg: dict[str, Any], run_dir: Path, skip_refit: bool = False) -> dict[str, Any]:
    """Run both parts and write metrics.json."""
    results: dict[str, Any] = {"smoke": bool(cfg.get("smoke")), "seed": int(cfg["seed"]),
                               "r2_identity": "per trial and axis: r2 = 2 r a - a^2 - b^2",
                               "predictions": analyze_predictions(cfg["diagnostics"], run_dir)}
    if not skip_refit:
        results["refit"] = refit_folds(cfg, run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
