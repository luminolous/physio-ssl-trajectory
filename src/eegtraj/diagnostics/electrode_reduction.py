"""Electrode reduction curve: collect the montage runs, score them, and compare them with all 32 channels.

A run group is one config (data x montage x reference x form) with one or more seeds. Per group:

- test, per subject: r, R2, NRMSE (from metrics.json) and residual_r_specific (from predictions.npz, with the
  same templates, seed, and permutations as ``scripts/compute_residual_metrics.py``);
- inner validation, per fold: the mean over the two inner-validation subjects of the same four metrics;
- compute: parameters, median epoch time and peak GPU memory in training (run.log), and batch-1 inference latency
  (``scripts/benchmark_montages.py``).

Seeds are averaged per subject (test) or fold (inner validation) before the paired comparison against the
comparator of the same data and form (all 32 channels). Fewer input channels also means fewer parameters in the
spatial layer, so montage and model capacity are confounded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from eegtraj.diagnostics.residual import SubjectTemplates, score_predictions

METRICS = ("r", "r2", "nrmse", "residual_r_specific")
EPOCH_RE = re.compile(r"epoch \d+: .*\(([\d.]+) s\)")
PEAK_RE = re.compile(r"peak GPU memory ([\d.]+) GB")


@dataclass(frozen=True)
class Group:
    name: str
    data: str  # ica | raw
    montage: str
    reference: str  # car | mastoid
    form: str  # direct | residual
    n_chans: int
    folder: Path

    @property
    def comparator_key(self) -> tuple[str, str]:
        return self.data, self.form


def seed_runs(folder: Path, seeds: tuple[int, ...] | None = None, include_smoke: bool = False) -> list[Path]:
    """Run directories of a group with a metrics.json, optionally only the given seeds. Smoke runs are skipped
    unless ``include_smoke`` (only for a dry run of the comparison)."""
    out = []
    for f in sorted(folder.glob("2*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("smoke") and not include_smoke:
            continue
        if seeds is not None and m.get("seed") not in seeds:
            continue
        out.append(f.parent)
    return out


def run_scores(run_dir: Path, templates: dict[str, SubjectTemplates], residual_seed: int, n_permutations: int
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Test scores per subject, inner-validation scores per fold, and training cost of one run."""
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test = pd.DataFrame({s: {k: v[k]["mean"] if isinstance(v[k], dict) else v[k] for k in ("r", "r2", "nrmse")}
                         for s, v in m["per_subject"].items()}).T
    with np.load(run_dir / "predictions.npz") as z:
        keep = {k: z[k] for k in z.keys() if k.split("__")[0] in templates}
    # Smoke runs score a few trials only: restrict the templates to the saved trials (score_predictions still checks
    # that the saved samples equal the template samples).
    subset = {}
    for s in {k.split("__")[0] for k in keep}:
        t = templates[s]
        sel = np.isin(t.trial_row, keep[f"{s}__trial_row"])
        subset[s] = SubjectTemplates(t.y_true[sel], t.trial_row[sel], t.cue[sel], t.oracle[sel], t.time_bin[sel])
    spec = {r["subject"]: r["residual_r_specific"] for r in score_predictions(keep, subset, residual_seed, n_permutations)}
    test["residual_r_specific"] = pd.Series(spec)
    inner = pd.DataFrame({f["fold"]: {k: np.mean([v[k] for v in f["inner_val"].values()]) for k in METRICS}
                          for f in m["per_fold"] if f.get("inner_val")}).T
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    epochs = [float(x) for x in EPOCH_RE.findall(log)]
    peaks = [float(x) for x in PEAK_RE.findall(log)]
    cost = {"n_params": m["per_fold"][0]["baseline_info"]["n_parameters"],
            "epoch_s_median": float(np.median(epochs)) if epochs else np.nan,
            "train_peak_gpu_gb": max(peaks) if peaks else np.nan, "wall_s": m.get("wall_seconds", np.nan)}
    return test, inner, cost


def group_scores(group: Group, templates: dict[str, SubjectTemplates], residual_seed: int, n_permutations: int,
                 seeds: tuple[int, ...] | None = None, include_smoke: bool = False) -> dict | None:
    """Seed-averaged scores of a group, or None if it has no finished run."""
    runs = seed_runs(group.folder, seeds, include_smoke)
    if not runs:
        return None
    parts = [run_scores(r, templates, residual_seed, n_permutations) for r in runs]
    test = pd.concat([p[0] for p in parts]).groupby(level=0).mean()
    inner = pd.concat([p[1] for p in parts]).groupby(level=0).mean() if any(len(p[1]) for p in parts) else pd.DataFrame()
    cost = pd.DataFrame([p[2] for p in parts]).mean().to_dict()
    return {"test": test, "inner": inner, "cost": cost, "n_seeds": len(runs), "runs": [r.as_posix() for r in runs]}


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Paired difference a - b over shared index; Wilcoxon p when at least 5 pairs differ from 0."""
    d = (a - b).dropna()
    out = {"n": int(d.size), "mean_a": float(a[d.index].mean()) if d.size else np.nan,
           "mean_b": float(b[d.index].mean()) if d.size else np.nan, "diff": float(d.mean()) if d.size else np.nan,
           "n_up": int((d > 0).sum()), "wilcoxon_p": np.nan}
    if (d != 0).sum() >= 5:
        out["wilcoxon_p"] = float(scipy.stats.wilcoxon(d).pvalue)
    return out


def comparison_table(groups: list[Group], scores: dict[str, dict], comparators: dict[tuple[str, str], str]) -> pd.DataFrame:
    """One row per group: mean scores (test, inner validation), paired difference to the comparator, compute."""
    rows = []
    for g in groups:
        sc = scores.get(g.name)
        if sc is None:
            continue
        ref = scores.get(comparators.get(g.comparator_key, ""))
        row = {"group": g.name, "data": g.data, "montage": g.montage, "reference": g.reference, "form": g.form,
               "n_chans": g.n_chans, "n_seeds": sc["n_seeds"]}
        for split in ("inner", "test"):
            df = sc[split]
            for metric in METRICS:
                row[f"{split}_{metric}"] = float(df[metric].mean()) if metric in df else np.nan
                if ref is not None and g.name != comparators.get(g.comparator_key) and metric in df and metric in ref[split]:
                    p = paired(df[metric], ref[split][metric])
                    row[f"{split}_d_{metric}"], row[f"{split}_n_up_{metric}"] = p["diff"], f"{p['n_up']}/{p['n']}"
                    row[f"{split}_p_{metric}"] = p["wilcoxon_p"]
        rows.append(row | sc["cost"])
    return pd.DataFrame(rows)


def plot_curve(table: pd.DataFrame, path: Path) -> None:
    """residual_r_specific (inner validation and test) vs number of channels, per data x reference x form."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    styles = {("ica", "car"): dict(color="tab:blue", marker="o"), ("ica", "mastoid"): dict(color="tab:orange", marker="s"),
              ("raw", "car"): dict(color="tab:gray", marker="^"), ("raw", "mastoid"): dict(color="tab:red", marker="v")}
    for i, form in enumerate(("direct", "residual")):
        for j, split in enumerate(("inner", "test")):
            ax = axes[i, j]
            t = table[table.form == form]
            for (data, ref), g in t.groupby(["data", "reference"]):
                st = styles.get((data, ref), {})
                cent = g[g.montage != "headband_ring"].sort_values("n_chans")
                ring = g[g.montage == "headband_ring"]
                ax.plot(cent.n_chans, cent[f"{split}_residual_r_specific"], label=f"{data}, {ref}", **st)
                if len(ring):
                    ax.scatter(ring.n_chans, ring[f"{split}_residual_r_specific"], s=90, facecolors="none",
                               edgecolors=st.get("color", "k"), linewidths=1.5, marker=st.get("marker", "o"))
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_xticks([1, 3, 6, 9, 10, 32], ["1", "3", "6", "9", "10", "32"])
            ax.set_title(f"{form}, {'inner validation' if split == 'inner' else 'LOSO test'}")
            ax.set_ylabel("residual_r_specific")
        axes[i, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("number of EEG channels (open marker: headband ring)")
    fig.suptitle("Electrode reduction (CNN-LSTM 2 s). Fewer channels also means fewer spatial-layer parameters.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_montages(montages: dict[str, tuple[str, ...]], all_names: list[str], mastoids: tuple[str, ...], path: Path) -> None:
    """Scalp diagram per montage: used channels filled, linked-mastoid reference electrodes marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    info = mne.create_info(all_names, 100.0, "eeg")
    info.set_montage("standard_1005")
    layout = mne.channels.find_layout(info)
    xy = {n: layout.pos[i, :2] + layout.pos[i, 2:] / 2 for i, n in enumerate(layout.names)}
    fig, axes = plt.subplots(1, len(montages), figsize=(3.0 * len(montages), 3.3))
    for ax, (name, chans) in zip(np.atleast_1d(axes), montages.items()):
        pts = np.array([xy[n] for n in all_names])
        ax.scatter(pts[:, 0], pts[:, 1], s=25, facecolors="none", edgecolors="0.6")
        used = np.array([xy[n] for n in chans])
        ax.scatter(used[:, 0], used[:, 1], s=60, color="tab:blue")
        ref = np.array([xy[n] for n in mastoids])
        ax.scatter(ref[:, 0], ref[:, 1], s=60, marker="s", facecolors="none", edgecolors="tab:orange", linewidths=1.5)
        for n in chans:
            ax.annotate(n, xy[n], fontsize=6, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
        ax.set_title(f"{name} ({len(chans)})", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Montages (blue). Orange squares: TP9/TP10, the linked-mastoid reference of option (b).", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
