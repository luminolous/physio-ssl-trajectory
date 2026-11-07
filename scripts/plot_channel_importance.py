"""Scalp maps of channel importance: occlusion drops per model and Haufe patterns of the lagged ridge.

Reads every run in output/diagnostics/pre-design/channel-importance/<config>/<run-id>/ (occlusion.csv, and
models/*.npz for ridge). Runs of one config are seeds and are averaged per subject first.
Importance = score with nothing occluded minus score with the channel occluded, per test subject, then the mean
over subjects. Positive = the model needs that channel.

Writes to a new output/diagnostics/pre-design/channel-importance/summary/<run-id>/:
    occlusion_<config>.png       topographies of the drop in r and in residual_r_specific
    ridge_patterns.png           Haufe patterns per lag and axis (mean over folds)
    channels.csv, groups.csv, regions.csv, summary.md

Usage (from repo root):
    .venv\\Scripts\\python scripts\\plot_channel_importance.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats

from eegtraj.preprocessing.reference import CHANNEL_SETS

ROOT = Path("output/diagnostics/pre-design/channel-importance")
CACHE = Path("dataset/cache/98f5fa6e66")
METRICS = ("r", "residual_r_specific")
REGIONS = {"prefrontal": CHANNEL_SETS["prefrontal"], "contralateral": CHANNEL_SETS["contralateral"],
           "ipsilateral": CHANNEL_SETS["ipsilateral"], "occipital": ("O1", "Oz", "O2", "PO9", "PO10"),
           "temporal": CHANNEL_SETS["temporal"]}
RIDGE_LAGS_MS = (0, 100, 200, 300, 480)


def eeg_info() -> mne.Info:
    names = json.loads((CACHE / "P1" / "manifest.json").read_text(encoding="utf-8"))["eeg_names"]
    info = mne.create_info(names, 100.0, "eeg")
    info.set_montage("standard_1005")
    return info


def drops(config_dir: Path) -> pd.DataFrame:
    """Per subject and occluded input: drop in each metric, averaged over seeds."""
    parts = []
    for f in sorted(config_dir.glob("2*/occlusion.csv")):
        d = pd.read_csv(f)
        base = d[d.kind == "none"].set_index("subject")[list(METRICS)]
        d = d[d.kind != "none"].copy()
        for m in METRICS:
            d[m] = d.subject.map(base[m]) - d[m]
        d["run"] = f.parent.name
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).groupby(["kind", "occluded", "subject"], as_index=False)[list(METRICS)].mean()


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, name), g in d.groupby(["kind", "occluded"]):
        row = {"kind": kind, "occluded": name, "n_subjects": len(g)}
        for m in METRICS:
            v = g[m].to_numpy()
            row[f"drop_{m}"], row[f"sd_{m}"], row[f"n_positive_{m}"] = v.mean(), v.std(ddof=1), int((v > 0).sum())
            row[f"wilcoxon_p_{m}"] = float(scipy.stats.wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_occlusion(table: pd.DataFrame, info: mne.Info, title: str, path: Path) -> None:
    ch = table[table.kind == "channel"].set_index("occluded").reindex(info.ch_names)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4))
    for ax, m in zip(axes, METRICS):
        v = ch[f"drop_{m}"].to_numpy()
        lim = float(np.nanmax(np.abs(v)))
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim), names=info.ch_names)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"drop in {m}")
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ridge_patterns(config_dir: Path, n_chans: int) -> tuple[np.ndarray, np.ndarray] | None:
    files = sorted(config_dir.glob("2*/models/*.npz"))
    if not files:
        return None
    pats, lags = [], None
    for f in files:
        z = np.load(f)
        lags = z["lags_samples"]
        pats.append(z["pattern"].reshape(n_chans, len(lags), 3))  # channel-major features
    return np.mean(pats, axis=0), lags


def plot_ridge(pattern: np.ndarray, lags: np.ndarray, info: mne.Info, path: Path) -> pd.DataFrame:
    picks = [int(np.argmin(np.abs(lags * 10 - ms))) for ms in RIDGE_LAGS_MS]
    fig, axes = plt.subplots(3, len(picks), figsize=(2.6 * len(picks), 7.5))
    for a, axis in enumerate("xyz"):
        lim = float(np.abs(pattern[:, picks, a]).max())
        for j, k in enumerate(picks):
            mne.viz.plot_topomap(pattern[:, k, a], info, axes=axes[a, j], show=False, cmap="RdBu_r", vlim=(-lim, lim))
            axes[a, j].set_title(f"{axis}, EEG {lags[k] * 10} ms before")
    fig.suptitle("Lagged ridge (lowfreq), Haufe patterns, mean over LOSO folds")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    norm = np.linalg.norm(pattern, axis=2)  # (C, lags), over x, y, z
    return pd.DataFrame(norm, index=info.ch_names, columns=[f"lag_{int(l) * 10}ms" for l in lags])


def main() -> None:
    info = eeg_info()
    out = ROOT / "summary" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    channels, groups, regions, lines = [], [], [], []
    for config_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "summary"):
        d = drops(config_dir)
        if d.empty:
            continue
        n_runs = len(list(config_dir.glob("2*/occlusion.csv")))
        table = summarize(d)
        table.insert(0, "config", config_dir.name)
        plot_occlusion(table, info, f"{config_dir.name}: occlusion drop, mean over {table.n_subjects.max()} test subjects, "
                                    f"{n_runs} seed(s)", out / f"occlusion_{config_dir.name}.png")
        channels.append(table[table.kind == "channel"])
        groups.append(table[table.kind == "group"])
        ch = table[table.kind == "channel"].set_index("occluded")
        for region, names in REGIONS.items():
            row = {"config": config_dir.name, "region": region, "n_channels": len(names)}
            for m in METRICS:
                row[f"mean_channel_drop_{m}"] = float(ch.loc[list(names), f"drop_{m}"].mean())
                row[f"rank_share_{m}"] = float((ch[f"drop_{m}"].rank(ascending=False).loc[list(names)] <= 8).mean())
            regions.append(row)
        top = ch.sort_values("drop_residual_r_specific", ascending=False).head(8)
        lines.append(f"- `{config_dir.name}`: top 8 channels by drop in residual_r_specific: "
                     + ", ".join(f"{n} ({v:.3f})" for n, v in top.drop_residual_r_specific.items()))
        found = ridge_patterns(config_dir, len(info.ch_names))
        if found is not None:
            norm = plot_ridge(*found, info, out / "ridge_patterns.png")
            norm.to_csv(out / "ridge_pattern_norm.csv")
            peak = norm.mean(axis=1).sort_values(ascending=False).head(8)
            lines.append("- `ridge_lag` Haufe pattern norm (mean over lags), top 8: "
                         + ", ".join(f"{n} ({v:.3f})" for n, v in peak.items()))
    channels, groups, regions = pd.concat(channels), pd.concat(groups), pd.DataFrame(regions)
    channels.to_csv(out / "channels.csv", index=False)
    groups.to_csv(out / "groups.csv", index=False)
    regions.to_csv(out / "regions.csv", index=False)
    gcols = ["config", "occluded", "n_subjects", "drop_r", "n_positive_r", "wilcoxon_p_r", "drop_residual_r_specific",
             "n_positive_residual_r_specific", "wilcoxon_p_residual_r_specific"]
    (out / "summary.md").write_text(
        "# Channel importance\n\nDrop = score with nothing occluded minus score with the input occluded (set to its "
        "normalizer mean), per LOSO test subject, mean over subjects. rank_share = fraction of the region's channels "
        "among the 8 largest drops. CAR spreads each channel into all others, so single-channel drops understate "
        "regions; see the group occlusions.\n\n## Groups\n\n" + groups[gcols].round(3).to_markdown(index=False)
        + "\n\n## Regions (single-channel drops)\n\n" + regions.round(3).to_markdown(index=False) + "\n\n## Top channels\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(groups[gcols].round(3).to_string(index=False))
    print(regions.round(3).to_string(index=False))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
