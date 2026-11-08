"""Report of the ocular-cleaned variant: what ICA removed, and how decoding changed against the uncleaned runs.

Cache part (needs only the cleaned cache): components and variance removed per subject, topographies of the removed
components, and Fp1 and Cz onset-aligned averages before vs after cleaning. Run part (skipped for runs that do not
exist yet): cleaned vs uncleaned for the direct, residual, and prefrontal-only CNN-LSTM, on inner validation (when
both sides have it) and on test; and the shift in occlusion importance.

Writes to a new output/diagnostics/final-checks/ocular/summary/<run-id>/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\report_ocular.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_channel_importance import drops, plot_occlusion, summarize  # noqa: E402

CACHE = Path("dataset/cache/98f5fa6e66")
SUBJECTS = ("P1", "P2", "P4", "P5", "P7", "P8", "P9", "P10", "P11", "P12")
BAND = "broadband_ica"
OUT_ROOT = Path("output/diagnostics/final-checks/ocular")
PRE = Path("output/diagnostics/pre-design")
SUMMARY = Path("output/diagnostics/summary")
PAIRS = {  # cleaned config -> (uncleaned group dir, label)
    "cnn_lstm_direct": (PRE / "screening" / "broadband__all", "direct"),
    "cnn_lstm_residual": (PRE / "residual" / "cnn_lstm", "residual"),
    "cnn_lstm_prefrontal": (PRE / "screening" / "broadband__prefrontal", "prefrontal only"),
}
OCCLUSION_BEFORE = PRE / "channel-importance" / "cnn_lstm_broadband"
METRICS = ("r", "r2", "nrmse", "residual_r_specific")
WINDOW_S = (-1.0, 1.5)


def info_montage(names: list[str]) -> mne.Info:
    info = mne.create_info(names, 100.0, "eeg")
    info.set_montage("standard_1005")
    return info


def components(subjects: list[str]) -> pd.DataFrame:
    rows = []
    for s in subjects:
        c = json.loads((CACHE / s / f"variant_{BAND}_car.json").read_text(encoding="utf-8"))["cleaning"]
        rows.append({"subject": s, "rank": c["rank"], "n_removed": len(c["selected"]),
                     "removed_variance_fit_data": c["removed_variance_fit_data"],
                     "veog_r": max(c["corr_selected"]["VEOG"], default=np.nan), "heog_r": max(c["corr_selected"]["HEOG"], default=np.nan)})
    return pd.DataFrame(rows)


def plot_topographies(subjects: list[str], path: Path) -> None:
    icas = {s: np.load(CACHE / s / f"ica_{BAND}_car.npz") for s in subjects}
    n_col = max(len(z["selected"]) for z in icas.values())
    fig, axes = plt.subplots(len(subjects), max(n_col, 1), figsize=(2.4 * max(n_col, 1), 2.3 * len(subjects)), squeeze=False)
    for i, (s, z) in enumerate(icas.items()):
        info = info_montage(list(z["names"]))
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if j >= len(z["selected"]):
                ax.axis("off")
                continue
            k = int(z["selected"][j])
            mne.viz.plot_topomap(z["mixing"][:, k], info, axes=ax, show=False, cmap="RdBu_r")
            ax.set_title(f"{s} IC{k}: |r| V {z['corr_veog'][k]:.2f}, H {z['corr_heog'][k]:.2f}", fontsize=7)
    fig.suptitle("Removed ICA components (mixing columns, arbitrary units)", y=1.03)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def onset_average(subject: str, band: str, channels: list[str]) -> np.ndarray:
    """Mean over lifts of the given channels from WINDOW_S around kinematic onset (lift samples only). (n_ch, n_t)."""
    names = json.loads((CACHE / subject / "manifest.json").read_text(encoding="utf-8"))["eeg_names"]
    eeg = np.load(CACHE / subject / f"eeg_{band}_car.npy", mmap_mode="r")
    trials = pd.read_csv(CACHE / subject / "trials.csv")
    rel = np.arange(int(WINDOW_S[0] * 100), int(WINDOW_S[1] * 100))
    rows = [names.index(c) for c in channels]
    segs = []
    for t in trials.itertuples():
        j = int(round(t.onset_s * 100)) - 1 + rel  # lift sample j sits at (j + 1) / fs
        if j[0] >= 0 and j[-1] < t.n_samples:
            segs.append(np.asarray(eeg[rows][:, t.offset_start + j]))
    return np.mean(segs, axis=0)


def plot_onset(subjects: list[str], path: Path) -> pd.DataFrame:
    chans = ["Fp1", "Cz"]
    t = np.arange(int(WINDOW_S[0] * 100), int(WINDOW_S[1] * 100)) / 100
    avg = {b: np.stack([onset_average(s, b, chans) for s in subjects]) for b in ("broadband", BAND)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for c, ax in enumerate(axes):
        for b, label in (("broadband", "uncleaned"), (BAND, "ICA-cleaned")):
            m, se = avg[b][:, c].mean(0), avg[b][:, c].std(0, ddof=1) / np.sqrt(len(subjects))
            ax.plot(t, m, label=label)
            ax.fill_between(t, m - se, m + se, alpha=0.2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{chans[c]}, onset-aligned, mean +- SEM over {len(subjects)} subjects")
        ax.set_xlabel("time from kinematic onset (s)")
        ax.set_ylabel("uV (0.1-40 Hz, CAR, causal)")
        ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    win = (t >= 0) & (t <= 0.5)
    return pd.DataFrame([{"channel": ch, "band": b, "mean_0_to_0.5s_uV": float(avg[b][:, c][:, win].mean()),
                          "peak_abs_uV": float(np.abs(avg[b][:, c].mean(0)).max())}
                         for c, ch in enumerate(chans) for b in avg])


def runs(folder: Path) -> list[dict]:
    out = []
    for f in sorted(folder.glob("2*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if not m.get("smoke"):
            m["_dir"] = f.parent.as_posix()
            out.append(m)
    return out


def per_subject(ms: list[dict], metric: str, specific: pd.DataFrame | None) -> pd.Series:
    if metric == "residual_r_specific":
        if specific is None:
            return pd.Series(dtype=float)
        cols = [c for c in specific.columns if any(c == f"{g}/{n}" for g, n in [_group_name(m["_dir"]) for m in ms])]
        return specific[cols[0]] if cols else pd.Series(dtype=float)
    return pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for m in ms]).mean(axis=0)


def _group_name(run_dir: str) -> tuple[str, str]:
    parts = Path(run_dir).parts
    if "final-checks" in parts:
        return "ocular", parts[-2]
    if "screening" in parts:
        return "screening", parts[-2]
    return "residual", parts[-2]


def inner_val(ms: list[dict], metric: str) -> pd.Series:
    if not all(f.get("inner_val") for m in ms for f in m["per_fold"]):
        return pd.Series(dtype=float)
    return pd.DataFrame([{f["fold"]: np.mean([v[metric] for v in f["inner_val"].values()]) for f in m["per_fold"]}
                         for m in ms]).mean(axis=0)


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    if d.size < 2:
        return {}
    return {"cleaned": float(a[d.index].mean()), "uncleaned": float(b[d.index].mean()), "diff": float(d.mean()),
            "n_up": int((d > 0).sum()), "n": int(d.size), "wilcoxon_p": float(scipy.stats.wilcoxon(d).pvalue)}


def main() -> None:
    out = OUT_ROOT / "summary" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    subjects = [s for s in SUBJECTS if (CACHE / s / f"variant_{BAND}_car.json").exists()]
    comp = components(subjects)
    comp.to_csv(out / "components.csv", index=False)
    plot_topographies(subjects, out / "removed_components.png")
    onset = plot_onset(subjects, out / "onset_fp1_cz.png")
    onset.to_csv(out / "onset_averages.csv", index=False)
    lines = ["# Ocular-cleaned variant\n", f"Subjects with a cleaned cache: {', '.join(subjects)}.\n",
             "## Components removed\n", comp.round(3).to_markdown(index=False), "",
             "## Onset-aligned averages (uV)\n", onset.round(2).to_markdown(index=False), ""]
    print(comp.round(3).to_string(index=False))
    print(onset.round(2).to_string(index=False))

    specific = pd.read_csv(SUMMARY / "per_subject_residual_r_specific.csv", index_col=0) \
        if (SUMMARY / "per_subject_residual_r_specific.csv").exists() else None
    rows = []
    for name, (before_dir, label) in PAIRS.items():
        a, b = runs(OUT_ROOT / name), runs(before_dir)
        if not a:
            continue
        for metric in METRICS:
            rows.append({"model": label, "metric": metric, "split": "inner_val", "seeds": f"{len(a)} vs {len(b)}",
                         **paired(inner_val(a, metric), inner_val(b, metric))})
            rows.append({"model": label, "metric": metric, "split": "test", "seeds": f"{len(a)} vs {len(b)}",
                         **paired(per_subject(a, metric, specific), per_subject(b, metric, specific))})
    if rows:
        comparison = pd.DataFrame(rows)
        comparison.to_csv(out / "comparison.csv", index=False)
        lines += ["## Cleaned vs uncleaned\n", comparison.round(3).to_markdown(index=False), ""]
        print(comparison.round(3).to_string(index=False))

    if any((OUT_ROOT / "cnn_lstm_direct").glob("2*/occlusion.csv")):
        info = info_montage(json.loads((CACHE / "P1" / "manifest.json").read_text(encoding="utf-8"))["eeg_names"])
        tables = {}
        for label, folder in (("uncleaned", OCCLUSION_BEFORE), ("cleaned", OUT_ROOT / "cnn_lstm_direct")):
            t = summarize(drops(folder))
            tables[label] = t
            plot_occlusion(t, info, f"CNN-LSTM broadband, {label}: occlusion drop", out / f"occlusion_{label}.png")
        g = {k: v[v.kind == "group"].set_index("occluded") for k, v in tables.items()}
        shift = pd.DataFrame({f"{k}_{m}": g[k][f"drop_{m}"] for k in g for m in ("r", "residual_r_specific")})
        shift.to_csv(out / "occlusion_groups.csv")
        lines += ["## Occlusion, group drops (test)\n", shift.round(3).to_markdown(), ""]
        print(shift.round(3).to_string())
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
