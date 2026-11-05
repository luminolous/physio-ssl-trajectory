"""Compare deep baselines across EEG history lengths, and check the fixed lr used for the comparison.

Reads every non-smoke run under output/diagnostics/history/<model>/history_<n>/ and, as the 0.5 s
reference, the seed runs under output/data-baselines/baselines/<model>/ (same lr and weight decay).
Per-subject values are averaged over seeds first. Each history is compared with the reference on the
seeds it has in common, so a partly finished grid still gives a fair difference.
Writes history.csv, per_subject.csv, flags.csv, lr_check.json, and summary.md to output/diagnostics/history/summary/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_history.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.data.pre_lift import history_flags

ROOT = Path("output/diagnostics/history")
REFERENCE = Path("output/data-baselines/baselines")
OUT_DIR = ROOT / "summary"
CACHE_DIR = "dataset/cache/98f5fa6e66"
MODELS = ("eegnet", "cnn_lstm")
METRICS = ("r", "r2", "nrmse")
FS_HZ = 100.0
REFERENCE_HISTORY = 50
FLAG_HISTORIES = (50, 100, 200, 300)
FLAG_SUBJECTS = ("P1", "P2", "P4", "P5", "P7", "P8", "P9", "P10", "P11", "P12")
LR_TOLERANCE = 0.01  # user decision: larger differences are reported before continuing
FIXED_LR = 0.001


def load_run(path: Path) -> dict | None:
    """Metrics of a finished non-smoke run, with the peak GPU memory from its log."""
    if not (path / "metrics.json").exists():
        return None
    m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if m["smoke"]:
        return None
    peaks = re.findall(r"peak GPU memory ([\d.]+) GB", (path / "run.log").read_text(encoding="utf-8"))
    m["peak_gpu_gb"] = max(map(float, peaks)) if peaks else float("nan")
    return m


def load_all(parent: Path) -> list[dict]:
    runs = [load_run(p) for p in sorted(parent.iterdir()) if p.is_dir()] if parent.exists() else []
    return [m for m in runs if m is not None]


def per_subject(runs: list[dict], metric: str) -> pd.Series:
    return pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for m in runs]).mean(axis=0)


def epoch_seconds(runs: list[dict]) -> float:
    """Median seconds per training epoch; the first epoch of each fold is left out as a warm-up."""
    values = [h["seconds"] for m in runs for f in m["per_fold"] for h in f["baseline_info"]["history"][1:]]
    return float(np.median(values)) if values else float("nan")


def history_row(model: str, history: int, runs: list[dict], reference: list[dict]) -> dict:
    seeds = sorted(m["seed"] for m in runs)
    row = {"model": model, "history_samples": history, "history_s": history / FS_HZ, "seeds": " ".join(map(str, seeds)),
           "n_folds": max(len(m["per_fold"]) for m in runs)}
    same_seeds = [m for m in reference if m["seed"] in seeds]
    for metric in METRICS:
        values = per_subject(runs, metric)
        row[metric] = values.mean()
        row[f"{metric}_sd"] = values.std(ddof=1)
        if same_seeds:
            base = per_subject(same_seeds, metric).reindex(values.index)
            row[f"{metric}_delta_vs_0.5s"] = (values - base).mean()
            row[f"{metric}_n_up"] = int((values > base).sum())
    row |= {"peak_gpu_gb": max(m["peak_gpu_gb"] for m in runs), "epoch_seconds": epoch_seconds(runs),
            "wall_seconds_per_seed": float(np.mean([m["wall_seconds"] for m in runs]))}
    return row


def lr_check(grid: dict, fixed: dict, tolerance: float = LR_TOLERANCE) -> dict:
    """Test-score difference between the lr grid and the fixed lr. Safe only if both r and R² stay within tolerance."""
    out = {"tolerance": tolerance, "seed": grid["seed"], "fixed_lr": FIXED_LR}
    for metric in ("r", "r2"):
        a, b = grid["aggregate"][metric]["mean"], fixed["aggregate"][metric]["mean"]
        out[metric] = {"grid": a, "fixed": b, "difference": a - b}
    out["safe"] = bool(abs(out["r"]["difference"]) <= tolerance and abs(out["r2"]["difference"]) <= tolerance)
    lrs = [f["baseline_info"]["selected_train"]["lr"] for f in grid["per_fold"]]
    out["selected_lr_counts"] = {f"{lr:g}": lrs.count(lr) for lr in sorted(set(lrs))}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, subjects = [], []
    for model in MODELS:
        reference = load_all(REFERENCE / model)
        groups = {REFERENCE_HISTORY: reference}
        for path in sorted((ROOT / model).glob("history_*")):
            groups[int(path.name.split("_")[1])] = load_all(path)
        for history, runs in sorted(groups.items()):
            if not runs:
                continue
            rows.append(history_row(model, history, runs, reference))
            subjects += [{"model": model, "history_s": history / FS_HZ, "subject": s, "r": v}
                         for s, v in per_subject(runs, "r").items()]
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "history.csv", index=False)
    pd.DataFrame(subjects).to_csv(OUT_DIR / "per_subject.csv", index=False)

    flags = pd.DataFrame([{"history_s": h / FS_HZ, "subject": s, **history_flags(CACHE_DIR, s, "broadband", "car", h)}
                          for h in FLAG_HISTORIES for s in FLAG_SUBJECTS])
    flags.to_csv(OUT_DIR / "flags.csv", index=False)
    flag_means = flags.drop(columns="subject").groupby("history_s").mean().reset_index()

    lines = ["# History-length ablation", "", "Fixed lr 1e-3, weight decay 0.01. r, R², NRMSE: mean over subjects, seeds averaged first.",
             "The 0.5 s rows are the data-baselines runs. Differences use the seeds both sides have.", "",
             table.round(3).to_markdown(index=False), "", "## Lifts flagged at the first scored sample (10 test subjects)", "",
             flag_means.round(3).to_markdown(index=False), ""]

    grids = load_all(ROOT / "eegnet_lr_grid" / "history_300")
    fixed = [m for m in load_all(ROOT / "eegnet" / "history_300") if grids and m["seed"] == grids[0]["seed"]]
    if grids and fixed:
        check = lr_check(grids[0], fixed[0])
        (OUT_DIR / "lr_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
        lines += ["## Fixed lr check (EEGNet, 3 s, seed 1)", "", f"```\n{json.dumps(check, indent=2)}\n```", ""]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
