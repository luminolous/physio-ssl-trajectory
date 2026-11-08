"""Compare CNN-LSTM 2 s with and without Euclidean alignment: inner validation (per fold) and LOSO test (per subject).

Reads metrics.json of output/diagnostics/pre-design/euclidean-alignment/cnn_lstm_ea/ and
output/diagnostics/pre-design/screening/broadband__all/ (the same config without alignment), and the residual tables
in output/diagnostics/summary/ (run scripts/compute_residual_metrics.py and scripts/summarize_runs.py --stage diagnostics
first). Seeds are averaged per fold or subject first. Writes to a new
output/diagnostics/pre-design/euclidean-alignment/summary/<run-id>/: comparison.csv, per_subject.csv, summary.md.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_alignment.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

PRE = Path("output/diagnostics/pre-design")
SUMMARY = Path("output/diagnostics/summary")
RUNS = {"aligned": ("euclidean-alignment", "cnn_lstm_ea"), "unaligned": ("screening", "broadband__all")}
FLOOR = "template_cue"
METRICS = ("r", "r2", "nrmse", "residual_r_specific")


def load(group: str, name: str) -> list[dict]:
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((PRE / group / name).glob("2*/metrics.json"))]
    return [m for m in runs if not m["smoke"]]


def inner_val(runs: list[dict], metric: str) -> pd.Series:
    return pd.DataFrame([{f["fold"]: np.mean([v[metric] for v in f["inner_val"].values()]) for f in m["per_fold"]}
                         for m in runs]).mean(axis=0)


def test(runs: list[dict], group: str, name: str, metric: str, specific: pd.DataFrame) -> pd.Series:
    if metric == "residual_r_specific":
        return specific[f"{group}/{name}"]
    return pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for m in runs]).mean(axis=0)


def paired(a: pd.Series, b: pd.Series) -> dict[str, float]:
    d = (a - b).dropna()
    return {"aligned": float(a[d.index].mean()), "other": float(b[d.index].mean()), "diff": float(d.mean()),
            "diff_sd": float(d.std(ddof=1)), "n_up": int((d > 0).sum()), "n": int(d.size),
            "wilcoxon_p": float(scipy.stats.wilcoxon(d, method="exact").pvalue)}


def main() -> None:
    specific = pd.read_csv(SUMMARY / "per_subject_residual_r_specific.csv", index_col=0)
    runs = {k: load(*v) for k, v in RUNS.items()}
    summary_runs = pd.read_csv(SUMMARY / "runs.csv")
    floor_runs = [json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))
                  for d in summary_runs[(summary_runs.group == "baselines") & (summary_runs.config == FLOOR)].run]
    rows, subjects = [], {}
    for metric in METRICS:
        a, b = inner_val(runs["aligned"], metric), inner_val(runs["unaligned"], metric)
        rows.append({"metric": metric, "split": "inner_val", "against": "unaligned", **paired(a, b)})
        ta = test(runs["aligned"], *RUNS["aligned"], metric, specific)
        tb = test(runs["unaligned"], *RUNS["unaligned"], metric, specific)
        subjects[f"aligned_{metric}"], subjects[f"unaligned_{metric}"] = ta, tb
        rows.append({"metric": metric, "split": "test", "against": "unaligned", **paired(ta, tb)})
        if metric in ("r", "r2", "nrmse"):
            fl = pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for m in floor_runs]).mean(axis=0)
            subjects[f"{FLOOR}_{metric}"] = fl
            rows.append({"metric": metric, "split": "test", "against": FLOOR, **paired(ta, fl)})
    comparison = pd.DataFrame(rows)
    epochs = {k: float(np.median([f["baseline_info"]["best_epoch"] for m in v for f in m["per_fold"]])) for k, v in runs.items()}
    out = PRE / "euclidean-alignment" / "summary" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    comparison.to_csv(out / "comparison.csv", index=False)
    table = pd.DataFrame(subjects).sort_index(key=lambda i: i.str[1:].astype(int))
    table.to_csv(out / "per_subject.csv")
    head = (f"# Euclidean alignment\n\nCNN-LSTM, broadband, all channels, 2 s. Seeds: aligned {len(runs['aligned'])}, unaligned "
            f"{len(runs['unaligned'])}, averaged per fold or subject first. Inner-validation folds share subjects, so p is "
            f"descriptive. Median best epoch: {epochs}.\n\n")
    (out / "summary.md").write_text(head + comparison.round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200)
    print(comparison.round(3).to_string(index=False))
    print(epochs)
    print(table.round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
