"""Compare the within-subject diagnostic split with LOSO, per model and per subject.

Within-subject: train S1-S5, validate S6, test S7-S9 (D6). Diagnostic only: S7-S9 are always weight
series, while S1-S6 mix conditions. Deep models use their best history from the ablation, on both sides.
Per-subject values are averaged over seeds first. Residual metrics come from the newest
output/diagnostics/residual/<run-id>/residual.csv.

The LOSO runs score all nine series, so they are also rescored from predictions.npz on S7-S9 only (the
*_loso_s789 columns), which removes the test-series difference. Each split subtracts its own template in
the residual metrics, so those columns are not measured against the same reference.
Writes comparison.csv, per_subject.csv, and summary.md to output/diagnostics/within_subject/summary/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_within_subject.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eegtraj.metrics import aggregate_subject, per_trial_metrics

WITHIN = Path("output/diagnostics/within_subject")
OUT_DIR = WITHIN / "summary"
RESIDUAL_ROOT = Path("output/diagnostics/residual")
METRICS = ("r", "r2", "nrmse")
RESIDUALS = ("residual_r", "residual_r_specific")
TEST_SERIES = (7, 8, 9)
CACHE_DIR = Path("dataset/cache/98f5fa6e66")
# label -> (LOSO runs, within-subject runs)
MODELS = {
    "template (oracle timing)": ("output/data-baselines/baselines/template", "template"),
    "template_cue": ("output/diagnostics/baselines/template_cue", "template_cue"),
    "ridge_lag 0.5 s": ("output/data-baselines/baselines/ridge_lag", "ridge_lag"),
    "eegnet 3 s": ("output/diagnostics/history/eegnet/history_300", "eegnet/history_300"),
    "cnn_lstm 2 s": ("output/diagnostics/history/cnn_lstm/history_200", "cnn_lstm/history_200"),
}


def load_runs(parent: Path) -> list[tuple[str, dict]]:
    runs = []
    for f in sorted(parent.glob("2*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if not m["smoke"]:
            runs.append((f.parent.as_posix(), m))
    return runs


def subject_values(runs: list[tuple[str, dict]], residuals: pd.DataFrame | None) -> pd.DataFrame:
    """One row per subject with each metric averaged over seeds."""
    out = {k: pd.DataFrame([{s: v[k]["mean"] for s, v in m["per_subject"].items()} for _, m in runs]).mean(axis=0)
           for k in METRICS}
    if residuals is not None:
        rows = residuals[residuals.run.isin([name for name, _ in runs])]
        for k in RESIDUALS:
            out[k] = rows.pivot(index="run", columns="subject", values=k).mean(axis=0)
    return pd.DataFrame(out)


def scores_on_series(y_true: np.ndarray, y_pred: np.ndarray, trial_row: np.ndarray, series_of_row: np.ndarray,
                     series: tuple[int, ...]) -> dict[str, float]:
    """D7 subject-level metrics using only the trials of the given series."""
    keep = np.isin(series_of_row[trial_row], series)
    res = aggregate_subject(per_trial_metrics(y_true[keep], y_pred[keep], trial_row[keep]))
    return {k: res[k]["mean"] for k in METRICS}


def loso_on_test_series(runs: list[tuple[str, dict]]) -> pd.DataFrame:
    """LOSO scores on S7-S9 per subject, averaged over seeds."""
    frames = []
    for name, _ in runs:
        with np.load(Path(name) / "predictions.npz") as z:
            subjects = sorted({k.split("__")[0] for k in z.keys()}, key=lambda s: int(s[1:]))
            frames.append(pd.DataFrame({
                s: scores_on_series(z[f"{s}__y_true"], z[f"{s}__y_pred"], z[f"{s}__trial_row"],
                                    pd.read_csv(CACHE_DIR / s / "trials.csv").series.to_numpy(), TEST_SERIES)
                for s in subjects}).T)
    return sum(frames) / len(frames)


def main() -> None:
    tables = sorted(RESIDUAL_ROOT.glob("2*/residual.csv"))
    residuals = pd.read_csv(tables[-1]) if tables else None
    rows, subjects = [], []
    for label, (loso_path, within_name) in MODELS.items():
        loso_runs, within_runs = load_runs(Path(loso_path)), load_runs(WITHIN / within_name)
        if not loso_runs or not within_runs:
            print(f"missing runs for {label}")
            continue
        loso, within = subject_values(loso_runs, residuals), subject_values(within_runs, residuals)
        row = {"model": label, "n_loso": len(loso_runs), "n_within": len(within_runs)}
        for k in loso.columns.intersection(within.columns):
            diff = within[k] - loso[k].reindex(within.index)
            row |= {f"{k}_loso": loso[k].mean(), f"{k}_within": within[k].mean(), f"{k}_delta": diff.mean(),
                    f"{k}_delta_sd": diff.std(ddof=1), f"{k}_n_up": int((diff > 0).sum())}
        same = loso_on_test_series(loso_runs)
        for k in METRICS:
            diff = within[k] - same[k].reindex(within.index)
            row |= {f"{k}_loso_s789": same[k].mean(), f"{k}_delta_s789": diff.mean(), f"{k}_delta_s789_sd": diff.std(ddof=1),
                    f"{k}_n_up_s789": int((diff > 0).sum())}
        rows.append(row)
        subjects += [{"model": label, "subject": s, "r_loso": loso.r[s], "r_loso_s789": same.r[s], "r_within": within.r[s],
                      "r_delta": within.r[s] - loso.r[s], "r_delta_s789": within.r[s] - same.r[s],
                      "r2_loso": loso.r2[s], "r2_loso_s789": same.r2[s], "r2_within": within.r2[s]} for s in within.index]
    table, per_subject = pd.DataFrame(rows), pd.DataFrame(subjects)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)
    per_subject.to_csv(OUT_DIR / "per_subject.csv", index=False)
    wide = per_subject.pivot(index="subject", columns="model", values="r_delta_s789")
    wide = wide.reindex(sorted(wide.index, key=lambda s: int(s[1:])))
    text = "\n".join([
        "# Within-subject vs LOSO", "",
        "Diagnostic only. Train S1-S5, validate S6, test S7-S9; S7-S9 are always weight series, S1-S6 mix conditions.",
        "delta = within-subject minus LOSO, mean over subjects. n_up = subjects with a higher within-subject value.",
        "*_s789 columns rescore LOSO on S7-S9 only. Residual metrics subtract a different template in each split.", "",
        table.round(3).to_markdown(index=False), "", "## r_delta per subject (within-subject minus LOSO on S7-S9)", "", wide.round(3).to_markdown(), ""])
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
