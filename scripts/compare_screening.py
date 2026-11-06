"""Compare the band and channel screening runs: inner validation (selection readout) and LOSO test (descriptive).

Reads metrics.json of every run in output/diagnostics/pre-design/screening/ and output/diagnostics/summary/ (run
scripts/compute_residual_metrics.py and scripts/summarize_runs.py --stage diagnostics first). Runs of one config are
seeds and are averaged per subject or fold first. Writes to a new output/diagnostics/pre-design/screening/summary/<run-id>/:
    configs.csv    one row per config
    contrasts.csv  paired contrasts on inner validation (per fold) and on test (per subject)
    per_subject.csv, per_fold_inner_val.csv
    summary.md

Inner-validation folds share validation subjects, so their paired tests are not independent; treat p as descriptive.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_screening.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

ROOT = Path("output/diagnostics/pre-design/screening")
SUMMARY = Path("output/diagnostics/summary")
METRICS = ("r", "r2", "nrmse", "residual_r", "residual_r_specific")
FLOOR = "template_cue"
REFERENCE = "broadband__all"
CONTRASTS = (
    ("contra vs ipsi, broadband", "broadband__contralateral", "broadband__ipsilateral"),
    ("contra vs ipsi, MRCP", "mrcp__contralateral", "mrcp__ipsilateral"),
    ("contra vs ipsi, envelope", "env__contralateral", "env__ipsilateral"),
    ("MRCP vs broadband, all", "mrcp__all", REFERENCE),
    ("envelope vs broadband, all", "env__all", REFERENCE),
    ("MRCP + envelope vs broadband, all", "mrcp_env__all", REFERENCE),
    ("MRCP + envelope vs MRCP, all", "mrcp_env__all", "mrcp__all"),
    ("MRCP vs envelope, all", "mrcp__all", "env__all"),
    ("no prefrontal vs all, broadband", "broadband__no_prefrontal", REFERENCE),
    ("sensorimotor vs all, broadband", "broadband__sensorimotor", REFERENCE),
    ("sensorimotor vs prefrontal, broadband", "broadband__sensorimotor", "broadband__prefrontal"),
    ("prefrontal vs contra, broadband", "broadband__prefrontal", "broadband__contralateral"),
    ("prefrontal vs contra, MRCP", "mrcp__prefrontal", "mrcp__contralateral"),
)


def load_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for f in sorted(ROOT.glob("*/2*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if not m["smoke"]:
            runs.setdefault(f.parent.parent.name, []).append(m)
    return runs


def inner_val_per_fold(runs: list[dict], metric: str) -> pd.Series:
    """One value per fold: mean over the fold's validation subjects, then over seeds."""
    frame = pd.DataFrame([{f["fold"]: np.mean([v[metric] for v in f["inner_val"].values()]) for f in m["per_fold"]} for m in runs])
    return frame.mean(axis=0)


def paired(a: pd.Series, b: pd.Series) -> dict[str, float]:
    d = (a - b).dropna()
    return {"a": float(a[d.index].mean()), "b": float(b[d.index].mean()), "diff": float(d.mean()), "diff_sd": float(d.std(ddof=1)),
            "n_up": int((d > 0).sum()), "n": int(d.size), "wilcoxon_p": float(scipy.stats.wilcoxon(d, method="exact").pvalue)}


def main() -> None:
    runs = load_runs()
    summary_runs = pd.read_csv(SUMMARY / "runs.csv")
    tables = {"residual_r": pd.read_csv(SUMMARY / "per_subject_residual_r.csv", index_col=0),
              "residual_r_specific": pd.read_csv(SUMMARY / "per_subject_residual_r_specific.csv", index_col=0)}

    def test_values(config: str, metric: str) -> pd.Series:
        if metric in tables:
            return tables[metric][f"screening/{config}"]
        return pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for m in runs[config]]).mean(axis=0)

    floor_dirs = summary_runs[(summary_runs.group == "baselines") & (summary_runs.config == FLOOR)].run
    floor = {k: pd.DataFrame([{s: v[k]["mean"] for s, v in json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))["per_subject"].items()}
                              for d in floor_dirs]).mean(axis=0) for k in ("r", "r2")}
    rows, subjects, folds = [], {}, {}
    for config, ms in runs.items():
        row = {"config": config, "n_seeds": len(ms), "n_parameters": ms[0]["per_fold"][0]["baseline_info"].get("n_parameters"),
               "median_best_epoch": float(np.median([f["baseline_info"]["best_epoch"] for m in ms for f in m["per_fold"]])),
               "wall_min": round(float(np.mean([m["wall_seconds"] for m in ms])) / 60, 1)}
        for metric in METRICS:
            fv, tv = inner_val_per_fold(ms, metric), test_values(config, metric)
            folds[f"{config}/{metric}"], subjects[f"{config}/{metric}"] = fv, tv
            row[f"val_{metric}"], row[f"test_{metric}"] = float(fv.mean()), float(tv.mean())
        for k in ("r", "r2"):
            d = subjects[f"{config}/{k}"] - floor[k].reindex(subjects[f"{config}/{k}"].index)
            row[f"test_{k}_vs_floor"], row[f"n_above_floor_{k}"] = float(d.mean()), int((d > 0).sum())
        rows.append(row)
    configs = pd.DataFrame(rows).sort_values("val_residual_r_specific", ascending=False)

    contrasts = []
    for label, a, b in CONTRASTS:
        if a not in runs or b not in runs:
            continue
        for metric in METRICS:
            contrasts.append({"contrast": label, "metric": metric, "split": "inner_val", **paired(folds[f"{a}/{metric}"], folds[f"{b}/{metric}"])})
            contrasts.append({"contrast": label, "metric": metric, "split": "test", **paired(subjects[f"{a}/{metric}"], subjects[f"{b}/{metric}"])})
    contrasts = pd.DataFrame(contrasts)

    out = ROOT / "summary" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    configs.to_csv(out / "configs.csv", index=False)
    contrasts.to_csv(out / "contrasts.csv", index=False)
    pd.DataFrame(subjects).sort_index(key=lambda i: i.str[1:].astype(int)).to_csv(out / "per_subject.csv")
    pd.DataFrame(folds).to_csv(out / "per_fold_inner_val.csv")
    view = contrasts[contrasts.metric.isin(("r", "r2", "residual_r_specific"))].round(3)
    (out / "summary.md").write_text("# Band and channel screening\n\nCNN-LSTM, 2 s, fixed budget. Inner validation is the selection readout "
                                    "(folds share validation subjects, so p is descriptive). LOSO test is descriptive only.\n\n"
                                    + configs.round(3).to_markdown(index=False) + "\n\n" + view.to_markdown(index=False) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(configs.round(3).drop(columns=["val_nrmse", "test_nrmse", "val_residual_r", "test_residual_r"]).to_string(index=False))
    print(view.pivot_table(index=["contrast", "split"], columns="metric", values=["diff", "n_up", "wilcoxon_p"], sort=False).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
