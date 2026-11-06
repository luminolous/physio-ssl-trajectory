"""Compare the residual formulation with direct prediction and the cue-locked template, per subject.

Reads output/diagnostics/summary/ (run scripts/summarize_runs.py --stage diagnostics first) and metrics.json of the
residual runs. Writes comparison.csv, per_subject.csv, and summary.md to output/diagnostics/pre-design/residual/summary/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_residual.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

SUMMARY = Path("output/diagnostics/summary")
ROOT = Path("output/diagnostics/pre-design/residual")
OUT = ROOT / "summary"
MODELS = ("cnn_lstm", "eegnet")
DIRECT = "{model}/history_200"
FLOOR = "template_cue"


def per_subject(group: str, config: str, metric: str, runs: pd.DataFrame) -> pd.Series:
    """Seed-averaged per-subject values of a configuration from its metrics.json files."""
    dirs = runs[(runs.group == group) & (runs.config == config)].run
    if dirs.empty:
        raise KeyError(f"no runs for {group} {config}")
    values = [{s: v[metric]["mean"] for s, v in json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))["per_subject"].items()}
              for d in dirs]
    return pd.DataFrame(values).mean()


def paired(a: pd.Series, b: pd.Series) -> dict[str, float]:
    d = (a - b).dropna()
    return {"a": float(a[d.index].mean()), "b": float(b[d.index].mean()), "diff": float(d.mean()), "diff_sd": float(d.std(ddof=1)),
            "n_up": int((d > 0).sum()), "n": int(d.size), "wilcoxon_p": float(scipy.stats.wilcoxon(d, method="exact").pvalue)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(SUMMARY / "runs.csv")
    specific = pd.read_csv(SUMMARY / "per_subject_residual_r_specific.csv", index_col=0)
    residual_r = pd.read_csv(SUMMARY / "per_subject_residual_r.csv", index_col=0)
    rows, subjects = [], {}
    for model in MODELS:
        direct = DIRECT.format(model=model)
        for metric in ("r", "r2", "nrmse"):
            res = per_subject("residual", model, metric, runs)
            subjects[f"{model}_residual_{metric}"] = res
            subjects[f"{model}_direct_{metric}"] = per_subject("history", direct, metric, runs)
            subjects[f"{FLOOR}_{metric}"] = per_subject("baselines", FLOOR, metric, runs)
            rows.append({"model": model, "metric": metric, "against": "direct 2 s", **paired(res, subjects[f"{model}_direct_{metric}"])})
            rows.append({"model": model, "metric": metric, "against": FLOOR, **paired(res, subjects[f"{FLOOR}_{metric}"])})
        for metric, table in (("residual_r", residual_r), ("residual_r_specific", specific)):
            res, base = table[f"residual/{model}"], table[f"history/{direct}"]
            subjects[f"{model}_residual_{metric}"], subjects[f"{model}_direct_{metric}"] = res, base
            rows.append({"model": model, "metric": metric, "against": "direct 2 s", **paired(res, base)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    table = pd.DataFrame(subjects).sort_index(key=lambda i: i.str[1:].astype(int))
    table.to_csv(OUT / "per_subject.csv")
    epochs = {}
    for model in MODELS:
        dirs = runs[(runs.group == "residual") & (runs.config == model)].run
        folds = [f for d in dirs for f in json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))["per_fold"]]
        epochs[model] = [f["baseline_info"]["best_epoch"] for f in folds]
    (OUT / "summary.md").write_text("# Residual formulation against direct prediction and the cue-locked template\n\n"
                                    "`a` is the residual run, `b` the comparator. LOSO, 10 subjects, 3 seeds, seed-averaged per subject.\n\n"
                                    + comparison.round(3).to_markdown(index=False) + "\n\n"
                                    + f"Median best epoch: { {m: float(np.median(v)) for m, v in epochs.items()} }\n", encoding="utf-8")
    print(comparison.round(3).to_string())
    print({m: (float(np.median(v)), min(v), max(v)) for m, v in epochs.items()})
    print(table[[c for c in table.columns if c.endswith("_r") or c.endswith("specific")]].loc[:, lambda d: ~d.columns.duplicated()].round(3).to_string())


if __name__ == "__main__":
    main()
