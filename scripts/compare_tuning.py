"""Compare tuned deep baselines with their untuned runs, and list the configuration chosen per fold.

Reads every non-smoke run under output/diagnostics/tuning/<model>/ and the seed runs under
output/data-baselines/baselines/<model>/. Per-subject values are averaged over seeds first.
Writes comparison.csv, selected.csv, per_subject.csv, and summary.md to output/diagnostics/tuning/summary/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_tuning.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TUNED = Path("output/diagnostics/tuning")
UNTUNED = Path("output/data-baselines/baselines")
OUT_DIR = TUNED / "summary"
MODELS = ("eegnet", "shallowconvnet", "cnn_lstm")
METRICS = ("r", "r2", "nrmse")
MAX_EPOCHS = 30  # configs/diagnostics/tuning/*.yaml


def load_runs(paths: list[Path]) -> list[dict]:
    runs = [json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in paths if (p / "metrics.json").exists()]
    return [m for m in runs if not m["smoke"]]


def per_subject(runs: list[dict], metric: str) -> pd.Series:
    return pd.DataFrame([{s: v[metric]["mean"] for s, v in m["per_subject"].items()} for m in runs]).mean(axis=0)


def compare(model: str, tuned: list[dict], untuned: list[dict]) -> dict:
    row = {"model": model, "n_seeds_tuned": len(tuned), "n_seeds_untuned": len(untuned)}
    for metric in METRICS:
        t = per_subject(tuned, metric)
        u = per_subject(untuned, metric).reindex(t.index)  # same subjects on both sides
        diff = t - u
        row |= {f"{metric}_untuned": u.mean(), f"{metric}_tuned": t.mean(), f"{metric}_delta": diff.mean(),
                f"{metric}_delta_sd": diff.std(ddof=1), f"{metric}_n_up": int((diff > 0).sum())}
    row["n_subjects"] = len(t)
    return row


def selections(model: str, tuned: list[dict]) -> list[dict]:
    """One row per fold and grid candidate, from the seed that searched the grid."""
    rows = []
    for m in tuned:
        for fold in m["per_fold"]:
            info = fold["baseline_info"]
            if info.get("selection_run"):
                continue
            for cand in info["grid_results"]:
                rows.append({"model": model, "fold": fold["fold"], "seed": m["seed"], **cand["train"],
                             "selected": cand["train"] == info["selected_train"], "val_mse": cand["best_val_mse"],
                             "best_epoch": cand["best_epoch"], "epochs_run": cand["epochs_run"]})
    return rows


def best_epoch_counts(runs: list[dict]) -> str:
    epochs = np.array([f["baseline_info"]["best_epoch"] for m in runs for f in m["per_fold"]])
    return f"{int((epochs == 1).sum())}/{len(epochs)} at epoch 1, median {np.median(epochs):g}"


def subject_table(model: str, tuned: list[dict], untuned: list[dict]) -> list[dict]:
    """Per test subject: lr selected by the grid seed and r averaged over seeds."""
    chosen = {f["test_subjects"][0]: f["baseline_info"]["selected_train"]["lr"]
              for m in tuned for f in m["per_fold"] if not f["baseline_info"].get("selection_run")}
    t, u = per_subject(tuned, "r"), per_subject(untuned, "r")
    return [{"model": model, "subject": s, "selected_lr": f"lr {chosen[s]:g}", "r_untuned": u[s], "r_tuned": t[s],
             "r_delta": t[s] - u[s]} for s in t.index]


def validation_noise(model: str, tuned: list[dict], untuned: list[dict]) -> dict:
    """Compare the lr-selection gain with the val MSE change from initial weights alone.

    ``init_noise``: mean |difference| in val MSE between the untuned run and the grid candidate of the same
    seed with the same settings (lr 1e-3). They differ only in the random state at the start of the fold.
    ``selection_gain``: mean val MSE of that candidate minus val MSE of the selected candidate.
    """
    grid = next(m for m in tuned if not m["per_fold"][0]["baseline_info"].get("selection_run"))
    base = next(m for m in untuned if m["seed"] == grid["seed"])
    noise, gain = [], []
    for g, b in zip(grid["per_fold"], base["per_fold"]):
        results = g["baseline_info"]["grid_results"]
        same = next(c for c in results if c["train"]["lr"] == base["per_fold"][0]["baseline_info"]["train_config"]["lr"])
        noise.append(abs(same["best_val_mse"] - b["baseline_info"]["best_val_mse"]))
        gain.append(same["best_val_mse"] - min(c["best_val_mse"] for c in results))
    return {"model": model, "init_noise": float(np.mean(noise)), "selection_gain": float(np.mean(gain))}


def main() -> None:
    comparison, selected, epochs, subjects, noise = [], [], [], [], []
    for model in MODELS:
        tuned = load_runs(sorted((TUNED / model).glob("2*")))
        if not tuned:
            print(f"{model}: no finished tuned runs")
            continue
        untuned = load_runs(sorted((UNTUNED / model).glob("2*_seed*")))
        comparison.append(compare(model, tuned, untuned))
        selected.extend(selections(model, tuned))
        epochs.append({"model": model, "untuned": best_epoch_counts(untuned), "tuned": best_epoch_counts(tuned)})
        subjects.extend(subject_table(model, tuned, untuned))
        noise.append(validation_noise(model, tuned, untuned))
    if not comparison:
        return
    comp, sel = pd.DataFrame(comparison), pd.DataFrame(selected)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(OUT_DIR / "comparison.csv", index=False)
    sel.to_csv(OUT_DIR / "selected.csv", index=False)
    subj = pd.DataFrame(subjects)
    subj.to_csv(OUT_DIR / "per_subject.csv", index=False)
    chosen = sel[sel.selected].groupby(["model", "lr"]).size().rename("n_folds").reset_index()
    val = sel.groupby(["model", "lr"]).val_mse.mean().rename("mean_val_mse").reset_index()
    capped = sel.assign(capped=sel.epochs_run == MAX_EPOCHS).groupby(["model", "lr"]).capped.sum().reset_index()
    lr_table = chosen.merge(val, how="right").merge(capped).fillna({"n_folds": 0}).astype({"n_folds": int})
    lr_table["lr"] = lr_table["lr"].map("lr {:g}".format)  # text label: tabulate would reformat a numeric string
    tables = [
        ("Tuned vs untuned (delta = tuned - untuned; n_up = subjects with a higher value, and lower is better for "
         "NRMSE)", comp.drop(columns=["n_seeds_untuned"])),
        (f"Folds selecting each lr (capped = candidates that ran all {MAX_EPOCHS} epochs)", lr_table),
        ("Selection gain vs initial-weight noise in val MSE", pd.DataFrame(noise)),
        ("Per subject", subj),
        ("Best epoch of the final models", pd.DataFrame(epochs)),
    ]
    md = "# Deep baseline tuning\n\n" + "\n\n".join(f"## {title}\n\n{t.to_markdown(index=False, floatfmt='.3f')}"
                                                    for title, t in tables) + "\n"
    (OUT_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
