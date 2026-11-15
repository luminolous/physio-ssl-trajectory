"""Summary of the decoder loss-weight sweep ("[DECISION] Loss weights and prior values"). No training.

Reads every finished run under output/proposed-model/decoder-sweep/<folder>/ (metrics.json), one row per run: loss
weights, test scores on the screening subjects P3 and P6 (screening folds), their mean, the template_cue floor, and
the selected lr / epochs. Rows are sorted by the selection score: mean r over P3 and P6 (decided 2026-11-15;
residual_r_specific and NRMSE reported). Smoke runs are skipped unless ``--include-smoke``; runs from before the
phase prior (SSL-20; no ``phase_prior`` in the fold records) unless ``--include-old``.

Writes to a new output/proposed-model/decoder-sweep/summary/<run-id>/: summary.csv, summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\summarize_decoder_sweep.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eegtraj.utils.run import setup_run

SWEEP = Path("output/proposed-model/decoder-sweep")
WEIGHTS = ("aux", "delta", "amp", "gate")


def collect(include_smoke: bool, include_old: bool = False) -> pd.DataFrame:
    rows = []
    for mfile in sorted(SWEEP.glob("*/*/metrics.json")):
        run = mfile.parent
        if run.parent.name == "summary" or ("smoke" in run.name and not include_smoke):
            continue
        m = json.loads(mfile.read_text(encoding="utf-8"))
        if "phase_prior" not in m["per_fold"][0] and not include_old:  # before SSL-20
            continue
        row = {"folder": run.parent.name, "run": run.name, "folds": m.get("folds"),
               **{w: m["loss_weights"].get(w) for w in WEIGHTS}}
        for s, sc in m["per_subject"].items():
            row[f"r_{s}"] = sc["r"]
            row[f"rrs_{s}"] = sc["residual_r_specific"]
            row[f"nrmse_{s}"] = sc["nrmse"]
            row[f"floor_r_{s}"] = m["floor_template_cue"][s]["r"]
        a = m["aggregate"]
        row.update(r=a["r"]["mean"], rrs=a["residual_r_specific"]["mean"], nrmse=a["nrmse"]["mean"], r2=a["r2"]["mean"],
                   floor_r=m["aggregate_floor_template_cue"]["r"]["mean"])
        row["head_floor_r"] = m.get("aggregate_floor_cue_phi", {}).get("r", {}).get("mean", float("nan"))  # cue phi, u = 0
        row["head"] = m["model"]["head"]
        row["lr"] = " ".join(f"{f['selected_lr']:g}" for f in m["per_fold"])
        row["best_epochs"] = " ".join(str(next(g["best_epoch"] for g in f["grid"] if g["lr"] == f["selected_lr"]))
                                      for f in m["per_fold"])
        row["minutes"] = round(m["wall_seconds"] / 60, 1)
        rows.append(row)
    if not rows:
        raise SystemExit(f"no finished sweep runs under {SWEEP}")
    return pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--include-old", action="store_true", help="also runs from before the phase prior (SSL-20)")
    args = parser.parse_args()
    table = collect(args.include_smoke, args.include_old)
    run_dir = setup_run(SWEEP / "summary", {"include_smoke": args.include_smoke,
                                            "include_old": args.include_old, "runs": table.run.tolist()})
    table.to_csv(run_dir / "summary.csv", index=False)
    cols = [c for c in ["head", "aux", "delta", "amp", "gate", "r", "r_P3", "r_P6", "rrs", "rrs_P3", "rrs_P6", "nrmse", "r2",
                        "floor_r", "head_floor_r", "lr", "best_epochs", "minutes", "run"] if c in table]
    (run_dir / "summary.md").write_text(
        "# Decoder loss-weight sweep (screening folds P3, P6)\n\nSelection score: mean r over P3 and P6 (per trial, per "
        "axis). Tie rule (decided 2026-11-15): differences < 0.005 keep the base / default value. `floor_r`: "
        "template_cue on the same samples. `head_floor_r`: cue phi (u = 0: landmark templates at "
        "clip((t - t_LED - tau0) / s0, 0, 1), uniform mixing, A = 1, Delta = 0); NaN for runs before SSL-19.\n\n" + table[cols].round(4).to_markdown(index=False) + "\n",
        encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"rows": table.to_dict(orient="records")}, indent=2) + "\n",
                                          encoding="utf-8")
    print(table[cols].round(4).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
