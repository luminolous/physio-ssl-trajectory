"""Summary of the pretraining budget sweep ("[DECISION] Pretraining budget and lambda"). No training.

Reads every finished run under output/proposed-model/pretrain-sweep/<variant>/ (those with metrics.json), takes the
final health of the online encoder, and ranks the (lam, steps) cells by the selection score: within-subject probe
``residual_r_specific`` on the screening subjects (decided 2026-11-15). Cross-subject probe scores, embedding health and
cost are reported next to it. Smoke runs are skipped unless ``--include-smoke``.

Writes to a new output/proposed-model/pretrain-sweep/summary/<run-id>/: summary.csv, summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\summarize_pretrain_sweep.py --variant lejepa_guided
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from eegtraj.utils.run import setup_run

SWEEP = Path("output/proposed-model/pretrain-sweep")
COLUMNS = ["lam", "steps", "within_rrs", "within_rrs_P3", "within_rrs_P6", "within_rrs_step0", "within_r", "cross_rrs",
           "cross_r",
           "emb_sd_min", "eff_rank", "final_pred", "final_sigreg", "train_minutes", "run"]


def collect(variant: str, include_smoke: bool) -> pd.DataFrame:
    rows = []
    for run in sorted((SWEEP / variant).glob("*")):
        if not (run / "metrics.json").exists() or ("smoke" in run.name and not include_smoke):
            continue
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        h = m["final_health"]["online"]
        health = pd.read_csv(run / "health.csv")
        step0 = health[(health.step == 0) & (health.encoder == "online")]
        rows.append({"lam": cfg["objective"]["lam"], "steps": m["steps"],
                     **{k: h.get(k) for k in COLUMNS if k in h},
                     "within_rrs_step0": float(step0.within_rrs.iloc[0]),  # random-init encoder, same seed
                     "final_pred": m["final_loss"].get("pred"), "final_sigreg": m["final_loss"].get("sigreg"),
                     "train_minutes": round(m["train_seconds"] / 60, 1), "run": run.name})
    if not rows:
        raise SystemExit(f"no finished sweep runs under {SWEEP / variant}")
    return pd.DataFrame(rows)[COLUMNS].sort_values("within_rrs", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="lejepa_guided")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    table = collect(args.variant, args.include_smoke)
    run_dir = setup_run(SWEEP / "summary", {"variant": args.variant, "include_smoke": args.include_smoke,
                                            "runs": table.run.tolist()})
    table.to_csv(run_dir / "summary.csv", index=False)
    grid = table.pivot_table(index="lam", columns="steps", values="within_rrs")
    (run_dir / "summary.md").write_text(
        f"# Pretraining budget sweep ({args.variant})\n\nSelection score: `within_rrs`, the within-subject "
        "(leave-one-series-out) linear-probe `residual_r_specific` on P3 and P6, final online encoder. Rows sorted by "
        "it. `within_rrs_step0`: the same probe on the random-init encoder (reference for the probe noise level).\n\n" + table.round(4).to_markdown(index=False) + "\n\n## within_rrs by lam x steps\n\n"
        + grid.round(4).to_markdown() + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"rows": table.to_dict(orient="records")}, indent=2) + "\n",
                                          encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
