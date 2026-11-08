"""Oracle-warp ceiling: fit parametric warps of the cue template to each LOSO test trial. ANALYSIS ONLY, CPU.

Every parameter is fitted to the test trial's own trajectory, so the result is an upper bound, never a decoder.
Writes to a new <output.root>/<run-id>/ (output/smoke/oracle-warp/ with --smoke): summary.csv, summary.md,
per_subject.csv, params.csv, params_summary.csv, bound_hits.csv, template_landmarks.csv, metrics.json,
oracle_predictions.npz, figures/.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\run_oracle_warp.py --config configs\\diagnostics\\oracle_warp.yaml
    .venv\\Scripts\\python scripts\\run_oracle_warp.py --config configs\\diagnostics\\oracle_warp.yaml --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from eegtraj.diagnostics.oracle_warp_report import (
    per_trial_r,
    pick_overlay_trials,
    plot_overlays,
    plot_parameters,
    plot_summary,
    run,
)
from eegtraj.utils.run import setup_run, teardown_logging

SMOKE_ROOT = Path("output/smoke/oracle-warp")
SMOKE_SUBJECTS = ["P1"]
SMOKE_MAX_TRIALS = 6
FS_HZ = 100.0
COLUMNS = ["method", "kind", "n_params", "r_mean", "r_sd", "r2_mean", "r2_sd", "nrmse_mean", "nrmse_sd",
           "residual_r_specific_mean", "residual_r_specific_sd", "d_r_vs_cue", "n_better_r", "d_r2_vs_cue", "n_better_r2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--max-trials", type=int, help="first N test trials per subject (debugging)")
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_SUBJECTS}, {SMOKE_MAX_TRIALS} trials")
    parser.add_argument("--tag")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["subjects"] = args.subjects or (SMOKE_SUBJECTS if args.smoke else None)
    cfg["max_trials"] = args.max_trials or (SMOKE_MAX_TRIALS if args.smoke else None)
    cfg["smoke"] = bool(args.smoke)
    root = SMOKE_ROOT if args.smoke else Path(cfg["output"]["root"])
    run_dir = setup_run(root, cfg, tag=args.tag)
    print(f"Run directory: {run_dir}")

    out = run(cfg, run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir()
    trials = pick_overlay_trials(per_trial_r(out["methods"], "template_cue"), tuple(cfg["overlay_quantiles"]))
    plot_overlays(out["methods"], out["templates"], trials, FS_HZ, fig_dir)
    plot_parameters(out["params"], fig_dir)
    plot_summary(out["per_subject"], fig_dir)

    table = out["summary"][COLUMNS]
    hits = out["bound_hits"]
    psum = out["params_summary"]
    rho = psum.groupby("variant")[["rho_tau_rt", "rho_scale_duration"]].median().dropna(how="all")
    (run_dir / "summary.md").write_text(
        "# Oracle-warp ceiling\n\nANALYSIS ONLY: warp parameters are fitted to each test trial's own trajectory. "
        "Base shape: the fold's cue-locked template. Per-axis A and c do not change per-trial r. Mean over test "
        "subjects; d_* and n_better_* are paired against template_cue. EEG decoders: mean over seeds per subject.\n\n"
        + table.round(3).to_markdown(index=False)
        + "\n\n## Median over subjects of the per-subject Spearman rho\n\n" + rho.round(3).to_markdown()
        + "\n\n## Fits at a bound\n\n" + (hits.round(4).to_markdown(index=False) if len(hits) else "none") + "\n",
        encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(3).to_string(index=False))
    print(rho.round(3).to_string())
    print(hits.round(4).to_string(index=False) if len(hits) else "no bound hits")
    print(f"Run directory: {run_dir}")
    teardown_logging()


if __name__ == "__main__":
    main()
