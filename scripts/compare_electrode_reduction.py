"""Electrode reduction curve: table, curve, and scalp diagrams from the montage runs. Reads runs only; trains nothing.

Groups come from configs/diagnostics/electrode_reduction/ (runs in output/diagnostics/final-checks/
electrode-reduction/<config>/). Comparators with all 32 channels, same data and form: the ocular-cleaned runs
(final-checks/ocular/) and the uncleaned runs (pre-design/screening/broadband__all, pre-design/residual/cnn_lstm).
Groups without a finished run are skipped. Seeds are averaged per subject (test) and per fold (inner validation).

Writes to a new output/diagnostics/final-checks/electrode-reduction/summary/<run-id>/: table.csv, summary.md,
specific_vs_channels.png, montages.png.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_electrode_reduction.py [--seeds 1] [--runs-root output\\smoke\\electrode-reduction --include-smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eegtraj.diagnostics import electrode_reduction as er
from eegtraj.diagnostics.residual import load_templates
from eegtraj.preprocessing.reference import CHANNEL_SETS, LINKED_MASTOID
from eegtraj.utils.run import setup_run, teardown_logging

CONFIGS = Path("configs/diagnostics/electrode_reduction")
RUNS_ROOT = Path("output/diagnostics/final-checks/electrode-reduction")
OUTPUT = RUNS_ROOT / "summary"
CACHE = "dataset/cache/98f5fa6e66"
SPLITS = "configs/splits/default.yaml"
COMPUTE = RUNS_ROOT / "compute"
RESIDUAL_SEED, N_PERMUTATIONS = 20261106, 5  # as scripts/compute_residual_metrics.py
COMPARATORS = {
    ("ica", "direct"): ("ica__all__car__direct", Path("output/diagnostics/final-checks/ocular/cnn_lstm_direct")),
    ("ica", "residual"): ("ica__all__car__residual", Path("output/diagnostics/final-checks/ocular/cnn_lstm_residual")),
    ("raw", "direct"): ("raw__all__car__direct", Path("output/diagnostics/pre-design/screening/broadband__all")),
    ("raw", "residual"): ("raw__all__car__residual", Path("output/diagnostics/pre-design/residual/cnn_lstm")),
}
SHOWN = ["group", "n_chans", "n_seeds", "inner_residual_r_specific", "inner_d_residual_r_specific",
         "inner_n_up_residual_r_specific", "inner_r", "inner_d_r", "test_residual_r_specific",
         "test_d_residual_r_specific", "test_n_up_residual_r_specific", "test_p_residual_r_specific", "test_r", "test_d_r",
         "test_r2", "test_nrmse", "n_params", "epoch_s_median", "train_peak_gpu_gb", "latency_cpu_median_ms",
         "latency_cuda_median_ms"]


def groups(runs_root: Path) -> list[er.Group]:
    out = [er.Group(name, data, "all", "car", form, 32, folder) for (data, form), (name, folder) in COMPARATORS.items()]
    for f in sorted(CONFIGS.glob("*.yaml")):
        data, montage, ref, form = f.stem.split("__")
        out.append(er.Group(f.stem, data, montage, ref, form, len(CHANNEL_SETS[montage]), runs_root / f.stem))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", help="only these seeds (default: every finished seed)")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--include-smoke", action="store_true", help="testing only: also read smoke runs")
    args = parser.parse_args()
    all_groups = groups(Path(args.runs_root))
    run_dir = setup_run(OUTPUT if not args.include_smoke else Path("output/smoke/electrode-reduction/summary"),
                        {"runs_root": args.runs_root, "seeds": args.seeds, "include_smoke": args.include_smoke,
                         "residual_seed": RESIDUAL_SEED, "n_permutations": N_PERMUTATIONS,
                         "groups": {g.name: g.folder.as_posix() for g in all_groups}})
    print(f"Run directory: {run_dir}")
    templates = load_templates(CACHE, SPLITS, "loso", "position", [])
    seeds = tuple(args.seeds) if args.seeds else None
    scores = {}
    for g in all_groups:
        sc = er.group_scores(g, templates, RESIDUAL_SEED, N_PERMUTATIONS, seeds, args.include_smoke)
        if sc is not None:
            scores[g.name] = sc
            print(f"{g.name}: {sc['n_seeds']} seed(s)")
    table = er.comparison_table(all_groups, scores, {k: v[0] for k, v in COMPARATORS.items()})
    compute = sorted(COMPUTE.glob("2*/compute.csv"))
    if compute:
        c = pd.read_csv(compute[-1])[["n_chans", "latency_cpu_median_ms", "latency_cuda_median_ms", "inference_peak_gpu_mb"]]
        table = table.merge(c.drop_duplicates("n_chans"), on="n_chans", how="left")
    table.to_csv(run_dir / "table.csv", index=False)
    pd.DataFrame([r for v in scores.values() for r in v["per_run"]]).to_csv(run_dir / "per_run.csv", index=False)
    (run_dir / "runs.json").write_text(json.dumps({k: v["runs"] for k, v in scores.items()}, indent=2), encoding="utf-8")
    er.plot_curve(table, run_dir / "specific_vs_channels.png")
    names = json.loads((Path(CACHE) / "P1" / "manifest.json").read_text(encoding="utf-8"))["eeg_names"]
    montages = {"all 32": tuple(names)} | {m: CHANNEL_SETS[m] for m in ("central9", "contralateral", "minimal3", "cz", "headband_ring")}
    er.plot_montages(montages, names, LINKED_MASTOID, run_dir / "montages.png")
    shown = table[[c for c in SHOWN if c in table.columns]]
    (run_dir / "summary.md").write_text(
        "# Electrode reduction curve\n\nCNN-LSTM 2 s, D11 budget. d_* = paired difference to all 32 channels of the "
        "same data and form (inner validation: per fold; test: per subject); n_up = folds/subjects above it. "
        "Reference (a) car = CAR over 32 then select; (b) mastoid = linked mastoid (TP9+TP10)/2. ICA is fitted on "
        "32 channels, so cleaned small montages are not realizable by a small device. Fewer channels also means "
        "fewer spatial-layer parameters (confound). Epoch times of the all-32 comparators come from runs that shared "
        "the GPU with other runs, so they overstate the cost of 32 channels; compare the new montage runs, which run "
        "one at a time, with each other.\n\n" + shown.round(3).to_markdown(index=False) + "\n",
        encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(shown.round(3).to_string(index=False))
    print(f"Run directory: {run_dir}")
    teardown_logging()


if __name__ == "__main__":
    main()
