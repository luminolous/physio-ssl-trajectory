"""Add ``residual_r_matched`` (design.md SSL-27) to finished decoder runs from their saved predictions. No training.

Runs scored before SSL-27 have ``residual_r_specific`` in metrics.json but not ``residual_r_matched``. For every run
under output/proposed-model/decoder/<name>__*/<run>/ with metrics.json and predictions.npz (smoke runs skipped), each
test subject is scored again with ``eegtraj.baselines.scoring.subject_scores``, the runner's own scoring path: the
fold's template_cue from the training subjects, the run's scoring seed and permutations. ``residual_r_specific`` is
recomputed as a check against metrics.json. The run directories are not changed.

The head floor and template_cue are one function of time since LED-on per subject, so their matched score is 0 by
construction (tests/test_residual_matched.py; diagnostics/head-floor-rrs); they are not rescored here.

Writes to a new output/proposed-model/rescoring/matched/<run-id>/: scores.csv (one row per run and test subject),
summary.md, metrics.json. Summary scripts read the newest scores.csv.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\rescore_matched.py
    .venv\\Scripts\\python scripts\\rescore_matched.py --names exp1 exp1-occl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.scoring import fit_cue_template, subject_scores
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.lift_streams import build_subject_streams
from eegtraj.data.windows import SubjectSamples
from eegtraj.utils.run import setup_run

logger = logging.getLogger("rescore_matched")
DECODER = Path("output/proposed-model/decoder")
OUT = Path("output/proposed-model/rescoring/matched")
KEEP = ("r", "residual_r", "residual_r_other_trial", "residual_r_specific", "residual_r_matched",
        "residual_r_matched_own", "residual_r_matched_cross")


def finished_runs(names: list[str]) -> list[Path]:
    out = []
    for name in names:
        for run in sorted(DECODER.glob(f"{name}__*/*")):
            if (run / "metrics.json").exists() and (run / "predictions.npz").exists() and "smoke" not in run.name:
                out.append(run)
    return out


def test_mask(s: SubjectSamples, sample_idx: np.ndarray, trial_row: np.ndarray) -> np.ndarray:
    """Mask of the saved samples in ``s``; raises if the saved order is not the sample order."""
    mask = np.isin(s.sample_idx, sample_idx)
    if not (np.array_equal(s.sample_idx[mask], sample_idx) and np.array_equal(s.trial_row[mask], trial_row)):
        raise RuntimeError(f"{s.subject}: saved samples are not the subject's scored samples in order")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", nargs="*", default=["exp1", "exp1-occl"], help="run name prefixes (before __)")
    args = parser.parse_args()
    runs = finished_runs(args.names)
    run_dir = setup_run(OUT, {"names": args.names, "runs": [r.as_posix() for r in runs]})
    logging.getLogger("eegtraj.baselines").setLevel(logging.WARNING)
    samples: dict[tuple, SubjectSamples] = {}
    templates: dict[tuple, object] = {}
    rows = []
    for run in runs:
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        cache, split, mode = cfg["cache_dir"], cfg["split_config"], cfg.get("folds", "loso")
        seed, n_perm = int(cfg["scoring"]["seed"]), int(cfg["scoring"]["n_permutations"])

        def get(s: str) -> SubjectSamples:
            if (cache, s) not in samples:
                samples[(cache, s)] = build_subject_streams(cache, s).samples  # as the runner
            return samples[(cache, s)]

        with np.load(run / "predictions.npz") as z:
            for fold in make_folds(load_split_config(split), mode):
                for s in fold.test_subjects:
                    if s not in m["per_subject"]:
                        continue
                    key = (cache, split, mode, fold.name)
                    if key not in templates:
                        train = [get(t) for t in fold.train_subjects]
                        ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0,
                                         train_masks={t.subject: np.ones(t.sample_idx.size, bool) for t in train},
                                         val_masks={}, fold_name=fold.name)
                        templates[key] = (fit_cue_template(train, ctx), ctx)
                    template, ctx = templates[key]
                    mask = test_mask(get(s), z[f"{s}__sample_idx"], z[f"{s}__trial_row"])
                    sc = subject_scores(get(s), mask, z[f"{s}__y"].astype(np.float64), template, ctx, seed, n_perm)
                    rows.append({"run": run.as_posix(), "name": run.parent.name, "seed": m["decoder_seed"],
                                 "subject": s, "rrs_saved": m["per_subject"][s]["residual_r_specific"],
                                 "r_saved": m["per_subject"][s]["r"], **{k: sc[k] for k in KEEP}})
        logger.info("%s: %d subjects", run, sum(r["run"] == run.as_posix() for r in rows))
    table = pd.DataFrame(rows)
    diff_rrs = float((table.residual_r_specific - table.rrs_saved).abs().max())
    diff_r = float((table.r - table.r_saved).abs().max())
    table.to_csv(run_dir / "scores.csv", index=False)
    per_run = table.groupby("name")[["residual_r_specific", "residual_r_matched"]].mean().round(4)
    md = (f"# residual_r_matched for saved decoder runs\n\n{len(runs)} runs, {len(table)} run x subject rows. "
          f"Reproduction against metrics.json: max |rrs difference| {diff_rrs:.2e}, max |r difference| {diff_r:.2e}.\n\n"
          "Mean over runs (seeds) and subjects:\n\n" + per_run.to_markdown() + "\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"n_runs": len(runs), "max_abs_diff_rrs": diff_rrs,
                                                      "max_abs_diff_r": diff_r}, indent=2) + "\n", encoding="utf-8")
    print(md)
    print(f"Run directory: {run_dir}")
    if diff_rrs > 1e-6 or diff_r > 1e-6:
        raise RuntimeError("rescored values differ from metrics.json; scores.csv kept for inspection")


if __name__ == "__main__":
    main()
