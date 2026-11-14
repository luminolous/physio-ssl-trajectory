"""Landmark-registered templates per LOSO fold, with overlays against the cue-locked template. No training.

For every LOSO fold: fit T(phi) per weight condition from the fold's training subjects
(``eegtraj.models.landmark_template.fold_templates``) and the ``template_cue`` baseline on the same subjects. For the
overlay, phi is mapped to time since LED-on through the fold's median landmark times, so both curves share one axis.

Writes to a new output/proposed-model/data/templates/<run-id>/: templates_<fold>.npz, summary.csv, summary.md,
metrics.json, figures/overlay_<fold>.png and figures/overlay_all_folds.png.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\build_landmark_templates.py --config configs\\decoder\\templates.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import load_subject_samples
from eegtraj.diagnostics.oracle_warp_report import fold_models
from eegtraj.models.landmark_template import LANDMARK_EVENTS, LandmarkTemplates, fold_templates
from eegtraj.utils.run import setup_run

OUTPUT = Path("output/proposed-model/data/templates")
AXES = ("x", "y", "z")
COLORS = {165: "#4c78a8", 330: "#f58518", 660: "#54a24b"}


def on_time(tmpl: LandmarkTemplates, t_since_cue: np.ndarray) -> dict[int, np.ndarray]:
    """Each condition's template as a function of time since LED-on, via the fold's median landmarks: (n, 3)."""
    phi = np.interp(t_since_cue, tmpl.landmarks_since_cue_s, tmpl.fractions)
    return {c: tmpl(phi, c) for c in tmpl.conditions}


def draw(ax_row, tmpl: LandmarkTemplates, cue, t: np.ndarray, title: str) -> None:
    curves = on_time(tmpl, t)
    cue_vals = cue(t)
    for a, ax in enumerate(ax_row):
        for c, v in curves.items():
            ax.plot(t, v[:, a], color=COLORS[c], lw=1.4, label=f"{c} g")
        ax.plot(t, cue_vals[:, a], color="k", lw=1.2, ls="--", label="template_cue")
        for m in tmpl.landmarks_since_cue_s:
            ax.axvline(m, color="0.7", lw=0.6)
        ax.set_title(f"{title} {AXES[a]}", fontsize=8)
        ax.tick_params(labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = setup_run(OUTPUT, cfg)
    print(f"Run directory: {run_dir}")
    (run_dir / "figures").mkdir()
    split = load_split_config(cfg["split_config"])
    folds = make_folds(split, "loso")
    needed = sorted({s for f in folds for s in f.train_subjects + f.val_subjects + f.test_subjects}, key=lambda s: int(s[1:]))
    subjects = [load_subject_samples(cfg["cache_dir"], s, "broadband", "car", "all", "position", history=1) for s in needed]
    t = np.arange(cfg["plot_window_s"][0], cfg["plot_window_s"][1], 0.01)
    rows = []
    fig_all, axes_all = plt.subplots(len(folds), 3, figsize=(11, 2.0 * len(folds)), sharex=True)
    for i, fold in enumerate(folds):
        tmpl = fold_templates(subjects, fold, conditions=tuple(cfg["conditions"]), n_grid=int(cfg["n_grid"]))
        cue = fold_models(subjects, fold)[0]
        np.savez(run_dir / f"templates_{fold.name}.npz", grid=tmpl.grid, values=tmpl.values, fractions=tmpl.fractions,
                 conditions=np.array(tmpl.conditions), landmarks_since_cue_s=tmpl.landmarks_since_cue_s,
                 subjects=np.array(tmpl.subjects))
        fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
        draw(axes, tmpl, cue, t, fold.name)
        axes[0].legend(fontsize=7)
        fig.supxlabel("time since LED-on (s); grey lines: median landmarks", fontsize=8)
        fig.supylabel("wrist position re-centered at onset (cm)", fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / f"overlay_{fold.name}.png", dpi=110)
        plt.close(fig)
        draw(axes_all[i], tmpl, cue, t, fold.name)
        curves = on_time(tmpl, t)
        mean_curve = np.mean([curves[c] for c in tmpl.conditions], axis=0)
        cue_vals = cue(t)
        row = {"fold": fold.name, "train_subjects": " ".join(tmpl.subjects), "n_excluded_lifts": tmpl.n_excluded,
               **{f"n_{c}g": tmpl.counts[c] for c in tmpl.conditions},
               **{f"frac_{e}": round(float(f), 4) for e, f in zip(LANDMARK_EVENTS, tmpl.fractions)},
               **{f"since_cue_{e}": round(float(m), 3) for e, m in zip(LANDMARK_EVENTS, tmpl.landmarks_since_cue_s)},
               **{f"peak_z_{c}g": round(float(tmpl.values[k, 2].max()), 2) for k, c in enumerate(tmpl.conditions)},
               "peak_z_cue": round(float(cue_vals[:, 2].max()), 2),
               **{f"r_vs_cue_{AXES[a]}": round(float(np.corrcoef(mean_curve[:, a], cue_vals[:, a])[0, 1]), 3) for a in range(3)}}
        rows.append(row)
        print(f"{fold.name}: counts {tmpl.counts}, fractions {np.round(tmpl.fractions, 3).tolist()}")
    axes_all[0, 0].legend(fontsize=6)
    fig_all.supxlabel("time since LED-on (s)")
    fig_all.tight_layout()
    fig_all.savefig(run_dir / "figures" / "overlay_all_folds.png", dpi=90)
    plt.close(fig_all)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.md").write_text(
        "# Landmark-registered templates per LOSO fold\n\nFitted on each fold's training subjects only "
        "(inner-validation and test subjects excluded). phi = 0 at the kinematic onset, 1 at the hand stop; landmark "
        "fractions are medians over the training lifts. Overlays map phi to time since LED-on through the fold's "
        "median landmark times; `r_vs_cue_*` correlates the condition-averaged template on that axis with "
        "`template_cue`. Peak z: maximum of the z axis (cm).\n\n" + table.to_markdown(index=False) + "\n",
        encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"folds": rows}, indent=2) + "\n", encoding="utf-8")
    print(table[["fold", "n_165g", "n_330g", "n_660g", "peak_z_165g", "peak_z_330g", "peak_z_660g", "peak_z_cue",
                 "r_vs_cue_x", "r_vs_cue_y", "r_vs_cue_z"]].to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
