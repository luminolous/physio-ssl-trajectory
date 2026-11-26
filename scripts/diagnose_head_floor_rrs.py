"""Why does the head floor have residual_r_specific (rrs) above 0? ANALYSIS ONLY, no training.

The head floor predicts one function of time since LED-on for every lift of a subject, so it holds no trial-specific
information. rrs = residual_r (same trial, the trial's own scored window [onset - 0.5 s, tHandStop]) minus
residual_r_other_trial (another trial, only the bins both windows share).

Part A, rigid predictors (one function of time since LED-on; LOSO folds, references from the training subjects only):
head floor; constant position (0); template_cue shifted by +-0.2 s; the mean training trajectory over all training
lifts at each time (no survivorship conditioning); 20 random smooth functions added to template_cue. Each is scored on:
- ``scored``: the scored window of every lift (the metric as reported);
- ``fixed``: one window for all lifts, [median(onset - 0.5 s - LED), median(tHandStop - LED)] of the training lifts;
- ``fixed_start`` / ``fixed_end``: only one edge fixed, the other edge as in ``scored``.
Lifts whose fixed window is not inside their lift window are left out (counted).
Decomposition: rrs = ``window_effect`` + (own - other_trial), with own = same-trial correlation on the shared bins of
the pairs (``eegtraj.metrics.residual_r_matched``); ``matched`` uses the shared bins for both terms and is exactly 0
for a rigid predictor.

Part B, finished Experiment 1 runs (saved predictions, scored window): rrs, ``matched``, ``window_effect`` per
configuration (seed-averaged per subject), and the paired comparisons of Experiments 1-2 under both.

The scored window and the fixed window's target use the test lift's kinematics: analysis only, never a model input.

Writes to a new output/proposed-model/diagnostics/head-floor-rrs/<run-id>/: rigid.csv, rigid_summary.csv,
configs.csv, subjects.csv, paired.csv, summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\diagnose_head_floor_rrs.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.scoring import fit_cue_template
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import SubjectSamples, load_subject_samples
from eegtraj.metrics import residual_r, residual_r_matched, residual_r_other_trial
from eegtraj.models.landmark_template import TemplateBank, expected_template, fit_phase_prior, fold_templates
from eegtraj.utils.run import setup_run

logger = logging.getLogger("diagnose_head_floor_rrs")
CACHE = "dataset/cache/98f5fa6e66"
SPLIT = "configs/splits/default.yaml"
DECODER = Path("output/proposed-model/decoder")
OUT = Path("output/proposed-model/diagnostics/head-floor-rrs")
CONDITIONS = (165, 330, 660)
N_GRID = 201
SEED, N_PERM = 0, 5  # as in configs/decoder/proposed.yaml (scoring)
N_RANDOM = 20
WINDOWS = ("scored", "fixed", "fixed_start", "fixed_end")


@dataclass
class Windowed:
    """One subject's samples in one window: target (n, 3), trial ids, time bins since LED-on, n lifts left out."""

    y: np.ndarray
    ids: np.ndarray
    bins: np.ndarray
    n_excluded: int


def windowed(s: SubjectSamples, kind: str, lo: float, hi: float) -> Windowed:
    """``lo`` / ``hi``: fixed window edges, s since LED-on (unused for ``scored``)."""
    fs = s.fs_hz
    y, ids, bins, excluded = [], [], [], 0
    for r, tr in enumerate(s.trials.itertuples(index=False)):
        pos = np.flatnonzero(s.trial_row == r)
        if not pos.size:
            continue
        j_all = np.arange(int(tr.offset_end - tr.offset_start))
        b_all = np.round(((j_all + 1) / fs - tr.ev_LEDOn) * fs).astype(np.int64)
        j = s.sample_idx[pos] - tr.offset_start
        if not np.array_equal(j, np.arange(j[0], j[-1] + 1)):
            raise RuntimeError(f"{s.subject} lift {r}: scored samples are not contiguous")
        b_lo, b_hi = b_all[j[0]], b_all[j[-1]]
        if kind in ("fixed", "fixed_start"):
            b_lo = int(round(lo * fs))
        if kind in ("fixed", "fixed_end"):
            b_hi = int(round(hi * fs))
        if b_lo < b_all[0] or b_hi > b_all[-1] or b_hi - b_lo < 50:
            excluded += 1
            continue
        m = (b_all >= b_lo) & (b_all <= b_hi)
        y.append(s.target[:, tr.offset_start + j_all[m]].T)
        ids.append(np.full(int(m.sum()), r))
        bins.append(b_all[m])
    return Windowed(np.concatenate(y).astype(np.float64), np.concatenate(ids), np.concatenate(bins), excluded)


def residual_scores(y: np.ndarray, pred: np.ndarray, cue: np.ndarray, ids: np.ndarray, bins: np.ndarray,
                    subject: str) -> dict[str, float]:
    key = [SEED, int(subject[1:])]
    res = residual_r(y, pred, cue, ids)["mean"]
    other = residual_r_other_trial(y, pred, cue, ids, bins, np.random.default_rng(key), N_PERM)["mean"]
    m = residual_r_matched(y, pred, cue, ids, bins, np.random.default_rng(key), N_PERM)
    return {"residual_r": res, "residual_r_other_trial": other, "rrs": res - other, "own_shared": m["own"],
            "cross_shared": m["cross"], "matched": m["matched"], "window_effect": res - m["own"],
            "shared_fraction": m["shared_fraction"]}


def unconditional_mean(train: list[SubjectSamples], bin0: int, n_bins: int) -> np.ndarray:
    """Mean target per bin since LED-on over all training lifts whose lift window holds the bin (3, n_bins)."""
    sums, counts = np.zeros((3, n_bins)), np.zeros(n_bins)
    for s in train:
        for tr in s.trials.itertuples(index=False):
            j = np.arange(int(tr.offset_end - tr.offset_start))
            b = np.round(((j + 1) / s.fs_hz - tr.ev_LEDOn) * s.fs_hz).astype(np.int64) - bin0
            ok = (b >= 0) & (b < n_bins)
            sums[:, b[ok]] += s.target[:, tr.offset_start + j[ok]]
            counts[b[ok]] += 1
    last = np.maximum.accumulate(np.where(counts > 0, np.arange(n_bins), 0))
    return sums[:, last] / counts[last]


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    p = float(wilcoxon(d).pvalue) if len(d) >= 5 and (d != 0).any() else np.nan
    return {"mean_diff": float(d.mean()), "n_subjects": int(len(d)), "n_pos": int((d > 0).sum()), "p_wilcoxon": p}


def main() -> None:
    run_dir = setup_run(OUT, {"cache_dir": CACHE, "split_config": SPLIT, "seed": SEED, "n_permutations": N_PERM,
                              "n_random_functions": N_RANDOM, "windows": list(WINDOWS)})
    folds = make_folds(load_split_config(SPLIT), "loso")
    names = sorted({s for f in folds for s in (*f.train_subjects, *f.val_subjects, *f.test_subjects)},
                   key=lambda s: int(s[1:]))
    subjects = {s: load_subject_samples(CACHE, s, "broadband", "car", "all", "position", 1) for s in names}
    cues: dict[str, np.ndarray] = {}
    floors: dict[str, np.ndarray] = {}
    scored: dict[str, Windowed] = {}
    rows = []
    for fold in folds:
        train = [subjects[s] for s in fold.train_subjects]
        ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0,
                         train_masks={s: np.ones(subjects[s].sample_idx.size, bool) for s in fold.train_subjects},
                         val_masks={}, fold_name=fold.name)
        template = fit_cue_template(train, ctx)
        tmpl = fold_templates([subjects[s] for s in names], fold, conditions=CONDITIONS, n_grid=N_GRID)
        prior = fit_phase_prior([subjects[s] for s in names], fold.train_subjects, tmpl.fractions)
        bank = TemplateBank.from_templates(tmpl)
        fs = template.fs_hz
        tr = pd.concat([s.trials for s in train])
        lo = float((tr.segment_start_s - tr.ev_LEDOn).median())
        hi = float((tr.segment_end_s - tr.ev_LEDOn).median())
        grid0, n_bins = int(-3 * fs), int(16 * fs)  # common grid of bins since LED-on
        g = np.arange(grid0, grid0 + n_bins)

        def at(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
            return values[:, np.clip(bins - grid0, 0, n_bins - 1)].T

        cue_grid = template.template[:, np.clip(g - template.bin0, 0, template.template.shape[1] - 1)].astype(np.float64)
        mu, sd = prior(g / fs)
        mu, sd = torch.from_numpy(np.asarray(mu)).float(), torch.from_numpy(np.asarray(sd)).float()
        w = torch.full((g.size, len(CONDITIONS)), 1.0 / len(CONDITIONS))
        head = expected_template(bank, mu, sd, w).numpy().astype(np.float64).T
        shift = int(0.2 * fs)
        rng = np.random.default_rng([SEED, 1000])
        predictors = {"head_floor": head, "constant_0": np.zeros_like(head),
                      "cue_shift_-0.2s": cue_grid[:, np.clip(np.arange(n_bins) + shift, 0, n_bins - 1)],
                      "cue_shift_+0.2s": cue_grid[:, np.clip(np.arange(n_bins) - shift, 0, n_bins - 1)],
                      "unconditional_mean": unconditional_mean(train, grid0, n_bins)}
        knots = np.linspace(g[0], g[-1], 39)  # 0.5 s between knots
        for i in range(N_RANDOM):
            predictors[f"random_{i:02d}"] = cue_grid + np.stack([np.interp(g, knots, rng.normal(size=knots.size))
                                                                 for _ in range(3)])
        for s in fold.test_subjects:
            for kind in WINDOWS:
                wd = windowed(subjects[s], kind, lo, hi)
                if kind == "scored":
                    scored[s], cues[s], floors[s] = wd, at(cue_grid, wd.bins), at(head, wd.bins)
                c = at(cue_grid, wd.bins)
                for name, values in predictors.items():
                    rows.append({"subject": s, "window": kind, "predictor": name, "n_excluded": wd.n_excluded,
                                 "lo_s": lo, "hi_s": hi,
                                 **residual_scores(wd.y, at(values, wd.bins), c, wd.ids, wd.bins, s)})
        logger.info("fold %s test %s: fixed window [%.2f, %.2f] s since LED-on", fold.name, fold.test_subjects, lo, hi)
    rigid = pd.DataFrame(rows)
    rigid["group"] = rigid.predictor.where(~rigid.predictor.str.startswith("random_"), "random_smooth (20)")
    vals = ["residual_r", "residual_r_other_trial", "rrs", "own_shared", "window_effect", "matched", "shared_fraction"]
    per_subject = rigid.groupby(["group", "window", "subject"])[vals + ["n_excluded"]].mean().reset_index()
    rs = per_subject.groupby(["group", "window"])
    rigid_summary = rs[vals].mean().join(rs.rrs.std(ddof=1).rename("rrs_sd")).join(
        rs.matched.agg(lambda x: x.abs().max()).rename("matched_abs_max")).join(
        rs.n_excluded.sum().rename("n_lifts_excluded")).reset_index()
    rnd = rigid[rigid.predictor.str.startswith("random_") & (rigid.window == "scored")].groupby("predictor").rrs.mean()

    runs = []
    max_floor_diff = 0.0
    for mfile in sorted(DECODER.glob("exp1__*/*/metrics.json")):
        run = mfile.parent
        if run.name.endswith("_smoke"):
            continue
        m = json.loads(mfile.read_text(encoding="utf-8"))
        _, branches, setting, encoder = run.parent.name.split("__", 1)[1].split("_", 3)
        with np.load(run / "predictions.npz") as z:
            for s in m["per_subject"]:
                wd = scored[s]
                if not (np.array_equal(z[f"{s}__trial_row"], wd.ids)
                        and np.allclose(z[f"{s}__y_true"], wd.y, atol=1e-4)):
                    raise RuntimeError(f"{run} {s}: saved samples differ from the scored windows")
                sc = residual_scores(wd.y, z[f"{s}__y"].astype(np.float64), cues[s], wd.ids, wd.bins, s)
                runs.append({"branches": branches, "setting": setting, "encoder": encoder, "seed": m["decoder_seed"],
                             "subject": s, "rrs_saved": m["per_subject"][s]["residual_r_specific"],
                             "floor_rrs_saved": m["floor_cue_phi"][s]["residual_r_specific"], **sc})
        logger.info("run %s", run)
    raw = pd.DataFrame(runs)
    floor = rigid[(rigid.predictor == "head_floor") & (rigid.window == "scored")].set_index("subject")
    max_floor_diff = float((raw.subject.map(floor.rrs) - raw.floor_rrs_saved).abs().max())
    max_model_diff = float((raw.rrs - raw.rrs_saved).abs().max())
    logger.info("reproduction: max |rrs - saved| models %.2e, head floor %.2e", max_model_diff, max_floor_diff)

    key = ["branches", "setting", "encoder"]
    keep = ["rrs", "matched", "window_effect", "residual_r", "own_shared", "shared_fraction"]
    subj = raw.groupby(key + ["subject"])[keep].mean().reset_index()
    cfg = subj.groupby(key)[keep].mean().join(subj.groupby(key)[["rrs", "matched"]].std(ddof=1).add_suffix("_sd"))
    cfg = cfg.reset_index().sort_values("rrs", ascending=False)

    def get(b: str, st: str, e: str) -> pd.DataFrame | None:
        t = subj[(subj.branches == b) & (subj.setting == st) & (subj.encoder == e)]
        return t.set_index("subject") if len(t) else None

    pairs = []
    for (b, st, e), _ in subj.groupby(key):
        cur = get(b, st, e)
        comps = {"config - head floor": floor}
        if b == "hybrid":
            comps["hybrid - ocular-only"] = get("ocular", "frozen", "noenc")
        if b == "neural":
            comps["neural - ocular-only"] = get("ocular", "frozen", "noenc")
        if e not in ("random0", "noenc"):
            comps["pretrained - random-init / scratch"] = get(b, st, "random0")
        for c, ref in comps.items():
            for k in ("rrs", "matched"):
                if ref is not None:
                    pairs.append({"comparison": c, "a": f"{b}/{st}/{e}", "metric": k, **paired(cur[k], ref[k])})
    pairs = pd.DataFrame(pairs)
    wide = pairs.pivot(index=["comparison", "a"], columns="metric", values=["mean_diff", "n_pos", "p_wilcoxon"])
    wide.columns = [f"{k}_{m}" for k, m in wide.columns]
    wide = wide.reset_index()

    rigid.to_csv(run_dir / "rigid.csv", index=False)
    rigid_summary.to_csv(run_dir / "rigid_summary.csv", index=False)
    cfg.to_csv(run_dir / "configs.csv", index=False)
    subj.to_csv(run_dir / "subjects.csv", index=False)
    pairs.to_csv(run_dir / "paired.csv", index=False)
    md = ("# Head-floor rrs check\n\nMean over the 10 LOSO test subjects. rrs = residual_r - residual_r_other_trial; "
          "window_effect = residual_r (whole scored window) - same-trial correlation on the shared bins; matched = "
          "same-trial - other-trial, both on the shared bins. A rigid predictor is one function of time since LED-on.\n\n"
          "## A. Rigid predictors\n\n" + rigid_summary.round(4).to_markdown(index=False)
          + f"\n\nRandom smooth functions, scored window, mean rrs per function over subjects: min {rnd.min():.4f}, "
            f"median {rnd.median():.4f}, max {rnd.max():.4f}; {int((rnd > 0).sum())} / {len(rnd)} positive.\n\n"
          "## B. Experiment 1 configurations (scored window, seed-averaged)\n\n" + cfg.round(4).to_markdown(index=False)
          + "\n\n## C. Paired comparisons, rrs vs matched\n\n" + wide.round(4).to_markdown(index=False)
          + f"\n\nReproduction of saved rrs: max |difference| models {max_model_diff:.2e}, head floor {max_floor_diff:.2e}.\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"rigid": rigid_summary.to_dict(orient="records"), "configs": cfg.to_dict(orient="records"),
         "paired": pairs.to_dict(orient="records"), "max_model_diff": max_model_diff, "max_floor_diff": max_floor_diff,
         "random_rrs": rnd.to_dict()}, indent=2, default=float) + "\n", encoding="utf-8")
    print(md)
    print(run_dir)


if __name__ == "__main__":
    main()
