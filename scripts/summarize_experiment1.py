"""Experiment 1 summary (RQ1, design.md SSL-21). No training.

Reads every finished run under output/proposed-model/decoder/exp1__*/ (metrics.json; interrupted runs without it are
ignored). Per configuration (branches x setting x encoder), each subject's score is averaged over the decoder seeds
first; tables report the mean and SD over the 10 LOSO test subjects, and the SD over seeds of the aggregate.

Paired comparisons over the 10 subjects (seed-averaged scores, Wilcoxon signed-rank, two-sided; n_pos = subjects with
a positive difference):
- hybrid - ocular-only, per encoder and setting (does the neural branch add to the ocular proxies?);
- pretrained - random-init control, per branches (frozen), and lejepa_guided - scratch (fine-tune);
- each configuration - template_cue (floor) and - head floor (cue phase prior).
Metrics: r, r2, nrmse, residual_r_specific, residual_r_matched (design.md SSL-27: the main trial-specific metric; for
runs scored before SSL-27 it comes from the newest scripts/rescore_matched.py table). The head floor and template_cue
are rigid (one function of time since LED-on), so their residual_r_matched is 0 by construction. A trial-specific
difference is a finding only if significant on residual_r_matched; the summary lists every comparison whose reading
changes from residual_r_specific to residual_r_matched.

Writes to a new output/proposed-model/summary/experiment1/<run-id>/: configs.csv, subjects.csv, paired.csv,
summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\summarize_experiment1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eegtraj.utils.rescored import matched_scores, subject_matched, verdicts
from eegtraj.utils.run import setup_run

DECODER = Path("output/proposed-model/decoder")
HEAD_FLOOR = Path("output/proposed-model/diagnostics/head-floor-rrs")
OUT = Path("output/proposed-model/summary/experiment1")
METRICS = ("r", "r2", "nrmse", "residual_r_specific", "residual_r_matched")
SAVED = METRICS[:4]
ENCODERS = ("lejepa_guided", "lejepa_unguided", "jepa_guided", "jepa_unguided", "mae", "random0", "noenc")


def collect() -> tuple[pd.DataFrame, str | None]:
    rows = []
    table, source = matched_scores()
    for mfile in sorted(DECODER.glob("exp1__*/*/metrics.json")):
        run = mfile.parent
        if run.name.endswith("_smoke"):
            continue
        m = json.loads(mfile.read_text(encoding="utf-8"))
        _, rest = run.parent.name.split("__", 1)
        head, branches, setting, encoder = rest.split("_", 3)
        for s, sc in m["per_subject"].items():
            rows.append({"branches": branches, "setting": setting, "encoder": encoder, "seed": m["decoder_seed"],
                         "subject": s, "run": str(run), **{k: sc[k] for k in SAVED},
                         "residual_r_matched": subject_matched(run, s, sc, table),
                         **{f"floor_{k}": m["floor_template_cue"][s][k] for k in SAVED},
                         **{f"headfloor_{k}": m["floor_cue_phi"][s][k] for k in SAVED},
                         "floor_residual_r_matched": 0.0, "headfloor_residual_r_matched": 0.0})
    if not rows:
        raise SystemExit(f"no finished Experiment 1 runs under {DECODER}")
    return pd.DataFrame(rows), source


def paired(a: pd.Series, b: pd.Series) -> dict:
    """a - b over subjects (aligned by index)."""
    d = (a - b).dropna()
    p = float(wilcoxon(d).pvalue) if len(d) >= 5 and (d != 0).any() else np.nan
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)), "n_subjects": int(len(d)),
            "n_pos": int((d > 0).sum()), "p_wilcoxon": p}


def rrs_note() -> str:
    """Head-floor rrs check: why a rigid predictor can have residual_r_specific above 0 (latest diagnostic run)."""
    title = "\n## Note on residual_r_specific\n\n"
    runs = sorted(p for p in HEAD_FLOOR.glob("*") if (p / "rigid_summary.csv").exists())
    if not runs:
        return title + "No head-floor rrs check found (scripts/diagnose_head_floor_rrs.py).\n"
    r = pd.read_csv(runs[-1] / "rigid_summary.csv").pivot(index="group", columns="window", values="rrs")
    c = pd.read_csv(runs[-1] / "configs.csv")
    return (title + f"Source: {runs[-1].as_posix()}. `residual_r` uses the whole scored window, the other-trial control "
            "only the bins two trials share. A predictor that is one function of time since LED-on therefore has rrs "
            "different from 0 on the scored windows, and 0 on one fixed window. Cause: the variable window end "
            "(tHandStop), see `fixed_end`. rrs above 0 alone is no evidence of trial-specific information.\n\n"
            + r.round(4).to_markdown() + f"\n\nExperiment 1 configurations: window effect {c.window_effect.min():.3f} to "
            f"{c.window_effect.max():.3f}; matched-bin score (0 for every rigid predictor) {c.matched.min():.3f} to "
            f"{c.matched.max():.3f}.\n")


def main() -> None:
    raw, matched_source = collect()
    key = ["branches", "setting", "encoder"]
    subj = raw.groupby(key + ["subject"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    n_seeds = raw.groupby(key).seed.nunique().rename("n_seeds")
    seed_agg = raw.groupby(key + ["seed"])[list(METRICS)].mean().groupby(key).std(ddof=1).add_suffix("_seed_sd")
    cfg = subj.groupby(key)[list(METRICS) + ["floor_r", "headfloor_r"]].agg(["mean", "std"])
    cfg.columns = [f"{a}_{'sd' if b == 'std' else 'mean'}" for a, b in cfg.columns]
    cfg = cfg.join(n_seeds).join(seed_agg).reset_index()
    cfg["encoder"] = pd.Categorical(cfg.encoder, ENCODERS, ordered=True)
    cfg = cfg.sort_values(["setting", "branches", "encoder"], ascending=[False, True, True]).reset_index(drop=True)

    def scores(branches: str, setting: str, encoder: str) -> pd.DataFrame | None:
        t = subj[(subj.branches == branches) & (subj.setting == setting) & (subj.encoder == encoder)]
        return t.set_index("subject") if len(t) else None

    pairs = []
    ocular = scores("ocular", "frozen", "noenc")
    for (branches, setting, encoder), _ in subj.groupby(key):
        cur = scores(branches, setting, encoder)
        for k in METRICS:
            pairs.append({"comparison": "config - template_cue", "a": f"{branches}/{setting}/{encoder}", "b": "template_cue",
                          "metric": k, **paired(cur[k], cur[f"floor_{k}"])}) if k != "residual_r_specific" else None
            pairs.append({"comparison": "config - head floor", "a": f"{branches}/{setting}/{encoder}", "b": "head floor",
                          "metric": k, **paired(cur[k], cur[f"headfloor_{k}"])}) if k != "residual_r_specific" else None
        if branches == "hybrid" and ocular is not None:
            for k in METRICS:
                pairs.append({"comparison": "hybrid - ocular-only", "a": f"hybrid/{setting}/{encoder}",
                              "b": "ocular/frozen/noenc", "metric": k, **paired(cur[k], ocular[k])})
        if setting == "frozen" and encoder not in ("random0", "noenc"):
            ctrl = scores(branches, "frozen", "random0")
            if ctrl is not None:
                for k in METRICS:
                    pairs.append({"comparison": "pretrained - random-init (frozen)", "a": f"{branches}/frozen/{encoder}",
                                  "b": f"{branches}/frozen/random0", "metric": k, **paired(cur[k], ctrl[k])})
        if setting == "finetune" and encoder != "random0":
            ctrl = scores(branches, "finetune", "random0")
            if ctrl is not None:
                for k in METRICS:
                    pairs.append({"comparison": "pretrained - scratch (fine-tune)", "a": f"{branches}/finetune/{encoder}",
                                  "b": f"{branches}/finetune/random0", "metric": k, **paired(cur[k], ctrl[k])})
        if setting == "finetune":
            froz = scores(branches, "frozen", encoder)
            if froz is not None:
                for k in METRICS:
                    pairs.append({"comparison": "fine-tune - frozen", "a": f"{branches}/finetune/{encoder}",
                                  "b": f"{branches}/frozen/{encoder}", "metric": k, **paired(cur[k], froz[k])})
    pairs = pd.DataFrame([p for p in pairs if p is not None])

    verdict = verdicts(pairs[~pairs.comparison.str.startswith("config - ")], ["comparison", "a"])
    changed = verdict[verdict.changed]
    hf = pairs[(pairs.comparison == "config - head floor") & (pairs.metric == "residual_r_matched")]
    run_dir = setup_run(OUT, {"runs": sorted(raw.run.unique().tolist()), "matched_source": matched_source})
    cfg.to_csv(run_dir / "configs.csv", index=False)
    subj.to_csv(run_dir / "subjects.csv", index=False)
    pairs.to_csv(run_dir / "paired.csv", index=False)
    verdict.to_csv(run_dir / "verdicts.csv", index=False)
    show = ["setting", "branches", "encoder", "n_seeds", "r_mean", "r_sd", "r_seed_sd", "r2_mean", "nrmse_mean",
            "residual_r_matched_mean", "residual_r_matched_sd", "residual_r_specific_mean", "residual_r_specific_sd",
            "floor_r_mean", "headfloor_r_mean"]
    pv = pairs[pairs.comparison != "config - template_cue"]
    pv = pv[pv.comparison != "config - head floor"]
    pv = pv[~pv.metric.isin(["residual_r_specific", "residual_r_matched"])]
    fl = pairs[pairs.comparison.isin(["config - template_cue", "config - head floor"]) & (pairs.metric == "r")]
    md = ("# Experiment 1 (RQ1): encoder x branches x setting\n\nLOSO, 10 test subjects; subject scores averaged over "
          "decoder seeds, then mean / SD over subjects; `r_seed_sd`: SD over seeds of the aggregate. Paired tests: "
          "Wilcoxon signed-rank over subjects (two-sided), `n_pos` = subjects with a positive difference.\n\n"
          "## Configurations\n\n" + cfg[show].round(4).to_markdown(index=False) +
          "\n\n## Paired comparisons (r, R2, NRMSE)\n\n" + pv.round(4).to_markdown(index=False) +
          "\n\n## Paired comparisons, trial-specific (SSL-27)\n\nA trial-specific difference is a finding only if "
          "p < 0.05 on residual_r_matched (Wilcoxon, uncorrected); significant on residual_r_specific only reads "
          f"'not shown'. Matched values for older runs from {matched_source or 'metrics.json'}.\n\n"
          + verdict.round(4).to_markdown(index=False) +
          f"\n\n### Readings that change ({len(changed)} of {len(verdict)})\n\n"
          + changed.round(4).to_markdown(index=False) +
          "\n\n## Versus the head floor on residual_r_matched\n\nThe head floor's matched score is 0, so this tests "
          "matched > 0. residual_r_specific is not compared with the floors (window effect, note below).\n\n"
          + hf.round(4).to_markdown(index=False) +
          "\n\n## Versus template_cue and the head floor (r)\n\n" + fl.round(4).to_markdown(index=False) + "\n")
    md += rrs_note()
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"configs": cfg.astype({"encoder": str}).to_dict(orient="records"),
                                                     "paired": pairs.to_dict(orient="records"),
                                                     "verdicts": verdict.to_dict(orient="records"),
                                                     "matched_source": matched_source}, indent=2, default=float)
                                          + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(cfg[show].round(4).to_string(index=False))
    print(pv.round(4).to_string(index=False))
    print(fl.round(4).to_string(index=False))
    print(changed.round(4).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
