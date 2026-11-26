"""Neural-branch occlusion summary for Experiment 1 (design.md SSL-22). No training.

Reads the finished runs under output/proposed-model/decoder/<name>__*/ (default name ``exp1-occl``; also ``exp1`` for
stage 2 runs, which carry occlusion). For each configuration and channel group: the drop (full - occluded) in r and
residual_r_specific per test subject (mean over seeds when several), its mean over subjects, n_pos (subjects with a
positive drop) and the Wilcoxon p against 0. Paired guided - unguided comparisons of the drop (same objective, same
setting and branches). Reproducibility: each run's aggregate r against the matching Experiment 1 stage 1 run with the
same seed (``exp1``), when present.

Trial-specific metric (design.md SSL-27): residual_r_matched is the main one. The runs saved no occluded predictions,
so the matched drops come from the newest scripts/rescore_occlusion.py table (saved fold models); without it they are
NaN and the summary says so. A drop is a finding only if significant on residual_r_matched; readings that change from
residual_r_specific are listed.

Writes to a new output/proposed-model/summary/occlusion/<run-id>/: drops.csv, summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\summarize_occlusion.py
    .venv\\Scripts\\python scripts\\summarize_occlusion.py --names exp1-occl exp1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eegtraj.utils.rescored import matched_scores, occlusion_drops, subject_matched, verdicts
from eegtraj.utils.run import setup_run

DECODER = Path("output/proposed-model/decoder")
OUT = Path("output/proposed-model/summary/occlusion")
METRICS = ("r", "residual_r_specific", "residual_r_matched")
SAVED = METRICS[:2]


def p_value(d: pd.Series) -> float:
    d = d.dropna()
    return float(wilcoxon(d).pvalue) if len(d) >= 5 and (d != 0).any() else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", nargs="+", default=["exp1-occl"])
    args = parser.parse_args()
    rows, repro = [], []
    matched, matched_source = matched_scores()
    occ_matched, occ_source = occlusion_drops()
    for name in args.names:
        for mfile in sorted(DECODER.glob(f"{name}__*/*/metrics.json")):
            run = mfile.parent
            if run.name.endswith("_smoke"):
                continue
            m = json.loads(mfile.read_text(encoding="utf-8"))
            if not m.get("occlusion"):
                continue
            config = run.parent.name.split("__", 1)[1].removeprefix("proposed_")
            seed = m["decoder_seed"]
            for s, groups in m["occlusion"].items():
                for g, rec in groups.items():
                    rows.append({"config": config, "seed": seed, "subject": s, "group": g, "run": str(run),
                                 **{f"full_{k}": m["per_subject"][s][k] for k in SAVED},
                                 "full_residual_r_matched": subject_matched(run, s, m["per_subject"][s], matched),
                                 **{f"drop_{k}": rec["drop"][k] for k in SAVED},
                                 "drop_residual_r_matched": rec["drop"].get(
                                     "residual_r_matched", occ_matched.get((run.as_posix(), s, g), np.nan))})
            ref = [p.parent for p in DECODER.glob(f"exp1__proposed_{config}/*_seed{seed}/metrics.json")
                   if p.parent != run and not p.parent.name.endswith("_smoke")]
            if ref:
                rm = json.loads((ref[0] / "metrics.json").read_text(encoding="utf-8"))
                repro.append({"config": config, "seed": seed, "r_this": m["aggregate"]["r"]["mean"],
                              "r_stage1": rm["aggregate"]["r"]["mean"],
                              "rrs_this": m["aggregate"]["residual_r_specific"]["mean"],
                              "rrs_stage1": rm["aggregate"]["residual_r_specific"]["mean"], "stage1_run": str(ref[0])})
    if not rows:
        raise SystemExit("no runs with occlusion")
    df = pd.DataFrame(rows)
    subj = df.groupby(["config", "group", "subject"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    out = []
    for (c, g), t in subj.groupby(["config", "group"]):
        rec = {"config": c, "group": g, "n_subjects": len(t)}
        for k in METRICS:
            d = t[f"drop_{k}"]
            if d.isna().all():
                rec.update({f"drop_{k}": np.nan, f"drop_{k}_sd": np.nan, f"n_pos_{k}": 0, f"p_{k}": np.nan,
                            f"rel_drop_{k}": np.nan})
                continue
            rec.update({f"drop_{k}": float(d.mean()), f"drop_{k}_sd": float(d.std(ddof=1)),
                        f"n_pos_{k}": int((d > 0).sum()), f"p_{k}": p_value(d),
                        f"rel_drop_{k}": float(d.mean() / t[f"full_{k}"].mean())})
        out.append(rec)
    table = pd.DataFrame(out).sort_values(["group", "config"]).reset_index(drop=True)
    pairs = []
    for c in subj.config.unique():
        if "_guided" not in c or c.startswith("ocular"):
            continue
        u = c.replace("_guided", "_unguided")
        for g in subj.group.unique():
            a = subj[(subj.config == c) & (subj.group == g)].set_index("subject")
            b = subj[(subj.config == u) & (subj.group == g)].set_index("subject")
            if len(a) and len(b):
                for k in METRICS:
                    d = (a[f"drop_{k}"] - b[f"drop_{k}"]).dropna()
                    pairs.append({"a": c, "b": u, "group": g, "metric": k, "mean_diff_of_drop": float(d.mean()),
                                  "n_pos": int((d > 0).sum()), "n_subjects": int(d.size), "p_wilcoxon": p_value(d)})
    pairs = pd.DataFrame(pairs)
    repro = pd.DataFrame(repro)
    have = df.drop_residual_r_matched.notna().any()
    long = pd.concat([table.assign(metric=k, mean_diff=table[f"drop_{k}"], n_pos=table[f"n_pos_{k}"],
                                   p_wilcoxon=table[f"p_{k}"]) for k in ("residual_r_specific", "residual_r_matched")])
    verdict = verdicts(long, ["config", "group"]) if have else pd.DataFrame()
    pair_verdict = (verdicts(pairs.rename(columns={"mean_diff_of_drop": "mean_diff"}), ["a", "group"])
                    if have and len(pairs) else pd.DataFrame())
    note = (f"Matched drops from {occ_source}." if have else "Matched drops not available: occluded predictions were "
            "not saved; run scripts/rescore_occlusion.py (saved fold models, GPU inference) and rerun this summary.")
    run_dir = setup_run(OUT, {"names": args.names, "runs": sorted(df.run.unique().tolist()),
                              "matched_source": matched_source, "occlusion_matched_source": occ_source})
    df.to_csv(run_dir / "drops.csv", index=False)
    show = ["group", "config", "n_subjects", "drop_r", "n_pos_r", "p_r", "drop_residual_r_matched",
            "n_pos_residual_r_matched", "p_residual_r_matched", "drop_residual_r_specific",
            "drop_residual_r_specific_sd", "n_pos_residual_r_specific", "p_residual_r_specific",
            "rel_drop_residual_r_specific"]
    md = ("# Neural-branch occlusion (Experiment 1)\n\nDrop = full - occluded score per test subject (seed mean), mean "
          "over subjects; channel group zeroed in the neural input only, ocular proxies intact. `p_*`: Wilcoxon of the "
          "drop against 0; `rel_drop_*`: mean drop / mean full score. " + note + "\n\n"
          + table[show].round(4).to_markdown(index=False)
          + "\n\n## Readings: residual_r_specific vs residual_r_matched (SSL-27)\n\n"
          + (verdict.round(4).to_markdown(index=False) if len(verdict) else note)
          + "\n\n## Guided - unguided (difference of drops)\n\n"
          + (pairs.round(4).to_markdown(index=False) if len(pairs) else "none") + "\n\n"
          + (pair_verdict.round(4).to_markdown(index=False) if len(pair_verdict) else "")
          + "\n\n## Reproducibility vs stage 1\n\n"
          + (repro.drop(columns="stage1_run").round(4).to_markdown(index=False) if len(repro) else "none") + "\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"drops": table.to_dict(orient="records"),
                                                     "guided_vs_unguided": pairs.to_dict(orient="records"),
                                                     "verdicts": verdict.to_dict(orient="records"),
                                                     "occlusion_matched_source": occ_source,
                                                     "reproducibility": repro.to_dict(orient="records")}, indent=2,
                                                    default=float) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table[show].round(4).to_string(index=False))
    if len(pairs):
        print(pairs.round(4).to_string(index=False))
    if len(repro):
        print(repro.drop(columns="stage1_run").round(4).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
