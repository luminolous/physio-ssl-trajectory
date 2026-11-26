"""Experiment 2 summary (RQ2: physiological branches, design.md SSL-21 / SSL-22). ANALYSIS ONLY, no training.

Reuses the Experiment 1 runs in the Experiments 2-3 setting (fine-tune, SSL-21 rule): the proposed head with the
proposed encoder (lejepa_guided) on the neural branch only, on the hybrid (neural + ocular) branches, and the
ocular-only decoder (no encoder, so the setting does not apply), 3 decoder seeds each (``exp1``). Group occlusion comes
from the seed 1 reruns with in-run neural-branch occlusion (``exp1-occl``; channel group zeroed in the neural input,
ocular proxies intact).

Reference rows (diagnostics stage, different decoder: CNN-LSTM direct, 2 s history, LOSO, uncleaned EEG):
the prefrontal-only control (Fp1/Fp2 input only) and the group-occlusion drops, read from the final ocular check.

Outputs: per-subject scores (seed mean), the branch table, paired Wilcoxon comparisons (hybrid - neural, hybrid - ocular,
neural - ocular, each branch - template_cue and - head floor), occlusion drops per group and branch, and a figure.

Trial-specific metric (design.md SSL-27): residual_r_matched is the main one, residual_r_specific is kept next to it.
Runs scored before SSL-27 take residual_r_matched from the newest scripts/rescore_matched.py table and the matched
occlusion drops from the newest scripts/rescore_occlusion.py table (NaN and a note when that table is missing). The
head floor and template_cue are rigid, so their residual_r_matched is 0. A trial-specific difference is a finding only
if significant on residual_r_matched; every comparison whose reading changes is listed.

Writes to a new output/proposed-model/summary/experiment2/<run-id>/: per_subject.csv, occlusion.csv, summary.md,
metrics.json, experiment2.png.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\summarize_experiment2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from eegtraj.utils.rescored import matched_scores, occlusion_drops, subject_matched, verdicts  # noqa: E402
from eegtraj.utils.run import setup_run  # noqa: E402

DECODER = Path("output/proposed-model/decoder")
OUT = Path("output/proposed-model/summary/experiment2")
REF = Path("output/diagnostics/final-checks/ocular/summary/20261108-115730")
BRANCHES = {"neural": "exp1__proposed_neural_finetune_lejepa_guided",
            "hybrid": "exp1__proposed_hybrid_finetune_lejepa_guided",
            "ocular": "exp1__proposed_ocular_frozen_noenc"}
OCCL = {"neural": "exp1-occl__proposed_neural_finetune_lejepa_guided",
        "hybrid": "exp1-occl__proposed_hybrid_finetune_lejepa_guided"}
GROUPS = ("prefrontal", "frontal", "occipital", "contralateral", "ipsilateral")
METRICS = ("r", "r2", "nrmse", "residual_r_specific", "residual_r_matched")
SAVED = METRICS[:4]
TRIAL = ("residual_r_specific", "residual_r_matched")


def runs(folder: str) -> list[Path]:
    return sorted(p.parent for p in (DECODER / folder).glob("*/metrics.json") if not p.parent.name.endswith("_smoke"))


def load(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def p_value(d: pd.Series) -> float:
    d = d.dropna()
    return float(wilcoxon(d).pvalue) if len(d) >= 5 and (d != 0).any() else np.nan


def main() -> None:
    rows, floors, used = [], [], []
    matched, matched_source = matched_scores()
    occ_matched, occ_source = occlusion_drops()
    for branch, folder in BRANCHES.items():
        rs = runs(folder)
        if not rs:
            raise SystemExit(f"no runs in {folder}")
        for run in rs:
            m = load(run)
            used.append(str(run))
            for s, sc in m["per_subject"].items():
                rows.append({"branch": branch, "seed": m["decoder_seed"], "subject": s, **{k: sc[k] for k in SAVED},
                             "residual_r_matched": subject_matched(run, s, sc, matched)})
                for name, key in (("template_cue", "floor_template_cue"), ("head_floor", "floor_cue_phi")):
                    floors.append({"branch": name, "seed": m["decoder_seed"], "subject": s,
                                   **{k: m[key][s].get(k, np.nan) for k in SAVED}, "residual_r_matched": 0.0})
    df = pd.DataFrame(rows)
    per_subj = df.groupby(["branch", "subject"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    seed_sd = df.groupby(["branch", "seed"]).mean(numeric_only=True).groupby("branch").std(ddof=1)
    fl = pd.DataFrame(floors).groupby(["branch", "subject"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    allp = pd.concat([per_subj, fl], ignore_index=True)
    table = allp.groupby("branch")[list(METRICS)].mean().reindex(["neural", "hybrid", "ocular", "head_floor",
                                                                  "template_cue"])
    for k in ("r", "residual_r_matched", "residual_r_specific"):
        table[f"{k}_seed_sd"] = seed_sd[k].reindex(table.index)
    wide = {k: allp.pivot(index="subject", columns="branch", values=k) for k in METRICS}
    pairs = []
    for a, b in (("hybrid", "neural"), ("hybrid", "ocular"), ("neural", "ocular"), ("neural", "template_cue"),
                 ("hybrid", "template_cue"), ("ocular", "template_cue"), ("neural", "head_floor"),
                 ("hybrid", "head_floor"), ("ocular", "head_floor")):
        for k in METRICS:
            if k == "residual_r_specific" and b in ("template_cue", "head_floor"):
                continue  # window effect: rrs is not compared with the rigid floors (SSL-27)
            d = (wide[k][a] - wide[k][b]).dropna()
            if d.empty:
                continue
            pairs.append({"a": a, "b": b, "metric": k, "mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()),
                          "n_subjects": int(d.size), "p_wilcoxon": p_value(d)})
    pairs = pd.DataFrame(pairs)
    verdict = verdicts(pairs[~pairs.b.isin(["template_cue", "head_floor"])], ["a", "b"])

    occ = []
    for branch, folder in OCCL.items():
        for run in runs(folder):
            m = load(run)
            used.append(str(run))
            for s, groups in m["occlusion"].items():
                for g, rec in groups.items():
                    occ.append({"branch": branch, "seed": m["decoder_seed"], "subject": s, "group": g,
                                **{f"full_{k}": m["per_subject"][s][k] for k in ("r", "residual_r_specific")},
                                "full_residual_r_matched": subject_matched(run, s, m["per_subject"][s], matched),
                                **{f"drop_{k}": rec["drop"][k] for k in ("r", "residual_r_specific")},
                                "drop_residual_r_matched": rec["drop"].get(
                                    "residual_r_matched", occ_matched.get((run.as_posix(), s, g), np.nan))})
    occ = pd.DataFrame(occ)
    occ_s = occ.groupby(["branch", "group", "subject"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    occ_tab = []
    for (b, g), t in occ_s.groupby(["branch", "group"]):
        rec = {"branch": b, "group": g, "n_subjects": len(t)}
        for k in ("r",) + TRIAL:
            d = t[f"drop_{k}"]
            rec.update({f"drop_{k}": float(d.mean()), f"n_pos_{k}": int((d > 0).sum()), f"p_{k}": p_value(d),
                        f"rel_drop_{k}": float(d.mean() / t[f"full_{k}"].mean())})
        occ_tab.append(rec)
    occ_tab = pd.DataFrame(occ_tab)
    occ_tab["group"] = pd.Categorical(occ_tab.group, GROUPS, ordered=True)
    occ_tab = occ_tab.sort_values(["group", "branch"]).reset_index(drop=True)
    occ_pairs = []
    for g in GROUPS:
        a = occ_s[(occ_s.branch == "hybrid") & (occ_s.group == g)].set_index("subject")
        b = occ_s[(occ_s.branch == "neural") & (occ_s.group == g)].set_index("subject")
        for k in TRIAL:
            d = (a[f"drop_{k}"] - b[f"drop_{k}"]).dropna()
            if len(d):
                occ_pairs.append({"group": g, "metric": k, "mean_diff": float(d.mean()),
                                  "n_pos": int((d > 0).sum()), "n_subjects": int(d.size), "p_wilcoxon": p_value(d)})
    occ_pairs = pd.DataFrame(occ_pairs)
    occ_long = pd.concat([occ_tab.assign(metric=k, mean_diff=occ_tab[f"drop_{k}"], n_pos=occ_tab[f"n_pos_{k}"],
                                         p_wilcoxon=occ_tab[f"p_{k}"]) for k in TRIAL])
    occ_verdict = (verdicts(occ_long.astype({"group": str}), ["branch", "group"])
                   if occ_tab.drop_residual_r_matched.notna().any() else pd.DataFrame())
    occ_pair_verdict = verdicts(occ_pairs, ["group"]) if occ_tab.drop_residual_r_matched.notna().any() else pd.DataFrame()
    occ_note = (f"Matched drops from {occ_source}." if occ_source else
                "Matched drops not available: occluded predictions were not saved; run scripts/rescore_occlusion.py "
                "(saved fold models, GPU inference) and rerun this summary.")

    ref_ctrl = pd.read_csv(REF / "comparison.csv")
    ref_ctrl = ref_ctrl[(ref_ctrl.model.isin(["prefrontal only", "direct"])) & (ref_ctrl.split == "test")
                        & ref_ctrl.metric.isin(["r", "residual_r_specific"])]
    ref_ctrl = ref_ctrl.pivot(index="model", columns="metric", values="uncleaned").rename(
        index={"direct": "CNN-LSTM direct, all 32 channels", "prefrontal only": "CNN-LSTM direct, Fp1/Fp2 only"})
    ref_occ = pd.read_csv(REF / "occlusion_groups.csv").set_index("occluded")[
        ["uncleaned_r", "uncleaned_residual_r_specific"]].rename(
        columns={"uncleaned_r": "drop_r", "uncleaned_residual_r_specific": "drop_residual_r_specific"})

    run_dir = setup_run(OUT, {"runs": used, "reference": str(REF), "setting": "finetune",
                              "encoder": "lejepa_guided", "matched_source": matched_source,
                              "occlusion_matched_source": occ_source})
    verdict.to_csv(run_dir / "verdicts.csv", index=False)
    allp.to_csv(run_dir / "per_subject.csv", index=False)
    occ.to_csv(run_dir / "occlusion.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    order = ["ocular", "neural", "hybrid"]
    pw = wide["residual_r_matched"]
    for s in pw.index:
        ax[0].plot(range(3), pw.loc[s, order], color="0.7", lw=0.8, marker="o", ms=3)
    ax[0].plot(range(3), pw[order].mean(), color="k", lw=2, marker="s", label="mean over subjects")
    ax[0].set_xticks(range(3), order)
    ax[0].set_ylabel("residual_r_matched (test)")
    ax[0].set_title("Branches (fine-tune, lejepa_guided)")
    ax[0].legend(frameon=False)
    x = np.arange(len(GROUPS))
    for i, k in enumerate(("residual_r_matched", "r")):
        a = ax[i + 1]
        for j, (b, c) in enumerate((("neural", "C0"), ("hybrid", "C1"))):
            t = occ_s[occ_s.branch == b]
            mean = [t[t.group == g][f"drop_{k}"].mean() for g in GROUPS]
            a.bar(x + (j - 0.5) * 0.35, mean, 0.35, color=c, alpha=0.6, label=f"{b} (seed 1)")
            for gi, g in enumerate(GROUPS):
                v = t[t.group == g][f"drop_{k}"].to_numpy()
                a.scatter(np.full(v.size, gi + (j - 0.5) * 0.35), v, s=10, color=c, zorder=3)
        ref = [ref_occ.loc[g, f"drop_{k}"] if g in ref_occ.index and f"drop_{k}" in ref_occ else np.nan
               for g in GROUPS]
        a.scatter(x, ref, marker="D", color="k", s=30, zorder=4, label="diagnostics CNN-LSTM (ref)")
        a.axhline(0, color="0.4", lw=0.8)
        a.set_xticks(x, GROUPS, rotation=20)
        a.set_ylabel(f"drop in {k} (full - occluded)")
        a.set_title(f"Neural-branch occlusion: {k}")
        a.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(run_dir / "experiment2.png", dpi=150)
    plt.close(fig)

    show_occ = ["group", "branch", "n_subjects", "drop_residual_r_matched", "n_pos_residual_r_matched",
                "p_residual_r_matched", "drop_residual_r_specific", "n_pos_residual_r_specific",
                "p_residual_r_specific", "rel_drop_residual_r_specific", "drop_r", "n_pos_r", "p_r"]
    md = ("# Experiment 2: physiological branches (RQ2)\n\nSetting fine-tune, proposed head, proposed encoder "
          "(lejepa_guided); ocular-only has no encoder. Test scores per LOSO subject, mean over 3 decoder seeds, then "
          "mean over 10 subjects. `head_floor`: the warp head at initialization (cue phase prior); `template_cue` is "
          "scored on the same samples. `*_seed_sd`: SD over seeds of the subject-mean score.\n\n"
          + table.round(4).to_markdown() + "\n\n## Paired comparisons (Wilcoxon over subjects)\n\nresidual_r_specific is not compared with the "
          "rigid floors (window effect, SSL-27); their residual_r_matched is 0.\n\n"
          + pairs.round(4).to_markdown(index=False)
          + "\n\n## Trial-specific readings: residual_r_specific vs residual_r_matched (SSL-27)\n\nFinding only if "
          f"p < 0.05 on residual_r_matched (uncorrected). Matched from {matched_source or 'metrics.json'}.\n\n"
          + verdict.round(4).to_markdown(index=False)
          + "\n\n## Neural-branch group occlusion (seed 1, `exp1-occl`)\n\nDrop = full - occluded per test subject; "
          "the group is zeroed in the neural input only, ocular proxies intact. `p_*`: Wilcoxon of the drop against 0; "
          "`rel_drop_*`: mean drop / mean full score. " + occ_note + "\n\n"
          + occ_tab[show_occ].round(4).to_markdown(index=False)
          + "\n\n### Occlusion readings: residual_r_specific vs residual_r_matched\n\n"
          + (occ_verdict.round(4).to_markdown(index=False) if len(occ_verdict) else occ_note)
          + "\n\n### Hybrid - neural (difference of drops)\n\n"
          + occ_pairs.round(4).to_markdown(index=False) + "\n\n"
          + (occ_pair_verdict.round(4).to_markdown(index=False) if len(occ_pair_verdict) else "")
          + "\n\n## Reference: diagnostics stage (CNN-LSTM direct, uncleaned, test)\n\nPrefrontal-only control "
          "(input Fp1/Fp2 only) and all-channel model:\n\n" + ref_ctrl.round(4).to_markdown()
          + "\n\nGroup-occlusion drops of the all-channel model:\n\n" + ref_occ.round(4).to_markdown()
          + "\n\nFigure: `experiment2.png`.\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"branches": table.reset_index().to_dict(orient="records"), "pairs": pairs.to_dict(orient="records"),
         "occlusion": occ_tab.astype({"group": str}).to_dict(orient="records"),
         "occlusion_hybrid_vs_neural": occ_pairs.to_dict(orient="records"),
         "verdicts": verdict.to_dict(orient="records"), "occlusion_verdicts": occ_verdict.to_dict(orient="records"),
         "matched_source": matched_source, "occlusion_matched_source": occ_source,
         "reference_control": ref_ctrl.reset_index().to_dict(orient="records"),
         "reference_occlusion": ref_occ.reset_index().to_dict(orient="records")}, indent=2, default=float) + "\n",
        encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table.round(4).to_string())
    print(pairs.round(4).to_string(index=False))
    print(occ_tab[show_occ].round(4).to_string(index=False))
    print(occ_pairs.round(4).to_string(index=False))
    print(ref_ctrl.round(4).to_string())
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
