"""Diagnostics numbers under residual_r_matched (design.md SSL-27). ANALYSIS ONLY, no training.

docs/diagnostics-results.md reports residual_r_specific (rrs). This script reads the newest
output/diagnostics/residual/<run-id>/residual.csv (scripts/compute_residual_metrics.py, which now adds
residual_r_matched for every saved predictions.npz) and recomputes the cited test-set numbers with both metrics:
per configuration (seeds averaged per subject, then mean over subjects, subjects above 0, Wilcoxon against 0), and the
paired comparisons the document draws (history, model, ICA cleaning, montages). A trial-specific difference is a finding
only if significant on residual_r_matched.

Not covered (no saved predictions): inner-validation scores, channel and group occlusion drops (scored in-run), and
the calibration fine-tuning runs; they stay rrs-only and are listed as a limitation.

The residual table uses its own permutation seed (20261106), so rrs here can differ slightly from in-run values.

Writes to a new output/diagnostics/summary/matched/<run-id>/: configs.csv, paired.csv, summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\compare_diagnostics_matched.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eegtraj.utils.rescored import verdicts
from eegtraj.utils.run import setup_run

TABLES = Path("output/diagnostics/residual")
OUT = Path("output/diagnostics/summary/matched")
D, B = "output/diagnostics", "output/data-baselines/baselines"
FC = f"{D}/final-checks"
ER = f"{FC}/electrode-reduction"
# label -> (config folder, doc value of rrs where the document cites one)
CONFIGS = {
    "Ridge 0.5 s": (f"{B}/ridge_lag", 0.045),
    "Kalman 0.5 s": (f"{B}/kalman", 0.118),
    "EEGNet 0.5 s": (f"{B}/eegnet", 0.065),
    "EEGNet 1 s": (f"{D}/history/eegnet/history_100", 0.073),
    "EEGNet 2 s": (f"{D}/history/eegnet/history_200", 0.090),
    "EEGNet 3 s": (f"{D}/history/eegnet/history_300", 0.094),
    "CNN-LSTM 0.5 s": (f"{B}/cnn_lstm", 0.093),
    "CNN-LSTM 1 s": (f"{D}/history/cnn_lstm/history_100", 0.118),
    "CNN-LSTM 2 s": (f"{D}/history/cnn_lstm/history_200", 0.134),
    "CNN-LSTM 3 s": (f"{D}/history/cnn_lstm/history_300", 0.135),
    "Onset template": (f"{D}/baselines/onset_template", 0.029),
    "Template (oracle timing)": (f"{B}/template", 0.195),
    "CNN-LSTM residual 2 s (uncleaned)": (f"{D}/pre-design/residual/cnn_lstm", 0.192),
    "EEGNet residual (uncleaned)": (f"{D}/pre-design/residual/eegnet", np.nan),
    "CNN-LSTM direct, ICA-cleaned": (f"{FC}/ocular/cnn_lstm_direct", 0.068),
    "CNN-LSTM residual, ICA-cleaned": (f"{FC}/ocular/cnn_lstm_residual", 0.080),
    "CNN-LSTM prefrontal only, ICA-cleaned": (f"{FC}/ocular/cnn_lstm_prefrontal", 0.015),
    "CNN-LSTM prefrontal only, uncleaned": (f"{D}/pre-design/screening/broadband__prefrontal", 0.062),
    "EEGNet prefrontal CAR 0.5 s": (f"{D}/artifact-controls/eegnet_prefrontal_car", np.nan),
    "EEGNet prefrontal CAR 3 s": (f"{D}/artifact-controls/eegnet_prefrontal_car/history_300", np.nan),
    "Euclidean alignment CNN-LSTM": (f"{D}/pre-design/euclidean-alignment/cnn_lstm_ea", np.nan),
    "Central 9 CAR direct": (f"{ER}/ica__central9__car__direct", 0.034),
    "Central 9 CAR residual": (f"{ER}/ica__central9__car__residual", 0.057),
    "Contralateral 6 CAR direct": (f"{ER}/ica__contralateral__car__direct", 0.032),
    "Contralateral 6 CAR residual": (f"{ER}/ica__contralateral__car__residual", 0.056),
    "Contralateral 6 mastoid direct": (f"{ER}/ica__contralateral__mastoid__direct", 0.020),
    "Contralateral 6 mastoid residual": (f"{ER}/ica__contralateral__mastoid__residual", 0.037),
    "Minimal 3 CAR direct": (f"{ER}/ica__minimal3__car__direct", 0.043),
    "Minimal 3 CAR residual": (f"{ER}/ica__minimal3__car__residual", 0.058),
    "Minimal 3 mastoid direct": (f"{ER}/ica__minimal3__mastoid__direct", 0.009),
    "Minimal 3 mastoid residual": (f"{ER}/ica__minimal3__mastoid__residual", 0.020),
    "Cz CAR direct": (f"{ER}/ica__cz__car__direct", 0.031),
    "Cz CAR residual": (f"{ER}/ica__cz__car__residual", 0.061),
    "Cz mastoid direct": (f"{ER}/ica__cz__mastoid__direct", 0.020),
    "Cz mastoid residual": (f"{ER}/ica__cz__mastoid__residual", 0.039),
    "Ring cleaned CAR direct": (f"{ER}/ica__headband_ring__car__direct", 0.041),
    "Ring cleaned CAR residual": (f"{ER}/ica__headband_ring__car__residual", 0.055),
    "Ring cleaned mastoid direct": (f"{ER}/ica__headband_ring__mastoid__direct", 0.020),
    "Ring cleaned mastoid residual": (f"{ER}/ica__headband_ring__mastoid__residual", 0.027),
    "Ring uncleaned CAR direct": (f"{ER}/raw__headband_ring__car__direct", 0.105),
    "Ring uncleaned CAR residual": (f"{ER}/raw__headband_ring__car__residual", 0.199),
    "Ring uncleaned mastoid direct": (f"{ER}/raw__headband_ring__mastoid__direct", 0.076),
    "Ring uncleaned mastoid residual": (f"{ER}/raw__headband_ring__mastoid__residual", 0.176),
}
for name in ("all", "contralateral", "ipsilateral", "no_prefrontal", "sensorimotor"):
    CONFIGS[f"Screening broadband {name}"] = (f"{D}/pre-design/screening/broadband__{name}", np.nan)
for name in ("all", "contralateral", "ipsilateral", "prefrontal"):
    CONFIGS[f"Screening mrcp {name}"] = (f"{D}/pre-design/screening/mrcp__{name}", np.nan)
CONFIGS["Screening mrcp+env all"] = (f"{D}/pre-design/screening/mrcp_env__all", np.nan)
for name in ("all", "contralateral", "ipsilateral"):
    CONFIGS[f"Screening env {name}"] = (f"{D}/pre-design/screening/env__{name}", np.nan)

PAIRS = [  # (a, b): a - b, as drawn in the document
    ("EEGNet 3 s", "EEGNet 0.5 s"), ("CNN-LSTM 2 s", "CNN-LSTM 0.5 s"), ("CNN-LSTM 3 s", "CNN-LSTM 2 s"),
    ("EEGNet 3 s", "EEGNet 2 s"), ("CNN-LSTM 0.5 s", "EEGNet 0.5 s"), ("CNN-LSTM 2 s", "EEGNet 3 s"),
    ("CNN-LSTM 2 s", "CNN-LSTM direct, ICA-cleaned"),
    ("CNN-LSTM residual 2 s (uncleaned)", "CNN-LSTM residual, ICA-cleaned"),
    ("CNN-LSTM prefrontal only, uncleaned", "CNN-LSTM prefrontal only, ICA-cleaned"),
    ("CNN-LSTM residual 2 s (uncleaned)", "CNN-LSTM 2 s"),
    ("Euclidean alignment CNN-LSTM", "CNN-LSTM 2 s"),
    ("Screening broadband all", "Screening broadband no_prefrontal"),
    ("Screening broadband contralateral", "Screening broadband ipsilateral"),
    ("Screening mrcp contralateral", "Screening mrcp ipsilateral"),
    ("Screening mrcp+env all", "Screening broadband all"),
] + [(k, "CNN-LSTM residual, ICA-cleaned") for k in CONFIGS if k.endswith("residual") and "mastoid" not in k
     and k.split()[0] in ("Central", "Contralateral", "Minimal", "Cz", "Ring")]
METRICS = ("residual_r_specific", "residual_r_matched")


def wil(d: pd.Series) -> float:
    d = d.dropna()
    return float(wilcoxon(d).pvalue) if len(d) >= 5 and (d != 0).any() else np.nan


def main() -> None:
    table_path = sorted(TABLES.glob("2*/residual.csv"))[-1]
    t = pd.read_csv(table_path)
    if "residual_r_matched" not in t:
        raise SystemExit(f"{table_path} has no residual_r_matched; run scripts/compute_residual_metrics.py --all")
    t["folder"] = t.run.str.rsplit("/", n=1).str[0]
    subj, rows, missing = {}, [], []
    for label, (folder, doc) in CONFIGS.items():
        x = t[t.folder == folder]
        if x.empty:
            missing.append(label)
            continue
        s = x.groupby("subject")[list(METRICS) + ["residual_r"]].mean()
        subj[label] = s
        rows.append({"config": label, "doc_rrs": doc, "n_runs": x.run.nunique(), "n_subjects": len(s),
                     "residual_r": s.residual_r.mean(),
                     **{f"{k}": s[k].mean() for k in METRICS},
                     **{f"{k}_sd": s[k].std(ddof=1) for k in METRICS},
                     **{f"n_above0_{k}": int((s[k] > 0).sum()) for k in METRICS},
                     **{f"p0_{k}": wil(s[k]) for k in METRICS}})
    configs = pd.DataFrame(rows)
    long = []
    for a, b in PAIRS:
        if a not in subj or b not in subj:
            continue
        for k in METRICS:
            d = (subj[a][k] - subj[b][k]).dropna()
            long.append({"a": a, "b": b, "metric": k, "mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()),
                         "n_subjects": int(d.size), "p_wilcoxon": wil(d)})
    long = pd.DataFrame(long)
    pairs = verdicts(long, ["a", "b"])
    above = configs.assign(metric="x")
    vs0 = verdicts(pd.concat([above.assign(metric=k, mean_diff=above[k], n_pos=above[f"n_above0_{k}"],
                                          p_wilcoxon=above[f"p0_{k}"]) for k in METRICS]), ["config"])

    run_dir = setup_run(OUT, {"table": table_path.as_posix(), "configs": {k: v[0] for k, v in CONFIGS.items()}})
    configs.to_csv(run_dir / "configs.csv", index=False)
    pairs.to_csv(run_dir / "paired.csv", index=False)
    show = ["config", "n_runs", "doc_rrs", "residual_r_specific", "residual_r_matched", "residual_r_matched_sd",
            "n_above0_residual_r_specific", "n_above0_residual_r_matched", "p0_residual_r_matched"]
    md = ("# Diagnostics numbers under residual_r_matched (SSL-27)\n\nSource: " + table_path.parent.as_posix()
          + ". Test set, LOSO; seeds averaged per subject, then mean over subjects. `doc_rrs`: the value cited in "
          "docs/diagnostics-results.md; `residual_r_specific`: recomputed from the saved predictions with the residual "
          "table's permutation seed. `p0_*`: Wilcoxon against 0.\n\n" + configs[show].round(4).to_markdown(index=False)
          + "\n\n## Above 0: readings\n\n" + vs0[vs0.changed].round(4).to_markdown(index=False)
          + "\n\n## Paired comparisons drawn in the document\n\nFinding only if p < 0.05 on residual_r_matched.\n\n"
          + pairs.round(4).to_markdown(index=False)
          + f"\n\nMissing configurations (no saved predictions in the table): {', '.join(missing) or 'none'}.\n\n"
          "Not covered (no saved predictions): inner-validation scores, channel and group occlusion drops, calibration "
          "fine-tuning runs.\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"configs": configs.to_dict(orient="records"),
                                                     "paired": pairs.to_dict(orient="records"),
                                                     "missing": missing}, indent=2, default=float) + "\n",
                                          encoding="utf-8")
    pd.set_option("display.width", 250)
    print(configs[show].round(4).to_string(index=False))
    print(pairs.round(4).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
