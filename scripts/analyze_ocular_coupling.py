"""Ocular coupling of the Experiment 1 predictions (RQ1 follow-up, decided 2026-11-16). ANALYSIS ONLY, no training.

Hypothesis: the guided encoders lose trial-specific information (residual_r_specific) because their ICA-cleaned target
removes ocular information, while jepa_unguided / mae keep it.

Per test subject and lift:
- EOG features: the VEOG and HEOG proxies (``eegtraj.preprocessing.ocular.proxy_rows`` from the CAR cache, z-scored per
  subject as in the decoder streams), mean over the lift's scored samples minus the mean over [LED-on - 1.5 s, LED-on).
- Trial-specific prediction: the prediction minus the mean prediction of the same subject at the same time since
  LED-on (0.01 s bins), averaged over the lift's scored samples, per axis. The same for the target (reference).
Coupling = Spearman rho across lifts between each axis of the trial-specific prediction and each EOG feature (3 x 2
pairs), per subject, then the mean |rho| over the 6 pairs and the mean over subjects. Seeds: rho per seed, then the
mean over seeds. The ``target`` row is the same coupling for the true trajectory.

Writes to a new output/proposed-model/diagnostics/ocular-coupling/<run-id>/: coupling.csv (per configuration, subject,
pair), summary.md, metrics.json.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\analyze_ocular_coupling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from eegtraj.data.lift_streams import build_subject_streams
from eegtraj.utils.run import setup_run

CACHE = "dataset/cache/98f5fa6e66"
DECODER = Path("output/proposed-model/decoder")
OUT = Path("output/proposed-model/diagnostics/ocular-coupling")
HISTORY, FS, BEFORE_LED = 180, 100.0, 150  # stream samples before LED-on (1.5 s)
EOG = ("VEOG", "HEOG")
AXES = ("x", "y", "z")


def subject_features(subject: str) -> dict:
    """EOG features per lift row, and time since LED-on (0.01 s bins) and lift row of every scored sample."""
    st = build_subject_streams(CACHE, subject)
    s = st.samples
    eog = {}
    for l in st.lifts:
        base = l.eog[:, HISTORY:HISTORY + BEFORE_LED].mean(axis=1)
        eog[l.trial_row] = l.eog[:, HISTORY + l.out_idx].mean(axis=1) - base
    t = (s.sample_idx - s.trials.offset_start.to_numpy()[s.trial_row] + 1) / s.fs_hz - s.trials.ev_LEDOn.to_numpy()[s.trial_row]
    return {"eog": eog, "bins": np.round(t * FS).astype(np.int64), "rows": s.trial_row, "sample_idx": s.sample_idx}


def trial_specific(y: np.ndarray, bins: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    """Per lift row: mean over its samples of y minus the subject's mean y in the same time bin."""
    b = bins - bins.min()
    cnt = np.bincount(b)
    mean = np.stack([np.bincount(b, weights=y[:, a]) / np.maximum(cnt, 1) for a in range(y.shape[1])], axis=1)
    resid = y - mean[b]
    return {int(r): resid[rows == r].mean(axis=0) for r in np.unique(rows)}


def coupling(spec: dict[int, np.ndarray], eog: dict[int, np.ndarray]) -> dict[str, float]:
    rows = sorted(spec)
    p = np.stack([spec[r] for r in rows])
    e = np.stack([eog[r] for r in rows])
    return {f"{AXES[a]}~{EOG[k]}": float(spearmanr(p[:, a], e[:, k]).statistic) for a in range(3) for k in range(2)}


def main() -> None:
    runs = sorted(p.parent for p in DECODER.glob("exp1__*/*/metrics.json") if not p.parent.name.endswith("_smoke"))
    if not runs:
        raise SystemExit("no Experiment 1 runs")
    subjects = list(json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))["per_subject"])
    run_dir = setup_run(OUT, {"runs": [str(r) for r in runs], "cache_dir": CACHE})
    feats = {s: subject_features(s) for s in subjects}
    recs = []
    for s in subjects:
        f = feats[s]
        z = np.load(runs[0] / "predictions.npz")
        if not np.array_equal(z[f"{s}__sample_idx"], f["sample_idx"]):
            raise RuntimeError(f"sample order differs for {s}")
        for pair, rho in coupling(trial_specific(z[f"{s}__y_true"], f["bins"], f["rows"]), f["eog"]).items():
            recs.append({"config": "target", "seed": 0, "subject": s, "pair": pair, "rho": rho})
    for r in runs:
        m = json.loads((r / "metrics.json").read_text(encoding="utf-8"))
        z = np.load(r / "predictions.npz")
        config = r.parent.name.split("__", 1)[1].removeprefix("proposed_")
        for s in subjects:
            f = feats[s]
            for pair, rho in coupling(trial_specific(z[f"{s}__y"], f["bins"], f["rows"]), f["eog"]).items():
                recs.append({"config": config, "seed": m["decoder_seed"], "subject": s, "pair": pair, "rho": rho})
    df = pd.DataFrame(recs)
    df["abs_rho"] = df.rho.abs()
    df.to_csv(run_dir / "coupling.csv", index=False)
    per_subj = df.groupby(["config", "subject", "seed"]).abs_rho.mean().groupby(["config", "subject"]).mean()
    summ = per_subj.groupby("config").agg(["mean", "std"]).rename(columns={"mean": "mean_abs_rho", "std": "sd_over_subjects"})
    pair_tab = df.groupby(["config", "pair"]).rho.mean().unstack()
    summ = summ.join(pair_tab).sort_values("mean_abs_rho", ascending=False)
    tests = []
    wide = per_subj.unstack("config")
    for c in wide.columns:
        for ref in ("hybrid_frozen_random0", "neural_frozen_random0"):
            if c != ref and ref in wide and c.split("_")[0] == ref.split("_")[0] and "frozen" in c:
                d = (wide[c] - wide[ref]).dropna()
                tests.append({"a": c, "b": ref, "mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()),
                              "n_subjects": int(d.size), "p_wilcoxon": float(wilcoxon(d).pvalue)})
    tests = pd.DataFrame(tests)
    md = ("# Ocular coupling of the Experiment 1 predictions\n\nMean |Spearman rho| across lifts between the trial-specific "
          "prediction (per axis) and the VEOG / HEOG change from the pre-cue baseline (6 pairs), per subject, mean over "
          "seeds and subjects. `target`: the same for the true trajectory. Paired tests: per-subject mean |rho| vs the "
          "random-init frozen control of the same branches (Wilcoxon).\n\n" + summ.round(3).to_markdown() +
          "\n\n## Versus random-init (frozen)\n\n" + tests.round(4).to_markdown(index=False) + "\n")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"summary": summ.reset_index().to_dict(orient="records"),
                                                     "tests": tests.to_dict(orient="records")}, indent=2, default=float)
                                          + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    print(tests.round(4).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
