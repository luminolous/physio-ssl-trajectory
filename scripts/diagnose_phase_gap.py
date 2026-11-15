"""Gap between the warp decoders and the template_cue floor on the screening folds (P3, P6). ANALYSIS ONLY, no training.

Per screening fold, with the fold's training-only landmark templates and template_cue:

1. ``oracle_phi``: phi from the test lift's own landmarks (time_to_phase with the fold fractions), A = 1, Delta = 0,
   mixing = the lift's true weight condition (oracle; also with uniform mixing). Should approach the oracle-warp
   ceiling. Far below it means a bug in the templates, the landmark registration or the re-centering.
2. ``cue_phi``: phi(t) = clip((t - t_LED - median_RT) / median_duration, 0, 1), medians over the fold's training
   lifts, A = 1, Delta = 0, uniform mixing (no EEG, no test-lift information). Should be close to template_cue.
   Also the variant with the runner's s0 (median hand stop since LED minus median RT).
3. For every decoder run given with ``--run`` (screening folds, predictions.npz): its scores, and for warp heads the
   predicted phi against the landmark phi (timing of the 0.05 / 0.5 / 0.95 crossings vs the true onset, mid-phase
   and hand stop, phi at onset and at hand stop, across-trial correlation of crossing times) plus phi substitution:
   the run's own A, mixing and Delta (recovered as y - A T_mix(phi)) with phi replaced by the oracle phi or the cue
   phi. Plots of predicted vs landmark phi for a few lifts per subject.

The oracle rows use the test lift's kinematic events and condition label: analysis only, never a model input.

Writes to a new output/proposed-model/diagnostics/phase-gap/<run-id>/: table.md, table.csv, phi_timing.csv,
metrics.json, phi_<run>_<subject>.png.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\diagnose_phase_gap.py --run <decoder-sweep run dir> [--run ...]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from eegtraj.baselines.base import FitContext  # noqa: E402
from eegtraj.baselines.scoring import fit_cue_template, subject_scores  # noqa: E402
from eegtraj.data.splits import load_split_config, make_folds  # noqa: E402
from eegtraj.data.windows import load_subject_samples  # noqa: E402
from eegtraj.models.landmark_template import fold_templates, lift_landmarks, ordered, time_to_phase  # noqa: E402
from eegtraj.utils.run import setup_run  # noqa: E402

logger = logging.getLogger("diagnose_phase_gap")
OUT = Path("output/proposed-model/diagnostics/phase-gap")
CACHE = "dataset/cache/98f5fa6e66"
SPLIT = "configs/splits/default.yaml"
CONDITIONS = (165, 330, 660)
SCORE_KEYS = ("r", "r2", "nrmse", "residual_r_specific")
CROSS = (0.05, 0.5, 0.95)


def load(subject: str):
    # same view as eegtraj.data.lift_streams (pre-lift cache, scored samples = protocol segment)
    return load_subject_samples(CACHE, subject, "broadband", "car", "all", "position", history=300, pre_lift=True)


def sample_times(s) -> tuple[np.ndarray, np.ndarray]:
    """Window seconds of every scored sample and its trial row."""
    rows = s.trial_row
    t = (s.sample_idx - s.trials.offset_start.to_numpy()[rows] + 1) / s.fs_hz
    return t, rows


def landmark_phi(s, fractions: np.ndarray) -> tuple[np.ndarray, int]:
    """Oracle phi of every scored sample from its lift's own landmarks. Lifts with non-increasing landmarks fall back
    to linear onset -> hand stop. Returns (phi, n_fallback_lifts)."""
    t, rows = sample_times(s)
    marks = lift_landmarks(s.trials)
    ok = ordered(marks)
    phi = np.empty(t.size)
    for row in np.unique(rows):
        sel = rows == row
        if ok[row]:
            phi[sel] = time_to_phase(t[sel], marks[row], fractions)
        else:
            phi[sel] = np.clip((t[sel] - marks[row, 0]) / (marks[row, -1] - marks[row, 0]), 0, 1)
    return phi, int((~ok[np.unique(rows)]).sum())


def t_mix(tmpl, phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sum_c w_c T_c(phi), (n, 3). w (n, C)."""
    return sum(w[:, i:i + 1] * tmpl(phi, c) for i, c in enumerate(tmpl.conditions))


def crossing(t: np.ndarray, phi: np.ndarray, level: float) -> float:
    """First time phi >= level (linear interpolation between samples); NaN if never."""
    k = np.flatnonzero(phi >= level)
    if k.size == 0:
        return np.nan
    i = k[0]
    if i == 0 or phi[i] == phi[i - 1]:
        return float(t[i])
    return float(t[i - 1] + (level - phi[i - 1]) / (phi[i] - phi[i - 1]) * (t[i] - t[i - 1]))


def phi_timing(s, phi_pred: np.ndarray, phi_true: np.ndarray, t: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Per lift: crossing times of predicted and landmark phi (s since LED-on), phi_pred at onset and hand stop."""
    marks = lift_landmarks(s.trials)
    led = s.trials.ev_LEDOn.to_numpy()
    out = []
    for row in np.unique(rows):
        sel = rows == row
        tt = t[sel] - led[row]
        rec = {"trial_row": int(row), "onset_s": marks[row, 0] - led[row], "hand_stop_s": marks[row, -1] - led[row]}
        for lv in CROSS:
            rec[f"pred_{lv}"] = crossing(tt, phi_pred[sel], lv)
            rec[f"true_{lv}"] = crossing(tt, phi_true[sel], lv)
        rec["phi_pred_at_onset"] = float(np.interp(rec["onset_s"], tt, phi_pred[sel]))
        rec["phi_pred_at_stop"] = float(phi_pred[sel][-1])
        rec["phi_pred_pre_onset_max"] = float(phi_pred[sel][tt < rec["onset_s"]].max()) if (tt < rec["onset_s"]).any() else np.nan
        rec["phi_mae"] = float(np.abs(phi_pred[sel] - phi_true[sel]).mean())
        out.append(rec)
    return pd.DataFrame(out)


def timing_summary(df: pd.DataFrame) -> dict:
    res = {"phi_mae": float(df.phi_mae.mean()), "phi_pred_at_onset": float(df.phi_pred_at_onset.median()),
           "phi_pred_pre_onset_max": float(df.phi_pred_pre_onset_max.median()),
           "phi_pred_at_stop": float(df.phi_pred_at_stop.median()), "frac_reach_0.95": float(df["pred_0.95"].notna().mean())}
    for lv in CROSS:
        d = df[f"pred_{lv}"] - df[f"true_{lv}"]
        ok = d.notna()
        res[f"lag_{lv}_median_s"] = float(d[ok].median()) if ok.any() else np.nan
        res[f"lag_{lv}_iqr_s"] = float(d[ok].quantile(0.75) - d[ok].quantile(0.25)) if ok.any() else np.nan
        both = df[[f"pred_{lv}", f"true_{lv}"]].dropna()
        res[f"trial_corr_{lv}"] = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]) if len(both) > 2 else np.nan
    return res


def plot_lifts(path: Path, s, phi_pred, phi_true, phi_cue, t, rows, title: str, n: int = 6, seed: int = 0) -> None:
    marks = lift_landmarks(s.trials)
    led = s.trials.ev_LEDOn.to_numpy()
    pick = np.random.default_rng(seed).choice(np.unique(rows), size=n, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, row in zip(axes.ravel(), sorted(pick)):
        sel = rows == row
        tt = t[sel] - led[row]
        ax.plot(tt, phi_true[sel], "k", lw=2, label="landmark phi (oracle)")
        ax.plot(tt, phi_pred[sel], "C0", lw=1.6, label="predicted phi")
        ax.plot(tt, phi_cue[sel], "C1--", lw=1.2, label="cue phi")
        for k, m in enumerate(marks[row]):
            ax.axvline(m - led[row], color="0.7", lw=0.8, ls=":" if 0 < k < 4 else "-")
        ax.set_title(f"lift row {row}, onset {marks[row, 0] - led[row]:.2f} s")
        ax.set_xlabel("s since LED-on")
    axes[0, 0].set_ylabel("phi")
    axes[1, 0].set_ylabel("phi")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", default=[], type=Path, help="decoder run dir (screening folds)")
    args = parser.parse_args()
    runs = []
    for r in args.run:
        cfg = yaml.safe_load((r / "config.yaml").read_text(encoding="utf-8"))
        if cfg.get("folds") != "screening":
            raise SystemExit(f"{r}: not a screening-fold run")
        runs.append((r, cfg))
    run_dir = setup_run(OUT, {"runs": [str(r) for r in args.run], "cache_dir": CACHE, "split_config": SPLIT})

    folds = make_folds(load_split_config(SPLIT), "screening")
    preds = {r: np.load(r / "predictions.npz") for r, _ in runs}
    rows_out, timing_rows, info = [], [], {}
    cache: dict = {}
    for fold in folds:
        names = sorted(set(fold.train_subjects) | set(fold.val_subjects) | set(fold.test_subjects), key=lambda x: int(x[1:]))
        for n in names:
            if n not in cache:
                cache[n] = load(n)
        subj = {n: cache[n] for n in names}
        ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=1,
                         train_masks={n: np.ones(subj[n].sample_idx.size, bool) for n in fold.train_subjects},
                         val_masks={}, logger=logger)
        cue_tmpl = fit_cue_template([subj[n] for n in fold.train_subjects], ctx)
        tmpl = fold_templates([subj[n] for n in names], fold, conditions=CONDITIONS, n_grid=201)
        marks = np.concatenate([lift_landmarks(subj[n].trials) for n in fold.train_subjects])
        led = np.concatenate([subj[n].trials.ev_LEDOn.to_numpy() for n in fold.train_subjects])
        rt_med = float(np.median(marks[:, 0] - led))
        dur_med = float(np.median(marks[:, -1] - marks[:, 0]))
        s0_runner = float(tmpl.landmarks_since_cue_s[-1] - tmpl.landmarks_since_cue_s[0])
        for test in fold.test_subjects:
            s = subj[test]
            mask = np.ones(s.sample_idx.size, bool)
            t, rows = sample_times(s)
            t_led = t - s.trials.ev_LEDOn.to_numpy()[rows]
            phi_true, n_fallback = landmark_phi(s, tmpl.fractions)
            cond = s.trials.weight_g.to_numpy()[rows]
            w_true = np.stack([(cond == c).astype(float) for c in tmpl.conditions], axis=1)
            w_uni = np.full((t.size, len(tmpl.conditions)), 1 / len(tmpl.conditions))
            phi_cue = np.clip((t_led - rt_med) / dur_med, 0, 1)
            phi_cue_s0 = np.clip((t_led - rt_med) / s0_runner, 0, 1)
            info[test] = {"fold": fold.name, "median_rt_s": rt_med, "median_duration_s": dur_med, "runner_s0_s": s0_runner,
                          "fractions": tmpl.fractions.tolist(), "n_fallback_lifts": n_fallback,
                          "lifts_without_condition": int((w_true.sum(1) == 0).sum())}

            def score(label: str, y: np.ndarray, kind: str) -> None:
                sc = subject_scores(s, mask, y, cue_tmpl, ctx, 0, 5)
                rows_out.append({"subject": test, "row": label, "kind": kind, **{k: sc[k] for k in SCORE_KEYS}})

            score("template_cue (floor)", cue_tmpl.predict(s, mask, ctx), "floor")
            score("1. oracle phi, true condition, A=1, Delta=0", t_mix(tmpl, phi_true, w_true), "oracle")
            score("1b. oracle phi, uniform mixing", t_mix(tmpl, phi_true, w_uni), "oracle")
            score("2. cue phi (median RT, median duration), uniform", t_mix(tmpl, phi_cue, w_uni), "no EEG")
            score("2b. cue phi with runner s0, uniform", t_mix(tmpl, phi_cue_s0, w_uni), "no EEG")

            for r, cfg in runs:
                z = preds[r]
                if not np.array_equal(z[f"{test}__sample_idx"], s.sample_idx):
                    raise RuntimeError(f"{r}: sample order differs from the cache for {test}")
                head = cfg["model"]["head"]
                label = f"{head} ({r.name.split('_seed')[0]})"
                y = z[f"{test}__y"]
                score(f"4. {label}", y, "decoder")
                if f"{test}__phi" not in z.files:
                    continue
                phi, w, amp = z[f"{test}__phi"].astype(np.float64), z[f"{test}__weights"], z[f"{test}__amp"]
                delta = y - amp * t_mix(tmpl, phi, w)
                score(f"4. {head}: phi -> oracle phi (own A, mix, Delta)", amp * t_mix(tmpl, phi_true, w) + delta, "oracle")
                score(f"4. {head}: phi -> cue phi (own A, mix, Delta)", amp * t_mix(tmpl, phi_cue, w) + delta, "no EEG")
                score(f"4. {head}: own phi, A=1, uniform, Delta=0", t_mix(tmpl, phi, w_uni), "decoder")
                df = phi_timing(s, phi, phi_true, t, rows)
                df.insert(0, "subject", test)
                df.insert(1, "head", head)
                timing_rows.append(df)
                timing_rows.append(phi_timing(s, phi_cue, phi_true, t, rows).assign(subject=test, head="cue_phi"))
                plot_lifts(run_dir / f"phi_{head}_{test}.png", s, phi, phi_true, phi_cue, t, rows,
                           f"{test}: {head} phi vs landmark phi ({r.name})")

    table = pd.DataFrame(rows_out)
    wide = table.pivot_table(index=["row", "kind"], columns="subject", values=list(SCORE_KEYS), sort=False)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for k in SCORE_KEYS:
        wide[k] = wide[[c for c in wide.columns if c.startswith(f"{k}_P")]].mean(axis=1)
    cols = ["row", "kind", "r", "r_P3", "r_P6", "residual_r_specific", "nrmse", "r2"]
    table.to_csv(run_dir / "table.csv", index=False)
    md = "# Phase gap diagnostic (screening folds P3, P6)\n\n" + wide[cols].round(4).to_markdown(index=False) + "\n"
    timing_md = ""
    if timing_rows:
        timing = pd.concat(timing_rows, ignore_index=True)
        timing.to_csv(run_dir / "phi_timing.csv", index=False)
        summ = pd.DataFrame([{"head": h, "subject": sj, **timing_summary(g)} for (h, sj), g in timing.groupby(["head", "subject"])])
        timing_md = ("\n## Phase timing (s; lag = predicted - landmark crossing; trial_corr = across-lift correlation "
                     "of crossing times)\n\n" + summ.round(3).to_markdown(index=False) + "\n")
        info["timing"] = summ.to_dict(orient="records")
    (run_dir / "table.md").write_text(md + timing_md + "\n## Fold info\n\n```\n" + json.dumps(info, indent=1, default=float)
                                      + "\n```\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"rows": wide.to_dict(orient="records"), "info": info}, indent=2,
                                                     default=float) + "\n", encoding="utf-8")
    print(wide[cols].round(4).to_string(index=False))
    if timing_md:
        print(timing_md)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "fractions"} for k, v in info.items() if k != "timing"},
                     default=float))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
