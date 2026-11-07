"""Extra scoring inside the fold loop: inner-validation metrics and channel occlusion.

Both use the cue-locked template fitted on the fold's training subjects, as in ``eegtraj.diagnostics.residual``.
Permutations of ``residual_r_other_trial`` depend only on ``seed`` and the subject, so every prediction of a
subject (for example every occluded channel) is scored against the same trial pairs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext
from eegtraj.baselines.template import TemplateCueBaseline, time_since_event
from eegtraj.data.windows import SubjectSamples
from eegtraj.metrics import aggregate_subject, per_trial_metrics, residual_r, residual_r_other_trial


def fit_cue_template(train: list[SubjectSamples], ctx: FitContext) -> TemplateCueBaseline:
    template = TemplateCueBaseline()
    template.fit(train, [], ctx)
    return template


def subject_scores(s: SubjectSamples, mask: np.ndarray, y_pred: np.ndarray, template: TemplateCueBaseline, ctx: FitContext,
                   seed: int, n_permutations: int) -> dict[str, float]:
    """r, R2, NRMSE, and the residual metrics of one subject's predictions at ``s.sample_idx[mask]``."""
    y_true, rows = s.target[:, s.sample_idx[mask]].T, s.trial_row[mask]
    scores = aggregate_subject(per_trial_metrics(y_true, y_pred, rows))
    cue = template.predict(s, mask, ctx)
    bins = np.round(time_since_event(s, "ev_LEDOn", mask) * s.fs_hz).astype(np.int64)
    res = residual_r(y_true, y_pred, cue, rows)["mean"]
    rng = np.random.default_rng([seed, int(s.subject[1:])])
    other = residual_r_other_trial(y_true, y_pred, cue, rows, bins, rng, n_permutations)["mean"]
    return {"r": scores["r"]["mean"], "r2": scores["r2"]["mean"], "nrmse": scores["nrmse"]["mean"],
            "residual_r": res, "residual_r_other_trial": other, "residual_r_specific": res - other}


def score_inner_validation(model: Baseline, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext,
                           seed: int, n_permutations: int,
                           template: TemplateCueBaseline | None = None) -> dict[str, dict[str, float]]:
    """Scores of a fitted model on the fold's inner-validation subjects.

    Early stopping used these subjects, so the values are optimistic in the same way for every configuration.
    They never use the fold's test subject.
    """
    template = template or fit_cue_template(train, ctx)
    return {s.subject: subject_scores(s, ctx.val_masks[s.subject], model.predict(s, ctx.val_masks[s.subject], ctx), template,
                                      ctx, seed, n_permutations) for s in val}


def occluded(s: SubjectSamples, rows: list[int], ctx: FitContext) -> SubjectSamples:
    """Copy of ``s`` with the given EEG rows set to their normalizer mean, so they are zero after z-scoring."""
    eeg = np.array(s.eeg, dtype=np.float32, copy=True)
    eeg[rows] = ctx.eeg_norm[s.subject].mean[rows].astype(np.float32)
    return replace(s, eeg=eeg)


def channel_occlusion(model: Baseline, s: SubjectSamples, mask: np.ndarray, ctx: FitContext, template: TemplateCueBaseline,
                      names: list[str], groups: dict[str, list[str]], seed: int, n_permutations: int) -> list[dict]:
    """Scores with nothing occluded, with each input row occluded, and with each group of channels occluded.

    Args:
        names: input row names, as from ``eegtraj.data.windows.input_channel_names``. A group lists channel names
            and occludes every row of those channels (all bands).
    """
    def bare(name: str) -> str:
        return name.split(":")[-1]

    targets = [("none", "none", [])] + [("channel", n, [i]) for i, n in enumerate(names)]
    for g, chans in groups.items():
        rows = [i for i, n in enumerate(names) if bare(n) in chans]
        if rows:
            targets.append(("group", g, rows))
    out = []
    for kind, name, rows in targets:
        x = occluded(s, rows, ctx) if rows else s
        scores = subject_scores(x, mask, model.predict(x, mask, ctx), template, ctx, seed, n_permutations)
        out.append({"subject": s.subject, "kind": kind, "occluded": name, "n_rows": len(rows), **scores})
    return out
