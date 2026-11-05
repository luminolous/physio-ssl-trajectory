"""Residual metrics for saved predictions: how well a model tracks what the mean template leaves over.

``residual_r`` uses the cue-locked template and ``residual_r_oracle`` the template aligned on the test
trial's kinematic onset (analysis only). Both templates come from the training subjects of the fold
that tested the subject, with the same masks and code as the ``template_cue`` and ``template`` baselines.

``residual_r`` also picks up what is shared across a subject's trials, such as a shape difference between
that subject and the training subjects. ``residual_r_other_trial`` measures that part by pairing trials at
random, and ``residual_r_specific`` is the difference. Permutations depend only on ``seed`` and the
subject, so every run of a subject uses the same trial pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.runner import trial_filter
from eegtraj.baselines.template import TemplateBaseline, TemplateCueBaseline, time_since_event
from eegtraj.data.splits import load_split_config, make_folds
from eegtraj.data.windows import SubjectSamples, load_subject_samples, sample_masks
from eegtraj.metrics import residual_r, residual_r_other_trial


@dataclass
class SubjectTemplates:
    """One test subject's templates at its scored samples, in the order the runner saves predictions."""

    y_true: np.ndarray  # (n, 3)
    trial_row: np.ndarray  # (n,)
    cue: np.ndarray  # (n, 3)
    oracle: np.ndarray  # (n, 3)
    time_bin: np.ndarray  # (n,) integer samples since LED-on


def fold_templates(subjects: list[SubjectSamples], fold) -> dict[str, SubjectTemplates]:
    """Fit both templates on a fold's training subjects and evaluate them on its test subjects."""
    masks = sample_masks(fold, subjects)
    ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=0, train_masks=masks["train"],
                     val_masks=masks["val"], fold_name=fold.name)
    train = [s for s in subjects if s.subject in fold.train_subjects]
    cue, oracle = TemplateCueBaseline(), TemplateBaseline()
    cue.fit(train, [], ctx)
    oracle.fit(train, [], ctx)
    out = {}
    for s in subjects:
        if s.subject in fold.test_subjects:
            m = masks["test"][s.subject]
            bins = np.round(time_since_event(s, "ev_LEDOn", m) * s.fs_hz).astype(np.int64)
            out[s.subject] = SubjectTemplates(s.target[:, s.sample_idx[m]].T, s.trial_row[m], cue.predict(s, m, ctx),
                                              oracle.predict(s, m, ctx), bins)
    return out


def load_templates(cache_dir: str | Path, splits_config: str | Path, mode: str, target: str,
                   exclude_flags: list[str]) -> dict[str, SubjectTemplates]:
    """Templates for every test subject of a split. They depend on the targets only, not on EEG or history."""
    folds = make_folds(load_split_config(splits_config), mode)
    names = sorted({s for f in folds for s in (*f.train_subjects, *f.val_subjects, *f.test_subjects)},
                   key=lambda s: int(s[1:]))
    subjects = [trial_filter(load_subject_samples(cache_dir, s, "broadband", "car", "all", target, 1), exclude_flags, None)
                for s in names]
    out: dict[str, SubjectTemplates] = {}
    for fold in folds:
        out |= fold_templates(subjects, fold)
    return out


def score_predictions(predictions, templates: dict[str, SubjectTemplates], seed: int = 0,
                      n_permutations: int = 5) -> list[dict]:
    """One row per subject. Raises if the saved samples are not the template's samples."""
    subjects = sorted({k.split("__")[0] for k in predictions.keys()}, key=lambda s: int(s[1:]))
    rows = []
    for s in subjects:
        t = templates[s]
        rows_saved, y_true, y_pred = predictions[f"{s}__trial_row"], predictions[f"{s}__y_true"], predictions[f"{s}__y_pred"]
        if not np.array_equal(rows_saved, t.trial_row) or not np.array_equal(y_true, t.y_true):
            raise ValueError(f"{s}: saved samples differ from the template's samples")
        row = {"subject": s, "n_samples": len(y_true)}
        for name, template in (("residual_r", t.cue), ("residual_r_oracle", t.oracle)):
            res = residual_r(y_true, y_pred, template, rows_saved)
            row |= {name: res["mean"], f"{name}_n_nan": res["n_nan"]}
            row |= {f"{name}_{a}": v for a, v in res["per_axis"].items()}
        rng = np.random.default_rng([seed, int(s[1:])])
        other = residual_r_other_trial(y_true, y_pred, t.cue, rows_saved, t.time_bin, rng, n_permutations)
        row |= {"residual_r_other_trial": other["mean"], "residual_r_other_trial_n_nan": other["n_nan"],
                "residual_r_specific": row["residual_r"] - other["mean"]}
        rows.append(row)
    return rows
