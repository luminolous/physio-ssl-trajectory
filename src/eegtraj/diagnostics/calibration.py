"""Calibration within subject: a few labeled lifts of the LOSO test subject (final pre-design checks).

Split inside the test subject (within-subject diagnostic, D6): calibration lifts are drawn from S1-S5, S6 is the
validation series (early stopping), S7-S9 are the test series. The S7-S9 targets never enter a template, a delta
scale, or a training step. The EEG normalizer stays per subject over all lift samples (D9, label-free, transductive).

Conditions, all CNN-LSTM in residual form over the **subject cue template** fitted on the k calibration lifts:
- ``template``: that template alone, the floor of this experiment;
- ``no_calibration``: the LOSO model with its own cross-subject template, unchanged (scored against the subject
  template, so residual metrics stay comparable);
- ``finetune``: the LOSO model fine-tuned on delta = target - subject template; the output scale stays the LOSO one;
- ``scratch``: the same architecture trained from random weights on the k lifts; delta scale from the k lifts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.deep import ARCHITECTURES, residual_subject
from eegtraj.baselines.template import TemplateCueBaseline
from eegtraj.data.windows import Standardizer, SubjectSamples, fit_target_normalizer
from eegtraj.models.trainer import GpuWindows, TrainConfig, predict_windows, train_model

ALL = "all"
CONDITION_CODES = {"finetune": 1, "scratch": 2}


@dataclass(frozen=True)
class Series:
    calibration: tuple[int, ...] = (1, 2, 3, 4, 5)
    validation: tuple[int, ...] = (6,)
    test: tuple[int, ...] = (7, 8, 9)


def series_mask(s: SubjectSamples, series: tuple[int, ...]) -> np.ndarray:
    """Boolean over ``s.sample_idx``: samples whose lift belongs to one of ``series``."""
    return s.trials.series.isin(series).to_numpy()[s.trial_row]


def rows_mask(s: SubjectSamples, rows: np.ndarray) -> np.ndarray:
    """Boolean over ``s.sample_idx``: samples of the given trial rows."""
    keep = np.zeros(len(s.trials), dtype=bool)
    keep[rows] = True
    return keep[s.trial_row]


def k_code(k: int | str) -> int:
    return 0 if k == ALL else int(k)


def calibration_rows(s: SubjectSamples, k: int | str, draw: int, seed: int, series: Series = Series()) -> np.ndarray:
    """Sorted trial rows of the k calibration lifts, drawn from the calibration series without replacement.

    ``k = "all"`` takes every calibration lift. The draw depends only on (seed, subject, k, draw).
    """
    pool = np.flatnonzero(s.trials.series.isin(series.calibration).to_numpy())
    pool = pool[np.isin(pool, np.unique(s.trial_row))]  # lifts that have scored samples
    if k == ALL:
        return pool
    if int(k) > len(pool):
        raise ValueError(f"{s.subject}: k = {k} but only {len(pool)} calibration lifts")
    rng = np.random.default_rng([seed, int(s.subject[1:]), k_code(k), draw])
    return np.sort(rng.choice(pool, size=int(k), replace=False))


def fit_seed(seed: int, subject: str, k: int | str, draw: int, condition: str) -> int:
    return int(np.random.SeedSequence([seed, int(subject[1:]), k_code(k), draw,
                                       CONDITION_CODES[condition]]).generate_state(1)[0] % (2 ** 31))


def context(s: SubjectSamples, eeg_norm: Standardizer, train_mask: np.ndarray, val_mask: np.ndarray, history: int,
            seed: int, log: logging.Logger) -> FitContext:
    return FitContext(eeg_norm={s.subject: eeg_norm}, target_norm=None, history=history, lag_step=1, seed=seed,
                      train_masks={s.subject: train_mask}, val_masks={s.subject: val_mask}, logger=log,
                      fold_name=s.subject)


def subject_template(s: SubjectSamples, cal_mask: np.ndarray, ctx: FitContext) -> TemplateCueBaseline:
    """Cue-locked template from the calibration samples of this subject only."""
    template = TemplateCueBaseline()
    template.fit([s], [], replace(ctx, train_masks={s.subject: cal_mask}))
    return template


def template_from_saved(saved: dict) -> TemplateCueBaseline:
    t = TemplateCueBaseline()
    t.template, t.bin0, t.fs_hz, t.counts = (np.asarray(saved["template"]), int(saved["bin0"]), float(saved["fs_hz"]),
                                             np.asarray(saved["counts"]))
    return t


def load_pretrained(path: Path, n_chans: int, history: int, device: torch.device
                    ) -> tuple[torch.nn.Module, Standardizer, TemplateCueBaseline, dict]:
    """LOSO model saved by ``DeepBaseline.save`` in residual form: weights, delta scale, cross-subject T_cue, and
    the architecture spec (``architecture``, ``model_kwargs``)."""
    saved = torch.load(path, map_location=device, weights_only=False)  # our own files
    if not saved.get("residual") or "output_norm" not in saved:
        raise ValueError(f"{path} is not a residual model saved with its output normalizer and template")
    model = ARCHITECTURES[saved["architecture"]](n_chans=n_chans, n_times=history, **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    norm = Standardizer(np.asarray(saved["output_norm"]["mean"]), np.asarray(saved["output_norm"]["std"]))
    spec = {"architecture": saved["architecture"], "model_kwargs": saved["model_kwargs"]}
    return model.to(device), norm, template_from_saved(saved["template"]), spec


def delta_scale(s: SubjectSamples, template: TemplateCueBaseline, ctx: FitContext) -> Standardizer:
    """SD of delta over the calibration samples, mean fixed at 0 (delta_hat = 0 stays the template)."""
    fitted = fit_target_normalizer([residual_subject(s, template, ctx)], (s.subject,), train_masks=ctx.train_masks)
    return Standardizer(np.zeros_like(fitted.mean), fitted.std)


def train_residual(model: torch.nn.Module, s: SubjectSamples, template: TemplateCueBaseline, output_norm: Standardizer,
                   ctx: FitContext, cfg: TrainConfig, device: torch.device) -> tuple[torch.nn.Module, list[dict]]:
    """Train ``model`` on delta over ``template``: training samples ``ctx.train_masks``, early stopping on
    ``ctx.val_masks``. Only those samples are read."""
    d = residual_subject(s, template, ctx)
    tr = GpuWindows([d], ctx.train_masks, ctx.eeg_norm, output_norm, ctx.history, device)
    va = GpuWindows([d], ctx.val_masks, ctx.eeg_norm, output_norm, ctx.history, device)
    torch.manual_seed(ctx.seed)
    return train_model(model, tr, va, cfg, ctx.seed, ctx.logger)


def predict_residual(model: torch.nn.Module, s: SubjectSamples, mask: np.ndarray, template: TemplateCueBaseline,
                     output_norm: Standardizer, ctx: FitContext, cfg: TrainConfig, device: torch.device) -> np.ndarray:
    """template + delta_hat at ``s.sample_idx[mask]``, (n, 3) in target units."""
    w = GpuWindows([s], {s.subject: mask}, ctx.eeg_norm, output_norm, ctx.history, device)
    z = predict_windows(model, w, cfg)
    return output_norm.inverse(z.T.astype(np.float32)).T + template.predict(s, mask, ctx)


def scratch_model(architecture: str, n_chans: int, history: int, model_kwargs: dict, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return ARCHITECTURES[architecture](n_chans=n_chans, n_times=history, **model_kwargs)
