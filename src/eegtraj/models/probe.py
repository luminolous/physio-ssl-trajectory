"""Linear-probe health check for pretrained encoders (design.md, "Pretraining health").

Labeled 2 s windows end at scored target samples (D2 segment, re-centered wrist position D1). The causal history
before the segment start comes from the pre-lift cache. EEG is z-scored per subject over its lift samples (D9,
the same statistics as ``eegtraj.data.unlabeled.UnlabeledWindows``). The probe input is the encoder's last token
(the streaming output for the patch that ends at the target sample), so it uses no EEG after the target.

A ridge probe is fitted on one group of subjects and scored on another with r and ``residual_r_specific``
(``eegtraj.baselines.scoring.subject_scores``; the ``template_cue`` floor is fitted on the probe's training subjects).
Only screening / inner-validation subjects should be used here, never a fold's test subject.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from eegtraj.baselines.base import FitContext
from eegtraj.baselines.scoring import fit_cue_template, subject_scores
from eegtraj.baselines.template import time_since_event
from eegtraj.data.windows import SubjectSamples, load_subject_samples
from eegtraj.metrics.trajectory import MIN_SHARED_BINS
from eegtraj.models.pretraining import effective_rank, embedding_std, ridge_fit, ridge_predict

logger = logging.getLogger(__name__)


@dataclass
class ProbeSubject:
    """One subject's probe windows: ``windows`` (n, 32, window) float32 z-scored; ``mask`` selects the probed target
    samples inside ``samples.sample_idx`` (target samples whose time since LED-on is a multiple of ``stride``
    samples, the same grid for every lift)."""

    samples: SubjectSamples
    mask: np.ndarray
    windows: np.ndarray
    stride: int = 1
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def subject(self) -> str:
        return self.samples.subject

    @property
    def target(self) -> np.ndarray:
        return self.samples.target[:, self.samples.sample_idx[self.mask]].T


def load_probe_subject(cache_dir: str, subject: str, window: int = 200, stride: int = 10) -> ProbeSubject:
    s = load_subject_samples(cache_dir, subject, "broadband", "car", "all", "position", history=window, pre_lift=True)
    stats = s.eeg_for_stats().astype(np.float64)
    mean, sd = stats.mean(axis=1, keepdims=True), np.sqrt(stats.var(axis=1, keepdims=True) + 1e-8)
    # One grid for every lift (bins of time since LED-on), so residual_r_other_trial finds shared bins across trials.
    bins = np.round(time_since_event(s, "ev_LEDOn", np.ones(s.sample_idx.size, bool)) * s.fs_hz).astype(np.int64)
    mask = bins % stride == 0
    idx = s.sample_idx[mask]
    eeg = ((s.eeg - mean) / sd).astype(np.float32)
    windows = np.stack([eeg[:, i - window + 1:i + 1] for i in idx])
    return ProbeSubject(s, mask, windows, stride)


@torch.no_grad()
def embed(encoder: torch.nn.Module, windows: np.ndarray, device, batch: int = 512, use_gate: bool = True) -> np.ndarray:
    """Last token of each window, float32 (n, d). Restores the encoder's train/eval mode."""
    was_training = encoder.training
    encoder.eval()
    out = []
    for i in range(0, len(windows), batch):
        x = torch.from_numpy(windows[i:i + batch]).to(device)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            out.append(encoder(x, use_gate=use_gate)[:, -1].float().cpu())
    encoder.train(was_training)
    return torch.cat(out).numpy()


def probe_report(encoder: torch.nn.Module, subjects: list[ProbeSubject], pairs: list[tuple[list[str], list[str]]],
                 device, alpha: float = 1.0, seed: int = 0, n_permutations: int = 5) -> dict[str, float]:
    """Embedding health and probe scores of one encoder.

    Args:
        pairs: (train subjects, evaluation subjects) for the ridge probe; scores are averaged over all evaluated
            subjects of all pairs.

    Returns:
        ``emb_sd_min``, ``emb_sd_mean``, ``eff_rank`` (over all probe windows), ``probe_r``,
        ``probe_residual_r_specific`` and per-subject values ``probe_r_<S>``, ``probe_rrs_<S>``.
    """
    by_name = {p.subject: p for p in subjects}
    emb = {p.subject: embed(encoder, p.windows, device) for p in subjects}
    every = np.concatenate(list(emb.values()))
    sd = embedding_std(every)
    out: dict[str, float] = {"emb_sd_min": float(sd.min()), "emb_sd_mean": float(sd.mean()),
                             "eff_rank": effective_rank(every)}
    rs, rrs = [], []
    for train_names, eval_names in pairs:
        train = [by_name[n] for n in train_names]
        w, st = ridge_fit(np.concatenate([emb[p.subject] for p in train]), np.concatenate([p.target for p in train]),
                          alpha)
        ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=seed,
                         train_masks={p.subject: p.mask for p in train}, val_masks={}, logger=logger)
        template = fit_cue_template([p.samples for p in train], ctx)
        for name in eval_names:
            p = by_name[name]
            scores = subject_scores(p.samples, p.mask, ridge_predict(emb[name], w, st), template, ctx, seed,
                                    n_permutations, min_shared_bins=math.ceil(MIN_SHARED_BINS / p.stride))
            out[f"probe_r_{name}"] = scores["r"]
            out[f"probe_rrs_{name}"] = scores["residual_r_specific"]
            rs.append(scores["r"])
            rrs.append(scores["residual_r_specific"])
    out["probe_r"] = float(np.mean(rs))
    out["probe_residual_r_specific"] = float(np.mean(rrs))
    return out
