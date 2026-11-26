"""Linear-probe health check for pretrained encoders (design.md, "Pretraining health").

Labeled 2 s windows end at scored target samples (D2 segment, re-centered wrist position D1). The causal history
before the segment start comes from the pre-lift cache. EEG is z-scored per subject over its lift samples (D9,
the same statistics as ``eegtraj.data.unlabeled.UnlabeledWindows``). The probe input is the encoder's last token
(the streaming output for the patch that ends at the target sample), so it uses no EEG after the target.

Ridge probes are scored with r and ``residual_r_specific`` (``eegtraj.baselines.scoring.subject_scores``; the
``template_cue`` floor is fitted on the probe's training data only). Only the screening subjects (P3, P6) are used:
pretraining is not per fold (SSL-1 option b), and inner-validation subjects are test subjects of other folds.
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
_quiet = logging.getLogger(__name__ + ".template")  # template_cue fit logs, once per probe fold
_quiet.setLevel(logging.WARNING)


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


def load_probe_subject(cache_dir: str, subject: str, window: int = 200, stride: int = 10,
                       band: str = "broadband") -> ProbeSubject:
    """``band``: ``broadband`` (the encoder's input) or ``broadband_ica`` (ICA-cleaned cache, its own z-score: probes
    neural information without the removed ocular components). Targets, scored samples and template_cue do not depend
    on the band."""
    s = load_subject_samples(cache_dir, subject, band, "car", "all", "position", history=window, pre_lift=True)
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


def _scores(p: ProbeSubject, mask: np.ndarray, pred: np.ndarray, train: list[tuple[SubjectSamples, np.ndarray]],
            seed: int, n_permutations: int) -> dict[str, float]:
    """r and residual_r_specific of ``pred`` at ``mask``; template_cue fitted on ``train`` (samples, mask) only."""
    ctx = FitContext(eeg_norm={}, target_norm=None, history=1, lag_step=1, seed=seed,
                     train_masks={s.subject: m for s, m in train}, val_masks={}, logger=_quiet)
    template = fit_cue_template([s for s, _ in train], ctx)
    return subject_scores(p.samples, mask, pred, template, ctx, seed, n_permutations,
                          min_shared_bins=math.ceil(MIN_SHARED_BINS / p.stride))


def within_subject_scores(p: ProbeSubject, z: np.ndarray, alpha: float, seed: int,
                          n_permutations: int) -> dict[str, float]:
    """Leave-one-series-out ridge probe inside one subject. Ridge and template_cue use the other series only.

    Per-series scores are averaged with weights = number of lifts in the series (= mean over lifts).
    """
    series = p.samples.trials.series.to_numpy()[p.samples.trial_row]  # per scored sample
    ser_probe = series[p.mask]
    r, rrs, matched, n = [], [], [], []
    for k in np.unique(ser_probe):
        held = ser_probe == k
        w, st = ridge_fit(z[~held], p.target[~held], alpha)
        sc = _scores(p, p.mask & (series == k), ridge_predict(z[held], w, st),
                     [(p.samples, p.mask & (series != k))], seed, n_permutations)
        r.append(sc["r"])
        rrs.append(sc["residual_r_specific"])
        matched.append(sc["residual_r_matched"])
        n.append(np.unique(p.samples.trial_row[p.mask & (series == k)]).size)
    return {"r": float(np.average(r, weights=n)), "rrs": float(np.average(rrs, weights=n)),
            "matched": float(np.average(matched, weights=n))}


def probe_report(encoder: torch.nn.Module, subjects: list[ProbeSubject], pairs: list[tuple[list[str], list[str]]],
                 device, alpha: float = 1.0, seed: int = 0, n_permutations: int = 5) -> dict[str, float]:
    """Embedding health and probe scores of one encoder.

    Two probes (decided 2026-11-15; the within-subject one is the selection score):
    - within-subject: leave-one-series-out ridge inside each probe subject (``within_r``, ``within_rrs``,
      ``within_matched`` = ``residual_r_matched``, design.md SSL-27);
    - cross-subject: ridge fitted on the first subjects of each pair, scored on the others (``cross_r``,
      ``cross_rrs``).

    Returns:
        ``emb_sd_min``, ``emb_sd_mean``, ``eff_rank`` (over all probe windows), the four means above (over subjects),
        and per-subject values ``within_r_<S>``, ``within_rrs_<S>``, ``cross_r_<S>``, ``cross_rrs_<S>``.
    """
    by_name = {p.subject: p for p in subjects}
    emb = {p.subject: embed(encoder, p.windows, device) for p in subjects}
    every = np.concatenate(list(emb.values()))
    sd = embedding_std(every)
    out: dict[str, float] = {"emb_sd_min": float(sd.min()), "emb_sd_mean": float(sd.mean()),
                             "eff_rank": effective_rank(every)}
    acc: dict[str, list[float]] = {"within_r": [], "within_rrs": [], "within_matched": [], "cross_r": [],
                                   "cross_rrs": [], "cross_matched": []}
    for p in subjects:
        sc = within_subject_scores(p, emb[p.subject], alpha, seed, n_permutations)
        for key in ("r", "rrs", "matched"):
            out[f"within_{key}_{p.subject}"] = sc[key]
            acc[f"within_{key}"].append(sc[key])
    for train_names, eval_names in pairs:
        train = [by_name[n] for n in train_names]
        w, st = ridge_fit(np.concatenate([emb[p.subject] for p in train]), np.concatenate([p.target for p in train]),
                          alpha)
        for name in eval_names:
            p = by_name[name]
            sc = _scores(p, p.mask, ridge_predict(emb[name], w, st), [(q.samples, q.mask) for q in train], seed,
                         n_permutations)
            out[f"cross_r_{name}"], out[f"cross_rrs_{name}"] = sc["r"], sc["residual_r_specific"]
            out[f"cross_matched_{name}"] = sc["residual_r_matched"]
            acc["cross_matched"].append(sc["residual_r_matched"])
            acc["cross_r"].append(sc["r"])
            acc["cross_rrs"].append(sc["residual_r_specific"])
    out.update({k: float(np.mean(v)) for k, v in acc.items()})
    return out
