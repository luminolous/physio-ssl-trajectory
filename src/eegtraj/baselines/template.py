"""Template baseline: mean training trajectory vs time since movement onset.

Uses the test trial's own kinematic onset, i.e. **oracle timing**. It is the floor that
every decoder must beat (see .agents/data-baselines/baselines.md).
"""

from __future__ import annotations

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.data.windows import SubjectSamples


def time_since_onset(subject: SubjectSamples, idx_mask: np.ndarray | None = None) -> np.ndarray:
    """Seconds since the kinematic onset for each (selected) target sample."""
    idx = subject.sample_idx if idx_mask is None else subject.sample_idx[idx_mask]
    rows = subject.trial_row if idx_mask is None else subject.trial_row[idx_mask]
    tr = subject.trials
    t = (idx - tr.offset_start.to_numpy()[rows] + 1) / subject.fs_hz
    return t - tr.onset_s.to_numpy()[rows]


@register("template")
class TemplateBaseline(Baseline):
    """Per-axis mean target on a grid of time since onset (bin width = 1 / fs)."""

    uses_eeg = False

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        taus, ys = [], []
        for s in train:
            m = ctx.train_masks[s.subject]
            taus.append(time_since_onset(s, m))
            ys.append(s.target[:, s.sample_idx[m]])
        tau = np.concatenate(taus)
        y = np.concatenate(ys, axis=1)
        self.fs_hz = train[0].fs_hz
        bins = np.round(tau * self.fs_hz).astype(np.int64)
        self.bin0 = int(bins.min())
        b = bins - self.bin0
        n_bins = int(b.max()) + 1
        counts = np.bincount(b, minlength=n_bins)
        sums = np.stack([np.bincount(b, weights=y[a], minlength=n_bins) for a in range(y.shape[0])])
        filled = counts > 0
        # Bins without training samples (rare gaps) take the nearest earlier filled bin.
        last = np.maximum.accumulate(np.where(filled, np.arange(n_bins), 0))
        self.template = (sums[:, last] / counts[last]).astype(np.float32)  # (3, n_bins)
        self.counts = counts
        ctx.logger.info("template: %d bins from %.2f to %.2f s after onset, %d training samples",
                        n_bins, self.bin0 / self.fs_hz, (self.bin0 + n_bins - 1) / self.fs_hz, len(tau))

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext) -> np.ndarray:
        b = np.round(time_since_onset(subject, mask) * self.fs_hz).astype(np.int64) - self.bin0
        b = np.clip(b, 0, self.template.shape[1] - 1)  # hold the edge values outside the training range
        return self.template[:, b].T

    def info(self) -> dict:
        return {"oracle_timing": True,
                "note": "uses the test trial's kinematic onset (oracle timing)",
                "n_bins": int(self.template.shape[1]),
                "min_count_per_bin": int(self.counts[self.counts > 0].min())}
