"""Template baselines: mean training trajectory vs time since an event of the lift.

``template`` aligns on the test trial's own kinematic onset, i.e. **oracle timing**. It is the
floor that every decoder must beat (see .agents/data-baselines/baselines.md).

``template_cue`` aligns on LED-on, which the experiment controls, so a real decoder could know
it (**cue timing**, not oracle). Its prediction depends only on time since LED-on. Both are scored
on the protocol segment [kinematic onset - 0.5 s, tHandStop]: the segment defines where we score
and is not a model input.
"""

from __future__ import annotations

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.data.windows import SubjectSamples


def time_since_event(subject: SubjectSamples, column: str, idx_mask: np.ndarray | None = None) -> np.ndarray:
    """Seconds since the event in ``trials[column]`` for each (selected) target sample."""
    idx = subject.sample_idx if idx_mask is None else subject.sample_idx[idx_mask]
    rows = subject.trial_row if idx_mask is None else subject.trial_row[idx_mask]
    tr = subject.trials
    t = (idx - tr.offset_start.to_numpy()[rows] + 1) / subject.fs_hz
    return t - tr[column].to_numpy()[rows]


def time_since_onset(subject: SubjectSamples, idx_mask: np.ndarray | None = None) -> np.ndarray:
    """Seconds since the kinematic onset for each (selected) target sample."""
    return time_since_event(subject, "onset_s", idx_mask)


def reaction_time_stats(subject: SubjectSamples, mask: np.ndarray) -> dict[str, float]:
    """Spread of the reaction time (LED-on to kinematic onset, s) over the selected lifts."""
    tr = subject.trials.iloc[np.unique(subject.trial_row[mask])]
    rt = (tr.onset_s - tr.ev_LEDOn).to_numpy()
    q25, q50, q75 = np.percentile(rt, [25, 50, 75])
    return {"median_s": float(q50), "iqr_s": float(q75 - q25), "sd_s": float(rt.std(ddof=1)), "min_s": float(rt.min()),
            "max_s": float(rt.max()), "n_lifts": int(rt.size), "n_negative": int((rt < 0).sum()),
            "frac_segment_starts_before_led": float((tr.segment_start_s < tr.ev_LEDOn).mean())}


@register("template")
class TemplateBaseline(Baseline):
    """Per-axis mean target on a grid of time since the event (bin width = 1 / fs)."""

    uses_eeg = False
    event_column = "onset_s"

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        taus, ys = [], []
        for s in train:
            m = ctx.train_masks[s.subject]
            taus.append(time_since_event(s, self.event_column, m))
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
        ctx.logger.info("%s: %d bins from %.2f to %.2f s after %s, %d training samples", self.name, n_bins,
                        self.bin0 / self.fs_hz, (self.bin0 + n_bins - 1) / self.fs_hz, self.event_column, len(tau))

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        b = np.round(time_since_event(subject, self.event_column, mask) * self.fs_hz).astype(np.int64) - self.bin0
        b = np.clip(b, 0, self.template.shape[1] - 1)  # hold the edge values outside the training range
        return self.template[:, b].T

    def info(self) -> dict:
        return {"oracle_timing": True,
                "note": "uses the test trial's kinematic onset (oracle timing)",
                "n_bins": int(self.template.shape[1]),
                "min_count_per_bin": int(self.counts[self.counts > 0].min())}


@register("template_cue")
class TemplateCueBaseline(TemplateBaseline):
    """Mean target vs time since LED-on: a fair floor with no information from the test lift's movement."""

    event_column = "ev_LEDOn"

    def __init__(self, **params):
        super().__init__(**params)
        self.reaction_times: dict[str, dict[str, float]] = {}

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        self.reaction_times[subject.subject] = reaction_time_stats(subject, mask)
        return super().predict(subject, mask, ctx, repeat)

    def info(self) -> dict:
        return {"oracle_timing": False, "cue_timing": True,
                "note": "aligned on LED-on; scored on [onset - 0.5 s, tHandStop], which is not a model input",
                "n_bins": int(self.template.shape[1]), "min_count_per_bin": int(self.counts[self.counts > 0].min()),
                "reaction_time": self.reaction_times}
