"""Onset decoder + template: a fair floor that finds the movement onset from EEG, causally.

Step 1, onset decoding. LDA on the lagged low-frequency EEG features (same features as ridge_lag)
separates "movement has not started" (the ``label_window_s`` before the kinematic onset) from
"movement has started" (the ``label_window_s`` after it), on training subjects. On a test lift, the
search starts at LED-on. The decoded onset is the first sample whose probability exceeds the
threshold. If none does by the deadline (a high quantile of the training reaction time), the
deadline itself is taken as the onset, so every lift gets a moving prediction.

Step 2, prediction. Before the decoded onset: the mean of the pre-onset part of the oracle-aligned
template (time since a future onset is unknown, so this is a constant). From the decoded onset on:
the template at (t - decoded onset).

The prediction at sample t uses only EEG up to t, LED-on, and quantities from training subjects.
The threshold is chosen on the inner validation subjects by per-trial r (D7 primary). The test
lift's kinematic onset is used only to report the onset error. Scoring uses the protocol segment
[kinematic onset - 0.5 s, tHandStop], which is not a model input.
"""

from __future__ import annotations

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.baselines.ridge import _chunks
from eegtraj.baselines.template import TemplateBaseline
from eegtraj.data.windows import SubjectSamples, lag_samples, lagged_features
from eegtraj.metrics import aggregate_subject, per_trial_metrics

MS_PER_S = 1000.0


def _led_sample(tr, fs_hz: float) -> int:
    """Index within the lift of the first sample at or after LED-on (sample j is at time (j + 1) / fs)."""
    return max(int(np.ceil(tr.ev_LEDOn * fs_hz)) - 1, 0)


def first_crossing(prob: np.ndarray, threshold: float) -> int | None:
    """Index of the first value above ``threshold``, or None."""
    above = np.flatnonzero(prob > threshold)
    return int(above[0]) if above.size else None


def onset_error_stats(decoded_s: np.ndarray, true_s: np.ndarray, within_ms: float) -> dict[str, float]:
    """Decoded minus true onset, in ms."""
    err = (decoded_s - true_s) * MS_PER_S
    q25, q50, q75 = np.percentile(err, [25, 50, 75])
    return {"median_ms": float(q50), "iqr_ms": float(q75 - q25), "pct_within": float(np.mean(np.abs(err) <= within_ms) * 100),
            "within_ms": within_ms, "n_lifts": int(err.size)}


@register("onset_template")
class OnsetTemplateBaseline(Baseline):
    """Params: ``thresholds`` (grid; 1.0 never fires, leaving the deadline only), ``label_window_s``,
    ``shrinkage`` (LDA covariance), ``deadline_quantile``, ``within_ms`` (for the onset error report)."""

    def __init__(self, thresholds: list[float], label_window_s: float, shrinkage: float, deadline_quantile: float,
                 within_ms: float, **params):
        super().__init__(thresholds=thresholds, label_window_s=label_window_s, shrinkage=shrinkage,
                         deadline_quantile=deadline_quantile, within_ms=within_ms, **params)
        self.thresholds = [float(t) for t in thresholds]
        self.label_window_s = float(label_window_s)
        self.shrinkage = float(shrinkage)
        self.deadline_quantile = float(deadline_quantile)
        self.within_ms = float(within_ms)
        self.onset_error: dict[str, dict[str, float]] = {}

    def _label_samples(self, s: SubjectSamples, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample indices and labels (0 before onset, 1 from onset on) around the onset of the given lifts."""
        half = int(round(self.label_window_s * s.fs_hz))
        idx, labels = [], []
        for tr in s.trials.iloc[rows].itertuples(index=False):
            onset = int(round(tr.onset_s * s.fs_hz)) - 1
            first, last = max(onset - half, self.history - 1), min(onset + half, tr.n_samples)
            j = np.arange(first, last)
            idx.append(tr.offset_start + j)
            labels.append(j >= onset)
        return np.concatenate(idx), np.concatenate(labels)

    def _fit_lda(self, train: list[SubjectSamples], ctx: FitContext) -> None:
        """Weights and bias of equal-prior LDA, from class means and the pooled within-class covariance."""
        p = train[0].eeg.shape[0] * len(self.lags)
        n, sums, scatter = np.zeros(2), np.zeros((2, p)), np.zeros((p, p))
        for s in train:
            eeg = ctx.eeg_norm[s.subject].transform(s.eeg)
            idx, labels = self._label_samples(s, np.unique(s.trial_row[ctx.train_masks[s.subject]]))
            for sl in _chunks(len(idx)):
                x = lagged_features(eeg, idx[sl], self.lags).astype(np.float64)
                y = labels[sl]
                scatter += x.T @ x
                for c, m in enumerate((~y, y)):
                    n[c] += m.sum()
                    sums[c] += x[m].sum(0)
        means = sums / n[:, None]
        cov = (scatter - n[0] * np.outer(means[0], means[0]) - n[1] * np.outer(means[1], means[1])) / n.sum()
        cov = (1 - self.shrinkage) * cov + self.shrinkage * (np.trace(cov) / p) * np.eye(p)
        self.w = np.linalg.solve(cov, means[1] - means[0])
        self.b = -0.5 * self.w @ (means[0] + means[1])
        self.n_lda = n.astype(int).tolist()

    def _search_probs(self, s: SubjectSamples, rows: np.ndarray, ctx: FitContext) -> dict[int, tuple[int, np.ndarray]]:
        """Per lift: (LED-on sample within the lift, P(started) from LED-on to the deadline)."""
        eeg = ctx.eeg_norm[s.subject].transform(s.eeg)
        out = {}
        for row in rows:
            tr = s.trials.iloc[int(row)]
            led = _led_sample(tr, s.fs_hz)
            stop = min(led + int(round(self.deadline_s * s.fs_hz)), int(tr.n_samples) - 1)
            idx = int(tr.offset_start) + np.arange(led, stop + 1)
            score = lagged_features(eeg, idx, self.lags).astype(np.float64) @ self.w + self.b
            out[int(row)] = (led, 1.0 / (1.0 + np.exp(-score)))
        return out

    def _decode_onsets(self, probs: dict[int, tuple[int, np.ndarray]], threshold: float) -> dict[int, tuple[int, bool]]:
        """Per lift: (decoded onset sample within the lift, True if the threshold fired before the deadline)."""
        out = {}
        for row, (led, p) in probs.items():
            hit = first_crossing(p, threshold)
            out[row] = (led + hit, True) if hit is not None else (led + len(p) - 1, False)
        return out

    def _trajectory(self, s: SubjectSamples, mask: np.ndarray, onsets: dict[int, tuple[int, bool]]) -> np.ndarray:
        """Predictions (n_selected, 3): constant before the decoded onset, template from it on."""
        rows = s.trial_row[mask]
        decoded = np.empty(len(s.trials), dtype=np.int64)
        for row, (j, _) in onsets.items():
            decoded[row] = j
        since = s.sample_idx[mask] - s.trials.offset_start.to_numpy()[rows] - decoded[rows]
        started = since >= 0
        template = self.template.template
        pred = np.tile(self.pre_value, (len(rows), 1)).astype(np.float32)
        pred[started] = template[:, np.minimum(since[started] - self.template.bin0, template.shape[1] - 1)].T
        return pred

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        self.lags = lag_samples(ctx.history, ctx.lag_step)
        self.history = ctx.history
        self.template = TemplateBaseline()
        self.template.fit(train, val, ctx)
        pre_bins = -self.template.bin0
        if pre_bins <= 0:
            raise ValueError("the template has no pre-onset part")
        self.pre_value = self.template.template[:, :pre_bins].mean(axis=1)
        rts = np.concatenate([(s.trials.onset_s - s.trials.ev_LEDOn).to_numpy()[np.unique(s.trial_row[ctx.train_masks[s.subject]])]
                              for s in train])
        self.deadline_s = float(np.quantile(rts, self.deadline_quantile))
        self._fit_lda(train, ctx)
        self.val_scores: dict[float, float] = {}
        cached = []
        for s in val:
            m = ctx.val_masks[s.subject]
            if m.any():
                cached.append((s, m, self._search_probs(s, np.unique(s.trial_row[m]), ctx)))
        for threshold in self.thresholds:
            scores = []
            for s, m, probs in cached:
                pred = self._trajectory(s, m, self._decode_onsets(probs, threshold))
                true = s.target[:, s.sample_idx[m]].T
                scores.append(aggregate_subject(per_trial_metrics(true, pred, s.trial_row[m]))["r"]["mean"])
            self.val_scores[threshold] = float(np.nanmean(scores)) if scores else float("nan")
        valid = {t: v for t, v in self.val_scores.items() if np.isfinite(v)}
        self.threshold = max(valid, key=valid.get) if valid else self.thresholds[0]
        ctx.logger.info("onset_template: LDA samples before/after onset %s, deadline %.3f s after LED-on, val r per "
                        "threshold %s, chosen threshold %.3f", self.n_lda, self.deadline_s,
                        {t: round(v, 4) for t, v in self.val_scores.items()}, self.threshold)

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        rows = np.unique(subject.trial_row[mask])
        onsets = self._decode_onsets(self._search_probs(subject, rows, ctx), self.threshold)
        tr = subject.trials.iloc[rows]
        decoded_s = np.array([(onsets[int(r)][0] + 1) / subject.fs_hz for r in rows])
        stats = onset_error_stats(decoded_s, tr.onset_s.to_numpy(), self.within_ms)
        stats["frac_deadline"] = float(np.mean([not onsets[int(r)][1] for r in rows]))
        stats["frac_fired_at_led"] = float(np.mean([onsets[int(r)][1] and onsets[int(r)][0] == _led_sample(t, subject.fs_hz)
                                                    for r, t in zip(rows, tr.itertuples(index=False))]))
        self.onset_error[subject.subject] = stats
        ctx.logger.info("%s onset error (decoded - true): median %.0f ms, IQR %.0f ms, %.1f%% within +-%.0f ms, "
                        "%.1f%% reached the deadline, %.1f%% fired at LED-on", subject.subject, stats["median_ms"],
                        stats["iqr_ms"], stats["pct_within"], self.within_ms, 100 * stats["frac_deadline"],
                        100 * stats["frac_fired_at_led"])
        return self._trajectory(subject, mask, onsets)

    def info(self) -> dict:
        return {"oracle_timing": False, "cue_timing": True,
                "note": "onset decoded causally from EEG after LED-on; scored on [onset - 0.5 s, tHandStop], "
                        "which is not a model input",
                "threshold": self.threshold, "val_r_per_threshold": {str(t): v for t, v in self.val_scores.items()},
                "deadline_s": self.deadline_s, "n_lda_samples": self.n_lda, "lags_samples": self.lags.tolist(),
                "causal": True, "onset_error": self.onset_error}
