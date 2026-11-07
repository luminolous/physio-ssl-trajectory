"""Lagged ridge regression (Bradberry et al., 2010 style), multi-output over x, y, z.

Features: normalized EEG of every channel at causal lags 0, step, ..., < history samples,
flattened (channel-major). Targets: normalized with training statistics. The full design
matrix does not fit in memory (~0.5 GB per subject), so the model is fitted from
sufficient statistics (X^T X, X^T y, sums) accumulated per subject in chunks.

Alpha is chosen from a fixed grid on the inner validation subjects by the primary metric
(per-trial Pearson r, mean over trials then axes, then subjects). The final model is the
training-subject fit at that alpha; validation data are used only for selection.
"""

from __future__ import annotations

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.data.windows import SubjectSamples, lag_samples, lagged_features
from eegtraj.metrics import aggregate_subject, per_trial_metrics

CHUNK_ROWS = 50_000


def haufe_pattern(cov_x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Activation pattern of a linear backward model (Haufe et al., 2014): cov(X) W cov(y_hat)^-1.

    Args:
        cov_x: (p, p) feature covariance. w: (p, k) weights.

    Returns:
        (p, k) pattern, in the units of the features per unit of each output.
    """
    return cov_x @ w @ np.linalg.inv(w.T @ cov_x @ w)


def _chunks(n: int, size: int = CHUNK_ROWS):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


@register("ridge_lag")
class RidgeLagBaseline(Baseline):
    """Params: ``alphas`` (grid, relative to mean feature variance), ``lag_step`` override (optional)."""

    def __init__(self, alphas: list[float] | None = None, **params):
        super().__init__(alphas=alphas, **params)
        self.alphas = np.asarray(alphas if alphas is not None else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], dtype=np.float64)

    def _features(self, s: SubjectSamples, mask: np.ndarray, ctx: FitContext, rows: slice | None = None) -> np.ndarray:
        eeg = ctx.eeg_norm[s.subject].transform(s.eeg)
        idx = s.sample_idx[mask]
        return lagged_features(eeg, idx if rows is None else idx[rows], self.lags)

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        self.lags = lag_samples(ctx.history, int(self.params.get("lag_step") or ctx.lag_step))
        p = train[0].eeg.shape[0] * len(self.lags)
        xtx = np.zeros((p, p))
        xty = np.zeros((p, 3))
        sx, sy, n = np.zeros(p), np.zeros(3), 0
        for s in train:
            m = ctx.train_masks[s.subject]
            eeg = ctx.eeg_norm[s.subject].transform(s.eeg)
            idx = s.sample_idx[m]
            y_all = ctx.target_norm.transform(s.target[:, idx]).T.astype(np.float64)
            for sl in _chunks(len(idx)):
                x = lagged_features(eeg, idx[sl], self.lags).astype(np.float64)
                y = y_all[sl]
                xtx += x.T @ x
                xty += x.T @ y
                sx += x.sum(0)
                sy += y.sum(0)
                n += len(x)
        mx, my = sx / n, sy / n
        xtx_c = xtx - n * np.outer(mx, mx)
        xty_c = xty - n * np.outer(mx, my)
        scale = np.trace(xtx_c) / p  # alphas are relative to the mean feature sum of squares
        self.n_features, self.n_train = p, n
        self.mx, self.my = mx, my
        weights = {a: np.linalg.solve(xtx_c + a * scale * np.eye(p), xty_c) for a in self.alphas}

        if val:
            scores = {a: self._val_score(val, ctx, w) for a, w in weights.items()}
            self.val_scores = {float(a): float(v) for a, v in scores.items()}
            self.alpha = float(max(scores, key=scores.get))
        else:
            self.val_scores, self.alpha = {}, float(self.alphas[len(self.alphas) // 2])
        self.w = weights[self.alpha]
        self.b = my - mx @ self.w
        self.pattern = haufe_pattern(xtx_c / (n - 1), self.w)  # training-subject covariance, normalized units
        ctx.logger.info("ridge_lag: %d features (%d lags), %d training samples, val r per alpha %s, chosen alpha %g",
                        p, len(self.lags), n, {f"{a:g}": round(v, 4) for a, v in self.val_scores.items()}, self.alpha)

    def _val_score(self, val: list[SubjectSamples], ctx: FitContext, w: np.ndarray) -> float:
        b = self.my - self.mx @ w
        per_subject = []
        for s in val:
            m = ctx.val_masks[s.subject]
            if not m.any():
                continue
            z = np.vstack([self._features(s, m, ctx, sl) @ w + b for sl in _chunks(int(m.sum()))])
            y_pred = ctx.target_norm.inverse(z.T.astype(np.float32)).T
            y_true = s.target[:, s.sample_idx[m]].T
            per_subject.append(aggregate_subject(per_trial_metrics(y_true, y_pred, s.trial_row[m]))["r"]["mean"])
        return float(np.nanmean(per_subject))

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        z = np.vstack([self._features(subject, mask, ctx, sl) @ self.w + self.b for sl in _chunks(int(mask.sum()))])
        return ctx.target_norm.inverse(z.T.astype(np.float32)).T

    def save(self, path) -> None:
        """Weights and Haufe patterns; feature index = channel * n_lags + lag index (channel-major)."""
        np.savez(path.with_suffix(".npz"), w=self.w, b=self.b, pattern=self.pattern, lags_samples=self.lags,
                 alpha=self.alpha, mx=self.mx, my=self.my)

    def info(self) -> dict:
        return {"alpha": self.alpha, "alpha_grid": self.alphas.tolist(), "val_r_per_alpha": self.val_scores,
                "n_features": self.n_features, "n_lags": len(self.lags), "lags_samples": self.lags.tolist(),
                "n_train_samples": self.n_train,
                "alpha_note": "relative to the mean diagonal of the centered X^T X"}
