"""Kalman filter decoder (baselines.md: kalman; Wu et al. 2006 style).

State x_t (6): normalized wrist position (3) and its velocity (3, finite difference x fs).
Observation z_t (k): first k principal components of the lagged low-frequency EEG
features (same features as ridge_lag), PCA fitted on training samples only.

Model, all fitted by least squares on training subjects:
    x_t = A x_{t-1} + w,  w ~ N(0, Q)   (consecutive samples within a lift)
    z_t = H x_t + c + v,  v ~ N(0, R)
Initial state of each lift: training mean and covariance of the state at the first
scored sample. Decoding runs the causal Kalman filter (no smoother), so each estimate
uses only EEG up to that sample. The gain sequence does not depend on data, so it is
computed once and all lifts of a subject are filtered in one batch.

The number of PCA components is chosen from a fixed grid on the inner validation
subjects by per-trial r (D7 primary).

Timing caveat: every lift's filter starts at the first scored sample, i.e. at the kinematic
onset - 0.5 s. With the learned dynamics A and initial state x0, the filter alone produces
a movement-shaped trajectory from that start, so the Kalman decoder inherits **oracle onset
timing** like the template. ``prior_only: true`` sets every gain to 0 (EEG ignored) and
measures what the dynamics prior alone achieves.
"""

from __future__ import annotations

import numpy as np

from eegtraj.baselines.base import Baseline, FitContext, register
from eegtraj.baselines.ridge import _chunks
from eegtraj.data.windows import SubjectSamples, lag_samples, lagged_features
from eegtraj.metrics import aggregate_subject, per_trial_metrics

N_STATE = 6
RIDGE_EPS = 1e-6  # added to covariance diagonals before inversion


def _state(s: SubjectSamples, idx: np.ndarray, ctx: FitContext) -> np.ndarray:
    """(n, 6) normalized position and velocity at target samples ``idx``."""
    pos = ctx.target_norm.transform(s.target[:, idx]).T
    prev = ctx.target_norm.transform(s.target[:, idx - 1]).T  # idx - 1 is inside the lift (history >= 2)
    return np.hstack([pos, (pos - prev) * s.fs_hz]).astype(np.float64)


def _gains(A, H, Q, R, P0, n_steps):
    """Kalman gains K_t (n_steps, 6, k) for the predict-then-update filter starting from P0."""
    gains = np.empty((n_steps, A.shape[0], H.shape[0]))
    P = P0
    for t in range(n_steps):
        if t > 0:
            P = A @ P @ A.T + Q
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T  # P H^T S^-1 (S symmetric)
        gains[t] = K
        P = (np.eye(A.shape[0]) - K @ H) @ P
    return gains


@register("kalman")
class KalmanBaseline(Baseline):
    """Params: ``n_components`` grid (PCA components of the lagged EEG features)."""

    def __init__(self, n_components: list[int] | None = None, prior_only: bool = False, **params):
        super().__init__(n_components=n_components, prior_only=prior_only, **params)
        self.grid = [int(k) for k in (n_components or [10, 20, 50])]
        self.prior_only = bool(prior_only)

    def _z(self, s: SubjectSamples, mask: np.ndarray, ctx: FitContext) -> np.ndarray:
        """(n, k_max) PCA scores of the lagged features for the selected target samples."""
        eeg = ctx.eeg_norm[s.subject].transform(s.eeg)
        idx = s.sample_idx[mask]
        return np.vstack([(lagged_features(eeg, idx[sl], self.lags) - self.mx) @ self.V
                          for sl in _chunks(len(idx))])

    def fit(self, train: list[SubjectSamples], val: list[SubjectSamples], ctx: FitContext) -> None:
        self.lags = lag_samples(ctx.history, ctx.lag_step)
        k_max = max(self.grid)

        # Pass 1: feature covariance for PCA.
        p = train[0].eeg.shape[0] * len(self.lags)
        xtx, sx, n = np.zeros((p, p)), np.zeros(p), 0
        for s in train:
            eeg = ctx.eeg_norm[s.subject].transform(s.eeg)
            idx = s.sample_idx[ctx.train_masks[s.subject]]
            for sl in _chunks(len(idx)):
                f = lagged_features(eeg, idx[sl], self.lags).astype(np.float64)
                xtx += f.T @ f
                sx += f.sum(0)
                n += len(f)
        self.mx = sx / n
        cov = (xtx - n * np.outer(self.mx, self.mx)) / n
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1][:k_max]
        self.V = evecs[:, order]
        self.explained = (evals[order] / evals.sum()).cumsum()

        # Pass 2: sufficient statistics for A, Q (transitions) and H, c, R (observations).
        sxx_prev = np.zeros((N_STATE, N_STATE))
        sx_next_prev = np.zeros((N_STATE, N_STATE))
        sxx_next = np.zeros((N_STATE, N_STATE))
        n_tr = 0
        sx_, sz_, sxx, szx, szz, n_obs = (np.zeros(N_STATE), np.zeros(k_max), np.zeros((N_STATE, N_STATE)),
                                          np.zeros((k_max, N_STATE)), np.zeros((k_max, k_max)), 0)
        first_states = []
        for s in train:
            m = ctx.train_masks[s.subject]
            idx, rows = s.sample_idx[m], s.trial_row[m]
            x = _state(s, idx, ctx)
            z = self._z(s, m, ctx)
            sx_ += x.sum(0); sz_ += z.sum(0); sxx += x.T @ x; szx += z.T @ x; szz += z.T @ z; n_obs += len(x)
            consecutive = (rows[1:] == rows[:-1]) & (idx[1:] == idx[:-1] + 1)
            xp, xn = x[:-1][consecutive], x[1:][consecutive]
            sxx_prev += xp.T @ xp; sx_next_prev += xn.T @ xp; sxx_next += xn.T @ xn; n_tr += len(xp)
            starts = np.r_[True, rows[1:] != rows[:-1]]
            first_states.append(x[starts])

        eye = np.eye(N_STATE)
        self.A = sx_next_prev @ np.linalg.inv(sxx_prev + RIDGE_EPS * eye)
        self.Q = (sxx_next - self.A @ sx_next_prev.T) / n_tr
        self.Q = (self.Q + self.Q.T) / 2 + RIDGE_EPS * eye
        first = np.vstack(first_states)
        self.x0, self.P0 = first.mean(0), np.cov(first.T) + RIDGE_EPS * eye
        mx_, mz_ = sx_ / n_obs, sz_ / n_obs
        cxx = sxx - n_obs * np.outer(mx_, mx_)
        czx = szx - n_obs * np.outer(mz_, mx_)
        czz = szz - n_obs * np.outer(mz_, mz_)
        self.models = {}
        for k in self.grid:
            H = czx[:k] @ np.linalg.inv(cxx + RIDGE_EPS * eye)
            c = mz_[:k] - H @ mx_
            R = (czz[:k, :k] - H @ czx[:k].T) / n_obs
            self.models[k] = (H, c, (R + R.T) / 2 + RIDGE_EPS * np.eye(k))
        self.n_train = n_obs

        if val:
            self.val_scores = {k: self._val_score(val, ctx, k) for k in self.grid}
            self.k = max(self.val_scores, key=self.val_scores.get)
        else:
            self.val_scores, self.k = {}, self.grid[len(self.grid) // 2]
        ctx.logger.info("kalman: %d features -> PCA %s comps (explained %s), %d training samples, "
                        "val r per k %s, chosen k %d", p, self.grid,
                        {k: round(float(self.explained[k - 1]), 3) for k in self.grid}, n_obs,
                        {k: round(v, 4) for k, v in self.val_scores.items()}, self.k)

    def _filter(self, s: SubjectSamples, mask: np.ndarray, ctx: FitContext, k: int) -> np.ndarray:
        """Causal Kalman filter over all selected lifts in one batch; returns positions (n, 3) in cm."""
        H, c, R = self.models[k]
        z = self._z(s, mask, ctx)[:, :k] - c
        rows = s.trial_row[mask]
        starts = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1]
        lengths = np.diff(np.r_[starts, len(rows)])
        gains = _gains(self.A, H, self.Q, R, self.P0, int(lengths.max()))
        if self.prior_only:
            gains[:] = 0.0  # control: dynamics prior from x0 only, no EEG
        x = np.tile(self.x0, (len(starts), 1))  # (n_trials, 6)
        out = np.empty((len(rows), N_STATE))
        for t in range(int(lengths.max())):
            live = lengths > t
            pos = starts[live] + t
            xt = x[live] if t == 0 else x[live] @ self.A.T
            innov = z[pos] - xt @ H.T
            xt = xt + innov @ gains[t].T
            x[live] = xt
            out[pos] = xt
        return ctx.target_norm.inverse(out[:, :3].T.astype(np.float32)).T

    def _val_score(self, val: list[SubjectSamples], ctx: FitContext, k: int) -> float:
        scores = []
        for s in val:
            m = ctx.val_masks[s.subject]
            if not m.any():
                continue
            pred = self._filter(s, m, ctx, k)
            true = s.target[:, s.sample_idx[m]].T
            scores.append(aggregate_subject(per_trial_metrics(true, pred, s.trial_row[m]))["r"]["mean"])
        return float(np.nanmean(scores))

    def predict(self, subject: SubjectSamples, mask: np.ndarray, ctx: FitContext, repeat: int = 0) -> np.ndarray:
        return self._filter(subject, mask, ctx, self.k)

    def info(self) -> dict:
        return {"n_components": self.k, "n_components_grid": self.grid, "prior_only": self.prior_only,
                "oracle_timing": True,
                "timing_note": "filter starts at kinematic onset - 0.5 s; dynamics prior carries onset timing",
                "val_r_per_k": {str(k): v for k, v in self.val_scores.items()},
                "pca_explained_variance": {str(k): float(self.explained[k - 1]) for k in self.grid},
                "n_train_samples": self.n_train, "causal": True, "lags_samples": self.lags.tolist(),
                "state": "normalized wrist position (3) + finite-difference velocity (3)"}
