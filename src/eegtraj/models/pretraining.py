"""Pretraining objectives for the neural-branch encoder (design.md "Stage 1"; decisions SSL-6 to SSL-12).

Shared pretext task: a contiguous block of 30-50 % of the 10 tokens is masked in the context view; a non-causal
predictor (2 layers, d = 64) fills the masked positions from the kept context tokens plus learnable mask tokens with
positional embeddings. The objectives differ in the target and in the collapse mechanism:

- ``lejepa`` (masked-predictor variant): target = the same encoder (no gate) on the target view, with gradients.
  Loss (1 - lambda) * MSE(pred, target) + lambda * SIGReg + lambda_gate * ||g - g_prior||^2. SIGReg is applied to the
  encoder tokens of each view (context: kept tokens only; target: all tokens), samples = batch x tokens, averaged over
  the two views. Sources: LeJEPA, arXiv 2511.08544 v3, Eq. 9 (Section 5.1) and SIGReg (Section 4.2.3); reference code
  rbalestr-lab/lejepa @ c293d291, ``lejepa/univariate/epps_pulley.py`` and ``lejepa/multivariate/slicing.py``.
- ``jepa`` (classic, I-JEPA): target = EMA copy of the encoder (no gate) under no-grad, layer-normed over the feature
  dimension; loss MSE(pred, target) + lambda_gate * gate penalty; EMA momentum 0.996 -> 1.0 linear over the steps.
  Sources: I-JEPA, arXiv 2301.08243 Section 3 and Appendix A.1; facebookresearch/ijepa @ 52c1ae95 ``src/train.py``.
- ``mae``: masked reconstruction of the raw window. A light decoder (same form as the predictor) maps the kept tokens
  plus mask tokens to the raw samples (32 x 20) of each masked patch; loss MSE on the masked patches + gate penalty.

Target view (``eegtraj.data.target_view``): guided = ICA-cleaned window x w_phys; unguided = the raw window.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eegtraj.data.target_view import target_view
from eegtraj.models.encoder import CausalEncoder

OBJECTIVES = ("lejepa", "jepa", "mae")


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer with the Epps-Pulley statistic (LeJEPA Section 4.2.3).

    For random unit directions a (resampled every call, seeded by ``step``), the statistic of the projections
    x = z a is N * integral |phi_hat(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt, by the trapezoid rule on t in [0, t_max]
    with doubled inner weights (symmetry), as in ``lejepa/univariate/epps_pulley.py``; averaged over slices.
    Computed in float32.
    """

    def __init__(self, n_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        if n_points % 2 != 1:
            raise ValueError("n_points must be odd")
        t = torch.linspace(0.0, t_max, n_points)
        dt = t_max / (n_points - 1)
        w = torch.full((n_points,), 2 * dt)
        w[[0, -1]] = dt
        phi = torch.exp(-0.5 * t ** 2)
        self.n_slices = n_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)

    def directions(self, dim: int, step: int, device) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(int(step))
        a = torch.randn(dim, self.n_slices, generator=g, device=device)
        return a / a.norm(dim=0, keepdim=True)

    def forward(self, z: torch.Tensor, step: int) -> torch.Tensor:
        """z: (N, D) samples. Returns the mean statistic over slices (scalar)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            x = z @ self.directions(z.shape[-1], step, z.device)  # (N, S)
            xt = x.unsqueeze(-1) * self.t  # (N, S, K)
            err = (torch.cos(xt).mean(0) - self.phi) ** 2 + torch.sin(xt).mean(0) ** 2  # (S, K)
            return ((err @ self.weights) * z.shape[0]).mean()


def sample_block_mask(batch: int, n_tokens: int, ratio_range: tuple[float, float], generator: torch.Generator,
                      device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """One contiguous masked block per window; the block length is shared by the batch, its start is per window.

    Returns:
        keep (B, n_tokens - L) and masked (B, L) token indices, both sorted. L is drawn uniformly from the integers in
        [ceil(lo * n_tokens), floor(hi * n_tokens)].
    """
    lo, hi = math.ceil(ratio_range[0] * n_tokens), math.floor(ratio_range[1] * n_tokens)
    if not 1 <= lo <= hi < n_tokens:
        raise ValueError(f"invalid mask ratio range {ratio_range} for {n_tokens} tokens")
    length = int(torch.randint(lo, hi + 1, (1,), generator=generator))
    start = torch.randint(0, n_tokens - length + 1, (batch,), generator=generator)
    idx = torch.arange(n_tokens).expand(batch, -1)
    is_masked = (idx >= start[:, None]) & (idx < start[:, None] + length)
    keep = idx[~is_masked].reshape(batch, n_tokens - length)
    masked = idx[is_masked].reshape(batch, length)
    return keep.to(device), masked.to(device)


class MaskedPredictor(nn.Module):
    """Non-causal transformer that fills masked positions (I-JEPA style predictor, also the MAE decoder).

    Input: context tokens (B, n_kept, d_in) at positions ``keep``; output (B, L, d_out) at positions ``masked``.
    """

    def __init__(self, n_tokens: int = 10, d_in: int = 128, d_model: int = 64, d_out: int = 128, n_layers: int = 2,
                 n_heads: int = 4, mlp_width: int = 128, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.pos = nn.Parameter(torch.zeros(n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, mlp_width, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, context: torch.Tensor, keep: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        b, n_tok = context.shape[0], self.pos.shape[0]
        h = (self.mask_token + self.pos).expand(b, n_tok, -1).clone()
        ctx = self.inp(context) + self.pos[keep]
        h = h.scatter(1, keep.unsqueeze(-1).expand(-1, -1, h.shape[-1]), ctx.to(h.dtype))
        h = self.norm(self.blocks(h))
        h = torch.gather(h, 1, masked.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        return self.out(h)


def gather_tokens(z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """z (B, n, d), idx (B, k) -> (B, k, d)."""
    return torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))


def patches(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, T) -> (B, T // patch, C * patch), patch k = samples [k * patch, (k + 1) * patch)."""
    b, c, t = x.shape
    return x.reshape(b, c, t // patch, patch).permute(0, 2, 1, 3).reshape(b, t // patch, c * patch)


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """I-JEPA schedule: linear from ``start`` to ``end`` over ``total_steps``."""
    return start + (end - start) * min(max(step, 0), total_steps) / max(total_steps, 1)


class Pretrainer(nn.Module):
    """Encoder + objective. ``forward(raw, cleaned, step)`` returns a dict of scalar losses with key ``loss``.

    Args:
        encoder: the online encoder (kept after pretraining).
        objective: one of :data:`OBJECTIVES`. guided: target view ICA-cleaned x w_phys (else the raw window).
        w_phys: (32,) target-view weights. lam: LeJEPA lambda. lambda_gate: weight of the gate prior term.
        mask_ratio: range of the masked fraction of tokens. total_steps: for the EMA schedule (``jepa``).
    """

    def __init__(self, encoder: CausalEncoder, objective: str, guided: bool, w_phys: torch.Tensor, lam: float = 0.05,
                 lambda_gate: float = 0.0, mask_ratio: tuple[float, float] = (0.3, 0.5), total_steps: int = 1,
                 predictor: dict | None = None, sigreg: dict | None = None, ema: tuple[float, float] = (0.996, 1.0),
                 seed: int = 0):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}")
        self.encoder, self.objective, self.guided = encoder, objective, guided
        self.lam, self.lambda_gate, self.mask_ratio = float(lam), float(lambda_gate), tuple(mask_ratio)
        self.total_steps, self.ema_range = int(total_steps), tuple(ema)
        self.register_buffer("w_phys", torch.as_tensor(w_phys, dtype=torch.float32))
        d, n_tok, patch = encoder.out_norm.normalized_shape[0], encoder.n_tokens, encoder.patch
        n_chans = encoder.gate.prior.numel()
        p = {"n_tokens": n_tok, "d_in": d, "d_out": d, **(predictor or {})}
        if objective == "mae":
            p["d_out"] = n_chans * patch
        self.predictor = MaskedPredictor(**p)
        self.sigreg = SIGReg(**(sigreg or {})) if objective == "lejepa" else None
        self.target_encoder = None
        if objective == "jepa":
            self.target_encoder = copy.deepcopy(encoder)
            for q in self.target_encoder.parameters():
                q.requires_grad_(False)
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

    def target_input(self, raw: torch.Tensor, cleaned: torch.Tensor | None) -> torch.Tensor:
        """The target-view window: cleaned x w_phys (guided) or the raw window (unguided)."""
        if self.guided and cleaned is None:
            raise ValueError("guided objectives need the cleaned window")
        return target_view(raw, cleaned, self.w_phys.to(raw.dtype), self.guided)

    def forward(self, raw: torch.Tensor, cleaned: torch.Tensor | None, step: int) -> dict[str, torch.Tensor]:
        keep, masked = sample_block_mask(raw.shape[0], self.encoder.n_tokens, self.mask_ratio, self._gen, raw.device)
        ctx = self.encoder(raw, keep=keep)  # gated context view, masked tokens dropped
        gate = self.encoder.gate.prior_penalty()
        out: dict[str, torch.Tensor] = {"gate": gate}
        if self.objective == "mae":
            pred = self.predictor(ctx, keep, masked)
            out["pred"] = F.mse_loss(pred.float(), gather_tokens(patches(raw, self.encoder.patch), masked).float())
            out["loss"] = out["pred"] + self.lambda_gate * gate
            return out
        pred = self.predictor(ctx, keep, masked)
        tgt_in = self.target_input(raw, cleaned)
        if self.objective == "lejepa":
            tgt = self.encoder(tgt_in, use_gate=False)  # gradients flow (no stop-gradient)
            out["pred"] = F.mse_loss(pred.float(), gather_tokens(tgt, masked).float())
            d = tgt.shape[-1]
            out["sigreg"] = 0.5 * (self.sigreg(ctx.reshape(-1, d), step) + self.sigreg(tgt.reshape(-1, d), step))
            out["loss"] = (1 - self.lam) * out["pred"] + self.lam * out["sigreg"] + self.lambda_gate * gate
            return out
        with torch.no_grad():
            tgt = self.target_encoder(tgt_in, use_gate=False)
            tgt = F.layer_norm(tgt.float(), (tgt.shape[-1],))
        out["pred"] = F.mse_loss(pred.float(), gather_tokens(tgt, masked))
        out["loss"] = out["pred"] + self.lambda_gate * gate
        return out

    @torch.no_grad()
    def after_step(self, step: int) -> float | None:
        """EMA update of the target encoder (``jepa`` only). Returns the momentum used."""
        if self.target_encoder is None:
            return None
        m = ema_momentum(step, self.total_steps, *self.ema_range)
        for q_online, q_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            q_target.mul_(m).add_(q_online.detach(), alpha=1 - m)
        return m


# ---------------------------------------------------------------- health metrics


def embedding_std(z: np.ndarray) -> np.ndarray:
    """Per-dimension SD of embeddings (N, D) -> (D,). Collapse: values near 0."""
    return np.asarray(z, np.float64).std(axis=0)


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of the normalized singular values of the centered embeddings (Roy & Vetterli 2007); in [1, D]."""
    x = np.asarray(z, np.float64)
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    p = s / s.sum() if s.sum() > 0 else np.ones_like(s) / s.size
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge with intercept on standardized features. Returns (W (D+1, k), feature mean/SD stack)."""
    x = np.asarray(x, np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    xs = np.hstack([(x - mu) / sd, np.ones((x.shape[0], 1))])
    reg = alpha * np.eye(xs.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xs.T @ xs + reg, xs.T @ np.asarray(y, np.float64))
    return w, np.stack([mu, sd])


def ridge_predict(x: np.ndarray, w: np.ndarray, stats: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, np.float64) - stats[0]) / stats[1]
    return np.hstack([xs, np.ones((xs.shape[0], 1))]) @ w


def linear_probe_r(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                   val_trial: np.ndarray, alpha: float = 1.0) -> float:
    """Ridge probe from embeddings to 3D targets; Pearson r per validation trial and axis, mean over trials then axes
    (D7 style, one pooled group). Trials with a constant axis are skipped for that axis."""
    w, st = ridge_fit(train_x, train_y, alpha)
    pred = ridge_predict(val_x, w, st)
    per_axis = []
    for a in range(val_y.shape[1]):
        rs = []
        for tr in np.unique(val_trial):
            m = val_trial == tr
            yt, yp = val_y[m, a], pred[m, a]
            if m.sum() > 2 and yt.std() > 0 and yp.std() > 0:
                rs.append(np.corrcoef(yt, yp)[0, 1])
        per_axis.append(np.mean(rs) if rs else np.nan)
    return float(np.nanmean(per_axis))
