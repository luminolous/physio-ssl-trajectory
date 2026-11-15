"""Stage 2 decoder of the proposed model (design.md "Stage 2: decoder"): branches, fusion, heads, loss.

A lift is decoded as a stream of 0.2 s patches (20 samples at 100 Hz). Stream inputs carry ``history`` = window -
patch = 180 samples before the first patch, so the token of patch n is computed from the 2 s window ending at the
patch's last sample. Stream sample s (0-based, after the history) is at ``stream_start + (s + 1) / fs``; the stream
starts at a fixed time relative to LED-on (decided 2026-11-15: LED-on - 1.5 s), never at a time derived from the
test lift's movement.

Causality and rates. Patch n "ends" at stream sample e_n = 20 n + 19: its features use EEG up to e_n only. Outputs are
requested at stream sample indices (the scored target samples). Per-patch quantities (A, Delta, mixture weights,
log sigma^2, global-warp tau and s, direct predictions) are turned into 100 Hz values by *causal* linear
interpolation: for s in [e_n, e_{n+1}) the value ramps from patch n-1 to patch n (one-patch delay), so a value at s
uses only patches that end at or before s. The phase tracker is a clock-warp integrator (SSL-19): an internal clock
tau starts at real time, tau(e_0) = t_LED(e_0) (seconds since LED-on), and runs at rate k_n = exp(c tanh(u_n)) in
[e^-c, e^c] (c = ln 2: 0.5x to 2x real time) from patch n's features: for s in [e_n, e_{n+1}),
tau(s) = tau(e_n) + k_n (s - e_n) / fs. Then phi(s) = clip((tau(s) - tau0) / s0, 0, 1), with tau0 the median reaction
time (LED-on -> onset) and s0 the median onset -> hand stop duration of the fold's training lifts. With u = 0 (the
zero-initialized output layer) the clock is real time and phi is exactly the cue phase
clip((t - t_LED - tau0) / s0, 0, 1). tau is continuous and strictly increasing, so phi is continuous, piecewise linear
and monotone non-decreasing. Before e_0 the clock runs at real time (never scored: scored samples start after e_0
by construction of the stream start).

Heads (``head``):
- ``direct``: y = Linear(h).
- ``residual``: y = template_cue(t) + Delta (template_cue supplied per output sample by the caller).
- ``global_warp``: phi(t) = clip((t - t_LED - tau) / s, 0, 1) with tau, s re-estimated at every patch; then the same
  template mixing, amplitude and residual as the proposed head (only the phase source differs).
- ``proposed``: phase tracker (clock warp) + template mixing + amplitude A + residual Delta: y = A * T_mix(phi) + Delta.
Every head also outputs per-axis log sigma^2 and the auxiliary phase logits (5 classes, training only).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eegtraj.models.encoder import CausalEncoder
from eegtraj.models.landmark_template import TemplateBank

HEADS = ("direct", "residual", "global_warp", "proposed")
WARP_HEADS = ("global_warp", "proposed")
# Auxiliary classes (design.md): pre-onset, reach (onset -> first touch), load (first touch -> lift-off),
# lift-hold (lift-off -> replace), release-return (replace -> hand stop).
AUX_CLASSES = ("pre_onset", "reach", "load", "lift_hold", "release_return")
# Cache phase labels (eegtraj.data.cache.PHASES: pre, reach, load, hold, replace, release, post) -> auxiliary class.
# replace (replace -> both released) and release (both released -> hand stop) form release-return; post (at or after
# the hand stop, only the last scored sample) is folded into release-return.
PHASE_TO_AUX = np.array([0, 1, 2, 3, 4, 4, 4], dtype=np.int64)


def aux_classes(phase: np.ndarray) -> np.ndarray:
    """Cache phase labels (int, 0..6) -> auxiliary classes (0..4)."""
    return PHASE_TO_AUX[np.asarray(phase, dtype=np.int64)]


def patch_end_index(n_patches: int, patch: int) -> torch.Tensor:
    return torch.arange(n_patches) * patch + patch - 1


def causal_interp(values: torch.Tensor, out_idx: torch.Tensor, patch: int) -> torch.Tensor:
    """Per-patch values (B, N, k) -> values at stream samples ``out_idx`` (B, S), shape (B, S, k).

    For s in [e_n, e_{n+1}) (e_n = patch * n + patch - 1): v_{n-1} + (s - e_n) / patch * (v_n - v_{n-1}), with
    v_{-1} = v_0. Before e_0 and after the last patch end the first / last value is held.
    """
    b, n, k = values.shape
    pos = (out_idx.to(values.dtype) - (patch - 1)) / patch  # patch-end units: e_n -> n
    cur = pos.floor().clamp(0, n - 1)
    frac = (pos - cur).clamp(0, 1)
    frac = torch.where(pos < 0, torch.zeros_like(frac), frac)
    hi = cur.long()
    lo = (hi - 1).clamp(min=0)
    gather = lambda i: torch.gather(values, 1, i.unsqueeze(-1).expand(-1, -1, k))  # noqa: E731
    v_lo, v_hi = gather(lo), gather(hi)
    return v_lo + frac.unsqueeze(-1) * (v_hi - v_lo)  # frac = 1 from one patch after the last patch end: hold


def clock_phase(rate: torch.Tensor, out_idx: torch.Tensor, patch: int, fs: float, t0: float, tau0: float,
                s0: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clock-warp phase tracker (SSL-19).

    Args:
        rate: (B, N) clock rate k_n > 0 (1 = real time) from patch n's features.
        out_idx: (B, S) stream sample indices. t0: seconds since LED-on of stream sample 0 (the stream starts at
            LED-on - 1.5 s, so t0 = -1.5 + 1 / fs). tau0, s0: cue-phase centre and scale (s).

    Returns:
        phi at ``out_idx`` (B, S), phi at the patch ends (B, N), clock tau at ``out_idx`` (B, S). The clock is
        tau(e_0) = t0 + e_0 / fs, tau(e_{n+1}) = tau(e_n) + k_n * patch / fs, linear inside a patch; before e_0 it
        runs at real time.
    """
    dt = patch / fs
    tau_e0 = t0 + (patch - 1) / fs
    tau_end = tau_e0 + torch.cumsum(rate * dt, dim=1) - rate * dt  # tau at e_n
    n = rate.shape[1]
    pos = (out_idx.to(rate.dtype) - (patch - 1)) / patch
    cur = pos.floor().clamp(0, n - 1).long()
    since = ((pos - cur.to(pos.dtype)) * dt).clamp(min=0)  # seconds after e_cur (beyond the last patch: keeps going)
    tau = torch.gather(tau_end, 1, cur) + torch.gather(rate, 1, cur) * since
    tau = torch.where(pos < 0, tau_e0 + pos * dt, tau)  # before e_0: real time
    phi = ((tau - tau0) / s0).clamp(0.0, 1.0)
    return phi, ((tau_end - tau0) / s0).clamp(0.0, 1.0), tau


class OcularBranch(nn.Module):
    """EOG proxies (B, 2, history + L) -> (B, N, d_out). Causal convolutions over the stream, mean per patch."""

    def __init__(self, n_in: int = 2, channels: int = 16, kernel: int = 25, n_layers: int = 2, patch: int = 20,
                 d_out: int = 32):
        super().__init__()
        self.patch, self.kernel = patch, kernel
        layers, c = [], n_in
        for _ in range(n_layers):
            layers.append(nn.Conv1d(c, channels, kernel))
            c = channels
        self.convs = nn.ModuleList(layers)
        self.act = nn.ELU()
        self.proj = nn.Linear(channels, d_out)

    def forward(self, eog: torch.Tensor, history: int) -> torch.Tensor:
        h = eog
        for conv in self.convs:
            h = self.act(conv(F.pad(h, (self.kernel - 1, 0))))
        h = h[..., history:]
        b, c, t = h.shape
        h = h.reshape(b, c, t // self.patch, self.patch).mean(-1).transpose(1, 2)
        return self.proj(h)


class StreamDecoder(nn.Module):
    """Neural and/or ocular branch -> fusion -> head.

    Args:
        head: one of :data:`HEADS`. encoder: the stage 1 encoder (None: ocular-only). ocular: use the ocular branch.
        templates: TemplateBank of the fold (warp heads). freeze_encoder: frozen setting (encoder and gate fixed,
            kept in eval mode). tau0, s0: global-warp centre (s): median onset after LED-on and median onset -> hand
            stop duration of the fold's training lifts; also the proposed tracker's cue-phase centre and scale.
        a_max: |log A| bound. clock_log_range: c, the tracker's clock rate lies in [e^-c, e^c]. stream_start_s: seconds
        since LED-on at the stream start (SSL-17: -1.5). fs: sampling rate.
    """

    def __init__(self, head: str, encoder: CausalEncoder | None, ocular: bool, templates: TemplateBank | None = None,
                 freeze_encoder: bool = False, d_fused: int = 128, d_ocular: int = 32, ocular_channels: int = 16,
                 a_max: float = 0.5, tau0: float = 0.35, s0: float = 5.8, tau_range: float = 1.0,
                 s_log_range: float = 0.7, clock_log_range: float = math.log(2.0), stream_start_s: float = -1.5,
                 logvar_init: float = math.log(4.0), fs: float = 100.0, dropout: float = 0.1):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}")
        if encoder is None and not ocular:
            raise ValueError("need at least one branch")
        if head in WARP_HEADS and templates is None:
            raise ValueError(f"head {head!r} needs templates")
        self.head, self.encoder, self.templates, self.freeze_encoder = head, encoder, templates, freeze_encoder
        self.fs, self.a_max, self.tau0, self.s0 = fs, a_max, tau0, s0
        self.tau_range, self.s_log_range = tau_range, s_log_range
        self.clock_log_range, self.stream_start_s = clock_log_range, stream_start_s
        self.patch = encoder.patch if encoder is not None else 20
        self.window = encoder.window if encoder is not None else 200
        self.history = self.window - self.patch
        if encoder is not None and freeze_encoder:
            for q in encoder.parameters():
                q.requires_grad_(False)
        self.ocular = OcularBranch(channels=ocular_channels, patch=self.patch, d_out=d_ocular) if ocular else None
        d_in = (encoder.out_norm.normalized_shape[0] if encoder is not None else 0) + (d_ocular if ocular else 0)
        self.fusion = nn.Sequential(nn.Linear(d_in, d_fused), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(d_fused, d_fused), nn.LayerNorm(d_fused))
        n_cond = templates.values.shape[1] if templates is not None else 0
        self.out = nn.ModuleDict({"logvar": nn.Linear(d_fused, 3), "aux": nn.Linear(d_fused, len(AUX_CLASSES))})
        if head == "direct":
            self.out["y"] = nn.Linear(d_fused, 3)
        else:
            self.out["delta"] = nn.Linear(d_fused, 3)
        if head in WARP_HEADS:
            self.out["amp"] = nn.Linear(d_fused, 3)
            self.out["mix"] = nn.Linear(d_fused, n_cond)
        if head == "proposed":
            self.out["rate"] = nn.Linear(d_fused, 1)
        if head == "global_warp":
            self.out["warp"] = nn.Linear(d_fused, 2)
        self._init_outputs(logvar_init)

    def _init_outputs(self, logvar_init: float) -> None:
        for name, lin in self.out.items():
            if name in ("delta", "amp", "mix", "warp", "rate"):  # start: Delta 0, A 1, uniform, tau0 / s0, real-time clock
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.out["logvar"].weight)
        nn.init.constant_(self.out["logvar"].bias, logvar_init)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder is not None and self.freeze_encoder:
            self.encoder.eval()
        return self

    def neural_tokens(self, eeg: torch.Tensor) -> torch.Tensor:
        """Encoder token per patch (B, N, d): last token of the 2 s window ending at each patch's last sample."""
        b, c, t = eeg.shape
        if (t - self.history) % self.patch or t < self.window:
            raise ValueError("eeg stream must be history + a whole number of patches")
        win = eeg.unfold(-1, self.window, self.patch)  # (B, C, N, window)
        n = win.shape[2]
        win = win.permute(0, 2, 1, 3).reshape(b * n, c, self.window)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            tok = self.encoder(win)[:, -1]
        return tok.reshape(b, n, -1)

    def features(self, eeg: torch.Tensor | None, eog: torch.Tensor | None,
                 tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Fused per-patch features (B, N, d_fused). eeg (B, 32, history + 20 N), eog (B, 2, history + 20 N).
        ``tokens``: precomputed neural tokens (B, N, d) instead of ``eeg`` (frozen setting only)."""
        parts = []
        if self.encoder is not None and tokens is not None:
            if not self.freeze_encoder:
                raise ValueError("precomputed tokens are only valid for a frozen encoder")
            parts.append(tokens)
        elif self.encoder is not None:
            parts.append(self.neural_tokens(eeg))
        if self.ocular is not None:
            parts.append(self.ocular(eog, self.history))
        return self.fusion(torch.cat(parts, dim=-1))

    def forward(self, eeg: torch.Tensor | None, eog: torch.Tensor | None, out_idx: torch.Tensor,
                t_since_led: torch.Tensor | None = None, cue: torch.Tensor | None = None,
                phi_override: torch.Tensor | None = None, weights_override: torch.Tensor | None = None,
                tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Decode a batch of lift streams.

        Args:
            out_idx: (B, S) stream sample indices to predict (scored samples; padding may hold any valid index).
            t_since_led: (B, S) seconds since LED-on of each output sample (``global_warp``).
            cue: (B, S, 3) template_cue at each output sample (``residual``).
            phi_override, weights_override: (B, S) and (B, S, C) for tests (warp heads).

        Returns:
            ``y`` (B, S, 3), ``logvar`` (B, S, 3), ``aux_logits`` (B, N, 5), and for the other heads ``delta_patch``
            (B, N, 3); warp heads also ``phi`` (B, S), ``weights`` (B, S, C), ``log_amp_patch`` (B, N, 3),
            ``amp`` (B, S, 3); the tracker ``phi_patch`` (B, N) and clock ``rate`` (B, N).
        """
        h = self.features(eeg, eog, tokens)
        p = self.patch
        out: dict[str, torch.Tensor] = {"aux_logits": self.out["aux"](h)}
        out["logvar"] = causal_interp(self.out["logvar"](h).clamp(-8.0, 8.0), out_idx, p)
        if self.head == "direct":
            out["y"] = causal_interp(self.out["y"](h), out_idx, p)
            return out
        delta_patch = self.out["delta"](h)
        out["delta_patch"] = delta_patch
        delta = causal_interp(delta_patch, out_idx, p)
        if self.head == "residual":
            if cue is None:
                raise ValueError("residual head needs template_cue values")
            out["y"] = cue + delta
            return out
        log_amp_patch = self.a_max * torch.tanh(self.out["amp"](h))
        out["log_amp_patch"] = log_amp_patch
        amp = torch.exp(causal_interp(log_amp_patch, out_idx, p))
        weights = causal_interp(torch.softmax(self.out["mix"](h), dim=-1), out_idx, p)
        if self.head == "proposed":
            rate = torch.exp(self.clock_log_range * torch.tanh(self.out["rate"](h).squeeze(-1)))
            phi, phi_patch, _ = clock_phase(rate, out_idx, p, self.fs, self.stream_start_s + 1.0 / self.fs, self.tau0,
                                            self.s0)
            out["rate"], out["phi_patch"] = rate, phi_patch
        else:
            if t_since_led is None:
                raise ValueError("global_warp head needs t_since_led")
            w = self.out["warp"](h)
            tau = causal_interp((self.tau0 + self.tau_range * torch.tanh(w[..., :1])), out_idx, p).squeeze(-1)
            scale = causal_interp(self.s0 * torch.exp(self.s_log_range * torch.tanh(w[..., 1:])), out_idx, p).squeeze(-1)
            phi = ((t_since_led - tau) / scale).clamp(0.0, 1.0)
        if phi_override is not None:
            phi = phi_override
        if weights_override is not None:
            weights = weights_override
        out.update({"phi": phi, "weights": weights, "amp": amp})
        out["y"] = amp * self.templates(phi, weights) + delta
        return out

    def gate_penalty(self) -> torch.Tensor:
        if self.encoder is None or self.freeze_encoder:
            return torch.zeros(())
        return self.encoder.gate.prior_penalty()


def decoder_loss(out: dict[str, torch.Tensor], y: torch.Tensor, valid: torch.Tensor, aux_target: torch.Tensor | None,
                 gate_penalty: torch.Tensor, weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """L = NLL_Gaussian + lam_aux CE(phase) + lam_delta ||Delta||^2 + lam_A ||log A||^2 + lam_gate ||g - g_prior||^2.

    Args:
        y: (B, S, 3) targets; valid: (B, S) bool (scored, non-padding samples). aux_target: (B, N) auxiliary class per
            patch, -1 = ignore (padding or no label); None skips the term (validation / inference).
        weights: ``aux``, ``delta``, ``amp``, ``gate``.
    """
    v = valid.to(y.dtype).unsqueeze(-1)
    n = v.sum().clamp(min=1) * y.shape[-1]
    logvar = out["logvar"].float()
    nll = 0.5 * (logvar + (y.float() - out["y"].float()) ** 2 * torch.exp(-logvar) + math.log(2 * math.pi))
    terms = {"nll": (nll * v).sum() / n}
    zero = torch.zeros((), device=y.device)
    terms["aux"] = (F.cross_entropy(out["aux_logits"].float().reshape(-1, len(AUX_CLASSES)), aux_target.reshape(-1),
                                    ignore_index=-1) if aux_target is not None and (aux_target >= 0).any() else zero)
    terms["delta"] = (out["delta_patch"].float() ** 2).mean() if "delta_patch" in out else zero
    terms["amp"] = (out["log_amp_patch"].float() ** 2).mean() if "log_amp_patch" in out else zero
    terms["gate"] = gate_penalty.to(y.device).float()
    terms["loss"] = (terms["nll"] + weights.get("aux", 0.0) * terms["aux"] + weights.get("delta", 0.0) * terms["delta"]
                     + weights.get("amp", 0.0) * terms["amp"] + weights.get("gate", 0.0) * terms["gate"])
    return terms
