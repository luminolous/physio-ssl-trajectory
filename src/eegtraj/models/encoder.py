"""Neural-branch encoder of the proposed model (design.md, "Stage 1"): gate -> tokenizer -> causal transformer.

Input: X float (B, 32, 200), a 2 s window at 100 Hz (broadband, CAR, D9 z-score). Output: 10 tokens of d = 128, one
per 0.2 s patch. Token k depends only on samples up to the end of patch k (``20 k + 19``): the temporal convolution is
causal (left zero padding inside the window), pooling does not cross patches, and attention is causal.

- Gate: 32 sigmoid weights, initialized at the physiological prior g_prior (``configs/pretrain/physio_prior.yaml``);
  ``prior_penalty`` is ||g - g_prior||^2 for the L2 pull. Used on the context view only (``use_gate=False`` for the
  target branch).
- Tokenizer: causal temporal conv (kernel 25, 16 filters) -> depthwise spatial conv over the 32 channels (depth
  multiplier 2) -> ELU -> average pooling per patch (20 samples) -> linear to d -> LayerNorm per token. No BatchNorm:
  batch statistics over time would mix future samples during training.
- Encoder: pre-LN transformer, causal attention mask, learned positional embedding.

Streaming (design.md, "Streaming"): a lift is encoded as a stream of 0.2 s patches; the token of patch n attends to
the last 10 patches (2 s). ``stream`` cuts the 2 s window ending at every patch boundary and returns that window's
last token, so a streamed token equals the per-window output by construction.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelGate(nn.Module):
    """Per-channel weights g = sigmoid(logits), initialized at ``prior`` (values in (0, 1))."""

    def __init__(self, prior: torch.Tensor):
        super().__init__()
        prior = torch.as_tensor(prior, dtype=torch.float32)
        if not ((prior > 0) & (prior < 1)).all():
            raise ValueError("gate prior must lie strictly between 0 and 1")
        self.register_buffer("prior", prior.clone())
        self.logits = nn.Parameter(torch.log(prior / (1 - prior)))

    @property
    def weights(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights[:, None].to(x.dtype)

    def prior_penalty(self) -> torch.Tensor:
        return ((self.weights - self.prior) ** 2).sum()


class Tokenizer(nn.Module):
    """(B, C, T) -> (B, T // patch, d)."""

    def __init__(self, n_chans: int = 32, n_filters: int = 16, kernel: int = 25, depth_multiplier: int = 2,
                 patch: int = 20, d_model: int = 128):
        super().__init__()
        self.patch, self.kernel = patch, kernel
        self.temporal = nn.Conv2d(1, n_filters, (1, kernel), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters * depth_multiplier, (n_chans, 1), groups=n_filters, bias=False)
        self.act = nn.ELU()
        self.proj = nn.Linear(n_filters * depth_multiplier, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.patch:
            raise ValueError(f"window length {t} is not a multiple of the patch length {self.patch}")
        h = nn.functional.pad(x.unsqueeze(1), (self.kernel - 1, 0))  # causal: zeros before the window start
        h = self.act(self.spatial(self.temporal(h)))  # (B, F*D, 1, T)
        h = h.squeeze(2).reshape(b, h.shape[1], t // self.patch, self.patch).mean(-1)  # (B, F*D, n_patches)
        return self.norm(self.proj(h.transpose(1, 2)))


class CausalEncoder(nn.Module):
    """Gate + tokenizer + causal transformer. ``forward`` returns (B, n_kept, d) tokens."""

    def __init__(self, gate_prior: torch.Tensor, n_chans: int = 32, window: int = 200, patch: int = 20,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 4, mlp_width: int = 256, dropout: float = 0.1,
                 n_filters: int = 16, kernel: int = 25, depth_multiplier: int = 2):
        super().__init__()
        self.window, self.patch = window, patch
        self.n_tokens = window // patch
        self.gate = ChannelGate(gate_prior)
        self.tokenizer = Tokenizer(n_chans, n_filters, kernel, depth_multiplier, patch, d_model)
        self.pos = nn.Parameter(torch.zeros(self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, mlp_width, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None, use_gate: bool = True) -> torch.Tensor:
        """Encode windows.

        Args:
            x: (B, C, window). keep: optional (n_kept,) sorted token indices to feed to the transformer (context view
                with masked tokens dropped); positions stay those of the full window. use_gate: False for the target
                branch.

        Returns:
            (B, n_kept, d), or (B, n_tokens, d) without ``keep``.
        """
        if x.shape[-1] != self.window:
            raise ValueError(f"expected windows of {self.window} samples, got {x.shape[-1]}")
        h = self.tokenizer(self.gate(x) if use_gate else x) + self.pos
        if keep is not None:
            if (keep[1:] <= keep[:-1]).any():
                raise ValueError("keep must be strictly increasing")
            h = h[:, keep]
        n = h.shape[1]
        mask = torch.triu(torch.full((n, n), float("-inf"), device=h.device), diagonal=1)
        return self.out_norm(self.blocks(h, mask=mask, is_causal=True))

    def stream(self, eeg: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """Encode a continuous recording as a stream of patches.

        Args:
            eeg: (C, T) with T a multiple of the patch length; the first ``window`` samples are history.

        Returns:
            (n_out, d): one token per patch from patch ``window / patch - 1`` on; row i is the last token of the
            window ``eeg[:, i * patch : i * patch + window]``.
        """
        c, t = eeg.shape
        if t < self.window or t % self.patch:
            raise ValueError("stream length must be >= window and a multiple of the patch length")
        windows = eeg.unfold(1, self.window, self.patch).permute(1, 0, 2)  # (n_out, C, window)
        return torch.cat([self(windows[i:i + chunk])[:, -1] for i in range(0, windows.shape[0], chunk)])


def n_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
