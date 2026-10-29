"""GPU training loop for seq-to-one deep decoders (baselines.md, shared deep setup).

Inputs are causal EEG windows (n_channels, history) ending at the target sample;
the target is the normalized 3D position at that sample. Normalized EEG of all
subjects in a split is kept on the GPU and windows are gathered per batch, which
avoids a Python call per window (a LOSO training set has ~1.3 M windows).

Setup: MSE loss, AdamW, mixed precision (CUDA), early stopping on the inner
validation loss with patience, best-epoch weights restored. Each epoch draws a
random subset of training windows (consecutive 100 Hz windows are highly redundant).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from eegtraj.data.windows import Standardizer, SubjectSamples

logger = logging.getLogger(__name__)


class GpuWindows:
    """Normalized EEG/targets of several subjects on one device, with batched window gathering."""

    def __init__(self, subjects: list[SubjectSamples], masks: dict[str, np.ndarray],
                 eeg_norm: dict[str, Standardizer], target_norm: Standardizer, history: int,
                 device: torch.device):
        eeg_parts, tgt_parts, pos_parts = [], [], []
        offset = 0
        self.history = history
        self.device = device
        for s in subjects:
            m = masks[s.subject]
            if not m.any():
                continue
            eeg_parts.append(eeg_norm[s.subject].transform(s.eeg))
            tgt_parts.append(target_norm.transform(s.target))
            pos_parts.append(s.sample_idx[m] + offset)
            offset += s.eeg.shape[1]
        self.eeg = torch.from_numpy(np.concatenate(eeg_parts, axis=1)).to(device)  # (C, T_total)
        self.target = torch.from_numpy(np.concatenate(tgt_parts, axis=1)).to(device)  # (3, T_total)
        self.positions = torch.from_numpy(np.concatenate(pos_parts)).to(device)  # (n,) window end indices
        self._steps = torch.arange(-history + 1, 1, device=device)

    def __len__(self) -> int:
        return len(self.positions)

    def batch(self, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows (B, C, history) and targets (B, 3) for window numbers ``k`` (indices into positions)."""
        end = self.positions[k]
        cols = end[:, None] + self._steps[None, :]  # (B, H)
        x = self.eeg[:, cols].permute(1, 0, 2)  # (B, C, H)
        return x, self.target[:, end].T


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 512
    max_epochs: int = 30
    samples_per_epoch: int = 200_000
    val_samples: int = 100_000  # fixed random subset of validation windows scored each epoch
    patience: int = 5
    amp: bool = True


def _amp_ctx(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, windows: GpuWindows, k: torch.Tensor, cfg: TrainConfig) -> float:
    model.eval()
    total, n = 0.0, 0
    for start in range(0, len(k), cfg.batch_size * 4):
        kb = k[start:start + cfg.batch_size * 4]
        x, y = windows.batch(kb)
        with _amp_ctx(windows.device, cfg.amp):
            out = model(x)
        total += torch.nn.functional.mse_loss(out.float(), y, reduction="sum").item()
        n += y.numel()
    return total / n


def train_model(model: torch.nn.Module, train: GpuWindows, val: GpuWindows, cfg: TrainConfig, seed: int,
                log: logging.Logger = logger) -> tuple[torch.nn.Module, list[dict]]:
    """Train with early stopping on validation MSE; returns the model with best weights and per-epoch history."""
    device = train.device
    gen = torch.Generator(device=device).manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    val_k = torch.randperm(len(val), generator=gen, device=device)[:cfg.val_samples]
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        k_all = torch.randint(len(train), (min(cfg.samples_per_epoch, len(train)),), generator=gen, device=device)
        total, n = 0.0, 0
        for start in range(0, len(k_all), cfg.batch_size):
            x, y = train.batch(k_all[start:start + cfg.batch_size])
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device, cfg.amp):
                loss = torch.nn.functional.mse_loss(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        val_loss = evaluate_loss(model, val, val_k, cfg)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "seconds": round(time.time() - t0, 2)})
        log.info("epoch %d: train MSE %.4f, val MSE %.4f (%.1f s)", epoch, total / n, val_loss, time.time() - t0)
        if val_loss < best:
            best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stop at epoch %d (best val MSE %.4f)", epoch, best)
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_windows(model: torch.nn.Module, windows: GpuWindows, cfg: TrainConfig) -> np.ndarray:
    """Normalized predictions (n, 3) for every window, in order."""
    model.eval()
    outs = []
    k = torch.arange(len(windows), device=windows.device)
    for start in range(0, len(k), cfg.batch_size * 4):
        x, _ = windows.batch(k[start:start + cfg.batch_size * 4])
        with _amp_ctx(windows.device, cfg.amp):
            outs.append(model(x).float().cpu())
    return torch.cat(outs).numpy()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
