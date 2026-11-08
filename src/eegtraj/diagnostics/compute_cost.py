"""Compute cost of a decoder per input-channel count: parameters, inference latency, inference memory.

No training and no data: the model gets random input of the right shape. Latency is the wall time of one forward
pass for a single window (batch 1, the streaming case), after warm-up; on GPU each pass is synchronized.
The residual form uses the same network (only the target changes), so its cost equals the direct form.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from eegtraj.baselines.deep import ARCHITECTURES
from eegtraj.models.trainer import count_parameters


@torch.no_grad()
def forward_latency_ms(model: torch.nn.Module, x: torch.Tensor, n_warmup: int, n_repeat: int) -> np.ndarray:
    """Wall time (ms) of ``n_repeat`` forward passes on ``x``, after ``n_warmup`` passes."""
    cuda = x.device.type == "cuda"
    for _ in range(n_warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    times = np.empty(n_repeat)
    for i in range(n_repeat):
        t0 = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


def montage_cost(architecture: str, model_kwargs: dict, n_chans: int, n_times: int, devices: list[str],
                 n_warmup: int, n_repeat: int, seed: int) -> dict:
    """Parameters plus median / p95 latency per device and peak GPU memory of one batch-1 forward pass."""
    torch.manual_seed(seed)
    model = ARCHITECTURES[architecture](n_chans=n_chans, n_times=n_times, **model_kwargs).eval()
    row: dict = {"n_chans": n_chans, "n_params": count_parameters(model),
                 "n_params_spatial": sum(p.numel() for n, p in model.named_parameters() if n.startswith("spatial"))}
    for device in devices:
        m = model.to(device)
        x = torch.randn(1, n_chans, n_times, device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t = forward_latency_ms(m, x, n_warmup, n_repeat)
        row[f"latency_{device}_median_ms"] = float(np.median(t))
        row[f"latency_{device}_p95_ms"] = float(np.percentile(t, 95))
        if device == "cuda":
            row["inference_peak_gpu_mb"] = (torch.cuda.max_memory_allocated() - base) / 1e6
    return row
