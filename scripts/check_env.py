"""Print environment info and run a small GPU sanity check.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
import time

PACKAGES = [
    "eegtraj",
    "numpy",
    "scipy",
    "h5py",
    "pandas",
    "matplotlib",
    "sklearn",
    "yaml",
    "tqdm",
    "pytest",
    "mne",
    "torch",
    "braindecode",
]

MATMUL_SIZE = 2048


def main() -> int:
    print(f"Python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform : {platform.platform()}")
    print()

    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<12} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # report every failing import, not only the first
            ok = False
            print(f"{name:<12} IMPORT FAILED: {exc!r}")
    print()

    import torch

    cuda = torch.cuda.is_available()
    print(f"torch.version.cuda     : {torch.version.cuda}")
    print(f"torch.cuda.is_available: {cuda}")
    if not cuda:
        print("CUDA NOT AVAILABLE")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU name               : {props.name}")
    print(f"Total VRAM             : {props.total_memory / 1024**3:.2f} GiB")
    print(f"cuDNN                  : {torch.backends.cudnn.version()}")

    a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda")
    b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    ref = (a.cpu() @ b.cpu())
    max_err = (c.cpu() - ref).abs().max().item()
    print(f"GPU matmul {MATMUL_SIZE}x{MATMUL_SIZE}: {dt_ms:.1f} ms, max |GPU-CPU| = {max_err:.2e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
