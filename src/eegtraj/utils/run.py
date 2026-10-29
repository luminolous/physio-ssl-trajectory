"""Run directory creation, environment capture, and logging setup.

Every run gets its own new directory containing:

- ``config.yaml``: resolved config after CLI overrides;
- ``env.txt``: Python, torch, CUDA, GPU, package versions, git commit, dirty flag;
- ``run.log``: everything logged through :mod:`logging` (console + file).

tqdm progress bars write to stderr directly and therefore never reach ``run.log``.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CONFIG_FILENAME = "config.yaml"
ENV_FILENAME = "env.txt"
LOG_FILENAME = "run.log"

ENV_PACKAGES = (
    "numpy",
    "scipy",
    "h5py",
    "pandas",
    "scikit-learn",
    "mne",
    "torch",
    "braindecode",
    "pyyaml",
)

# Handlers installed by setup_logging, so repeated calls replace rather than stack them.
_installed_handlers: list[logging.Handler] = []


def make_run_id(tag: str | None = None, now: datetime | None = None) -> str:
    """Return ``<YYYYmmdd-HHMMSS>`` or ``<YYYYmmdd-HHMMSS>_<tag>``."""
    stamp = (now or datetime.now()).strftime(RUN_ID_FORMAT)
    return f"{stamp}_{tag}" if tag else stamp


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def collect_env_info(repo_root: Path | None = None) -> str:
    """Return a human-readable environment report for ``env.txt``."""
    repo_root = repo_root or Path.cwd()
    lines = [
        f"python: {sys.version.split()[0]}",
        f"executable: {sys.executable}",
        f"platform: {platform.platform()}",
    ]

    try:
        import torch

        lines.append(f"torch_cuda_build: {torch.version.cuda}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            lines.append(f"gpu: {props.name}")
            lines.append(f"gpu_vram_gib: {props.total_memory / 1024**3:.2f}")
    except ImportError:
        lines.append("torch: not installed")

    lines.append("packages:")
    for name in ENV_PACKAGES:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"  {name}: {version}")

    commit = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    lines.append(f"git_commit: {commit or 'unknown'}")
    lines.append(f"git_dirty: {'unknown' if status is None else bool(status)}")
    return "\n".join(lines) + "\n"


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Log to console (stdout) and ``log_path``. Replaces handlers from earlier calls."""
    teardown_logging()
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _installed_handlers.append(handler)


def teardown_logging() -> None:
    """Remove and close handlers installed by :func:`setup_logging`."""
    root = logging.getLogger()
    while _installed_handlers:
        handler = _installed_handlers.pop()
        root.removeHandler(handler)
        handler.close()


def setup_run(
    base_dir: str | Path,
    config: dict[str, Any],
    tag: str | None = None,
    run_id: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Create a new run directory, save config and env info, and start logging.

    Args:
        base_dir: parent folder, e.g. ``output/data-baselines/baselines/ridge-lag``.
        config: resolved config (plain YAML-serializable dict).
        tag: optional suffix appended to the timestamp run ID.
        run_id: explicit run ID; overrides the timestamp (mainly for tests).
        repo_root: git working tree used for the commit hash; defaults to cwd.

    Returns:
        Path to the run directory ``<base_dir>/<run_id>``.

    Raises:
        FileExistsError: if the run directory already exists. Runs are never overwritten.
    """
    run_dir = Path(base_dir) / (run_id or make_run_id(tag))
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / CONFIG_FILENAME, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    (run_dir / ENV_FILENAME).write_text(collect_env_info(repo_root), encoding="utf-8")

    setup_logging(run_dir / LOG_FILENAME)
    logger.info("Run directory: %s", run_dir)
    return run_dir
