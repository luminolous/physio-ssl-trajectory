"""Tests for eegtraj.utils.run."""

from __future__ import annotations

import logging
from datetime import datetime

import pytest
import yaml

from eegtraj.utils.run import make_run_id, setup_run, teardown_logging


@pytest.fixture(autouse=True)
def _close_log_handlers():
    yield
    # Close run.log before tmp_path cleanup (Windows cannot delete open files).
    teardown_logging()


def test_make_run_id_format():
    now = datetime(2026, 10, 30, 14, 25, 1)
    assert make_run_id(now=now) == "20261030-142501"
    assert make_run_id("lowfreq", now=now) == "20261030-142501_lowfreq"


def test_setup_run_creates_expected_files(tmp_path):
    config = {"seed": 7, "model": {"name": "ridge", "alpha": 1.0}}
    run_dir = setup_run(tmp_path / "baselines" / "ridge-lag", config, tag="smoke")

    assert run_dir.parent == tmp_path / "baselines" / "ridge-lag"
    assert run_dir.name.endswith("_smoke")
    for name in ("config.yaml", "env.txt", "run.log"):
        assert (run_dir / name).is_file(), name

    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved == config

    env = (run_dir / "env.txt").read_text(encoding="utf-8")
    for key in ("python:", "torch_cuda_build:", "git_commit:", "git_dirty:", "packages:"):
        assert key in env

    logging.getLogger("eegtraj.test").info("hello from test")
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Run directory" in log_text
    assert "hello from test" in log_text


def test_setup_run_never_overwrites(tmp_path):
    setup_run(tmp_path, {"a": 1}, run_id="fixed")
    with pytest.raises(FileExistsError):
        setup_run(tmp_path, {"a": 2}, run_id="fixed")


def test_repeated_setup_does_not_stack_handlers(tmp_path):
    first = setup_run(tmp_path, {}, run_id="first")
    second = setup_run(tmp_path, {}, run_id="second")
    logging.getLogger("eegtraj.test").info("only in second")

    assert "only in second" not in (first / "run.log").read_text(encoding="utf-8")
    assert "only in second" in (second / "run.log").read_text(encoding="utf-8")
