"""eegtraj.baselines subpackage. Importing it registers every baseline."""

from eegtraj.baselines import template  # noqa: F401  (registers "template")
from eegtraj.baselines.base import REGISTRY, Baseline, FitContext, make_baseline, register

__all__ = ["REGISTRY", "Baseline", "FitContext", "make_baseline", "register"]
