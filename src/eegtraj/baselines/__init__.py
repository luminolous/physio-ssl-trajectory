"""eegtraj.baselines subpackage. Importing it registers every baseline."""

from eegtraj.baselines import deep, kalman, ridge, stereotypy, template  # noqa: F401  (registers baselines)
from eegtraj.baselines.base import REGISTRY, Baseline, FitContext, make_baseline, register

__all__ = ["REGISTRY", "Baseline", "FitContext", "make_baseline", "register"]
