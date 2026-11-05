"""eegtraj.metrics subpackage: trajectory metrics and D7 aggregation."""

from eegtraj.metrics.trajectory import (
    METRIC_DEFINITIONS,
    aggregate_across_subjects,
    aggregate_subject,
    nrmse,
    pearson_r,
    per_trial_metrics,
    pooled_r,
    residual_r,
    residual_r_other_trial,
    r2_score,
    rmse,
)

__all__ = [
    "METRIC_DEFINITIONS",
    "aggregate_across_subjects",
    "aggregate_subject",
    "nrmse",
    "pearson_r",
    "per_trial_metrics",
    "pooled_r",
    "residual_r",
    "residual_r_other_trial",
    "r2_score",
    "rmse",
]
