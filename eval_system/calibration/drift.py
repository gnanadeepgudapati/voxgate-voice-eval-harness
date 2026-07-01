"""Golden-set drift detection: KS-test a judge's current score distribution
against a frozen baseline. A judge that starts scoring systematically
differently (model update, prompt change, provider-side regression) needs
re-calibration before it's trusted again -- this is the tripwire."""
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import ks_2samp

DRIFT_P_VALUE_THRESHOLD = 0.05
MIN_SAMPLES = 2


@dataclass
class DriftResult:
    metric_name: str
    p_value: float
    statistic: float
    drifted: bool
    n_golden: int
    n_current: int


def detect_drift(
    metric_name: str,
    golden_scores: list[float],
    current_scores: list[float],
    *,
    p_value_threshold: float = DRIFT_P_VALUE_THRESHOLD,
) -> DriftResult:
    n_golden, n_current = len(golden_scores), len(current_scores)
    if n_golden < MIN_SAMPLES or n_current < MIN_SAMPLES:
        return DriftResult(metric_name, p_value=1.0, statistic=0.0, drifted=False, n_golden=n_golden, n_current=n_current)

    result = ks_2samp(golden_scores, current_scores)
    return DriftResult(
        metric_name,
        p_value=float(result.pvalue),
        statistic=float(result.statistic),
        drifted=bool(result.pvalue < p_value_threshold),
        n_golden=n_golden,
        n_current=n_current,
    )
