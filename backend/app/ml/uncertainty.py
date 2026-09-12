"""Distribution-free calibration helpers for probabilistic PM2.5 forecasts."""
from __future__ import annotations

from typing import Any

import numpy as np


def conformal_quantile(scores: np.ndarray, coverage: float = 0.90) -> float:
    """Finite-sample split-conformal quantile using the conservative rank."""
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("at least one finite calibration score is required")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between 0 and 1")
    rank = int(np.ceil((values.size + 1) * coverage))
    rank = min(max(rank, 1), values.size)
    return float(np.sort(values)[rank - 1])


def calibrate_interval(y_true, lower, upper, coverage: float = 0.90) -> float:
    """Return the conformal expansion needed around a base quantile band."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have matching shapes")
    scores = np.maximum(lo - y, y - hi)
    return max(0.0, conformal_quantile(scores, coverage))


def apply_conformal(lower, upper, qhat: float, minimum: float = 0.0, maximum: float = 500.0):
    """Expand and clip a prediction interval."""
    if qhat < 0:
        raise ValueError("qhat cannot be negative")
    lo = np.clip(np.asarray(lower, dtype=float) - qhat, minimum, maximum)
    hi = np.clip(np.asarray(upper, dtype=float) + qhat, minimum, maximum)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return {
        "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
    }


def predict_interval(bundle: dict[str, Any], features):
    """Predict conformalized lower/median/upper values from a saved bundle."""
    required = {"lower_model", "median_model", "upper_model", "qhat"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"uncertainty bundle is missing: {sorted(missing)}")
    lo = np.asarray(bundle["lower_model"].predict(features), dtype=float)
    median = np.asarray(bundle["median_model"].predict(features), dtype=float)
    hi = np.asarray(bundle["upper_model"].predict(features), dtype=float)
    lo, hi = apply_conformal(lo, hi, float(bundle["qhat"]))
    median = np.clip(median, lo, hi)
    return lo, median, hi
