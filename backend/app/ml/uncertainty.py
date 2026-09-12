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
    if y.size == 0:
        raise ValueError("calibration arrays cannot be empty")
    finite = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.all(finite):
        raise ValueError("calibration arrays must contain only finite values")
    # Quantile estimators can cross. Sorting each pair makes the base band
    # well-defined before computing the conformal nonconformity score.
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    scores = np.maximum(lo - y, y - hi)
    return max(0.0, conformal_quantile(scores, coverage))


def apply_conformal(lower, upper, qhat: float, minimum: float = 0.0, maximum: float = 500.0):
    """Expand and clip a prediction interval."""
    if not np.isfinite(qhat) or qhat < 0:
        raise ValueError("qhat cannot be negative")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
        raise ValueError("minimum and maximum must be finite and ordered")
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if lower_values.shape != upper_values.shape or not np.all(np.isfinite(lower_values)) or not np.all(np.isfinite(upper_values)):
        raise ValueError("lower and upper must be matching finite arrays")
    base_lo, base_hi = np.minimum(lower_values, upper_values), np.maximum(lower_values, upper_values)
    return np.clip(base_lo - qhat, minimum, maximum), np.clip(base_hi + qhat, minimum, maximum)


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape) or y.size == 0:
        raise ValueError("metric arrays must be non-empty with matching shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("metric arrays must contain only finite values")
    if np.any(lo > hi):
        raise ValueError("lower bounds cannot exceed upper bounds")
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
    expected_features = bundle.get("features")
    actual_features = list(getattr(features, "columns", []))
    if expected_features and actual_features != list(expected_features):
        raise ValueError("uncertainty feature schema does not match the serving frame")
    lo = np.asarray(bundle["lower_model"].predict(features), dtype=float)
    median = np.asarray(bundle["median_model"].predict(features), dtype=float)
    hi = np.asarray(bundle["upper_model"].predict(features), dtype=float)
    lo, hi = apply_conformal(lo, hi, float(bundle["qhat"]))
    median = np.clip(median, lo, hi)
    return lo, median, hi
