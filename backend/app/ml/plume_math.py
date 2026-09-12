"""Reverse-plume source attribution with explicit meteorological uncertainty."""
from __future__ import annotations

from datetime import datetime
import math

import numpy as np

from app.ml.features import PG_SIGMA_Y, PG_SIGMA_Z, haversine_km, stability_class


def _concentration_scores(hotspot_lat, hotspot_lon, wind_speed, wind_dir_deg, sources_list, hour, intensity_multipliers=None):
    """Return non-normalized Gaussian-plume compatibility scores."""
    wind_to_rad = np.radians((wind_dir_deg + 180.0) % 360.0)
    # East/north vector for a meteorological bearing measured clockwise from north.
    wind_vec = np.array([np.sin(wind_to_rad), np.cos(wind_to_rad)])
    stability = stability_class(wind_speed, hour)
    speed = max(float(wind_speed), 0.5)
    multipliers = intensity_multipliers
    if multipliers is None:
        multipliers = np.ones(len(sources_list), dtype=float)

    scores = np.zeros(len(sources_list), dtype=float)
    for index, source in enumerate(sources_list):
        dx_km = (hotspot_lon - source["lon"]) * 111.0 * np.cos(np.radians(hotspot_lat))
        dy_km = (hotspot_lat - source["lat"]) * 111.0
        displacement = np.array([dx_km, dy_km], dtype=float)
        downwind = float(np.dot(wind_vec, displacement))
        if downwind <= 0.1:
            continue
        crosswind = float(np.linalg.norm(displacement - downwind * wind_vec))
        sigma_y = max(PG_SIGMA_Y[stability] * (downwind ** 0.894), 0.5)
        sigma_z = max(PG_SIGMA_Z[stability] * (downwind ** 0.894), 0.3)
        intensity = float(source.get("intensity", 400.0)) * float(multipliers[index])
        scores[index] = intensity / (np.pi * speed * sigma_y * sigma_z) * np.exp(-0.5 * (crosswind / sigma_y) ** 2)
    return scores


def calculate_source_attribution_ensemble(
    hotspot_lat: float,
    hotspot_lon: float,
    wind_speed: float,
    wind_dir_deg: float,
    sources_list: list,
    hour: int | None = None,
    simulations: int = 600,
    wind_direction_sigma: float = 15.0,
    wind_speed_relative_sigma: float = 0.18,
    source_intensity_sigma: float = 0.30,
    seed: int = 42,
) -> list:
    """Monte-Carlo attribution probabilities under wind/inventory uncertainty.

    These probabilities are model-based source weights, not legal confidence
    and not proof that a source violated an environmental standard.
    """
    if hour is None:
        hour = datetime.now().hour
    if simulations < 50:
        raise ValueError("simulations must be at least 50")
    numeric_inputs = (hotspot_lat, hotspot_lon, wind_speed, wind_dir_deg,
                      wind_direction_sigma, wind_speed_relative_sigma, source_intensity_sigma)
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        raise ValueError("attribution inputs must be finite")
    if wind_speed < 0 or wind_direction_sigma < 0 or wind_speed_relative_sigma < 0 or source_intensity_sigma < 0:
        raise ValueError("wind speed and uncertainty scales cannot be negative")
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not sources_list:
        return []

    rng = np.random.default_rng(seed)
    n_sources = len(sources_list)
    shares = np.zeros((simulations, n_sources), dtype=float)
    rank_one = np.zeros(n_sources, dtype=int)
    active_simulations = 0

    for sample in range(simulations):
        sampled_direction = float(rng.normal(wind_dir_deg, wind_direction_sigma) % 360.0)
        sampled_speed = max(0.5, float(rng.normal(wind_speed, max(0.15, wind_speed * wind_speed_relative_sigma))))
        multipliers = rng.lognormal(mean=-0.5 * source_intensity_sigma ** 2, sigma=source_intensity_sigma, size=n_sources)
        raw = _concentration_scores(hotspot_lat, hotspot_lon, sampled_speed, sampled_direction, sources_list, hour, multipliers)
        total = float(raw.sum())
        if total > 0:
            shares[sample] = raw / total
            rank_one[int(np.argmax(raw))] += 1
            active_simulations += 1

    active = shares[np.sum(shares, axis=1) > 0]
    if active.size == 0:
        active = shares
    results = []
    for index, source in enumerate(sources_list):
        distribution = active[:, index] * 100.0
        probability = float(np.mean(distribution)) if distribution.size else 0.0
        p10 = float(np.quantile(distribution, 0.10)) if distribution.size else 0.0
        p90 = float(np.quantile(distribution, 0.90)) if distribution.size else 0.0
        rank_probability = float(rank_one[index] / active_simulations * 100.0) if active_simulations else 0.0
        active_fraction = active_simulations / simulations
        grade = "strong" if active_fraction >= 0.8 and rank_probability >= 70 and p10 >= 20 else "moderate" if active_fraction >= 0.5 and rank_probability >= 40 else "weak"
        results.append({
            **source,
            "distance_km": round(haversine_km(hotspot_lat, hotspot_lon, source["lat"], source["lon"]), 2),
            # These are relative shares conditional on at least one inventoried
            # source being plume-compatible, not calibrated source probabilities.
            "attribution_share": round(probability, 1),
            "attribution_probability": round(probability, 1), "probability_p10": round(p10, 1),
            "probability_p90": round(p90, 1), "rank_one_probability": round(rank_probability, 1),
            "confidence_score": round(probability, 1), "evidence_grade": grade,
            "active_simulation_fraction": round(active_fraction, 3),
            "active_simulations": active_simulations,
            "method": "Monte Carlo reverse Gaussian plume",
            "interpretation": "Relative contribution share among inventoried sources, conditional on plume compatibility.",
        })
    return sorted(results, key=lambda item: item["attribution_probability"], reverse=True)


def calculate_source_attribution(hotspot_lat, hotspot_lon, wind_speed, wind_dir_deg, sources_list, hour=None) -> list:
    """Backward-compatible entry point for the uncertainty-aware estimator."""
    return calculate_source_attribution_ensemble(
        hotspot_lat=hotspot_lat, hotspot_lon=hotspot_lon, wind_speed=wind_speed,
        wind_dir_deg=wind_dir_deg, sources_list=sources_list, hour=hour,
    )
