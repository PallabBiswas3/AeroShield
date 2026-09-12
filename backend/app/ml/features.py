"""Shared physical and temporal feature calculations for AeroShield.

Keeping these functions in one module prevents training/serving skew. All
angles follow the meteorological convention used by the existing pipeline:
wind direction is where the wind comes *from*, in degrees clockwise from north.
"""
from __future__ import annotations

import math


PG_SIGMA_Y = (0.22, 0.16, 0.11, 0.08, 0.06, 0.04)
PG_SIGMA_Z = (0.20, 0.12, 0.08, 0.06, 0.03, 0.016)


def stability_class(wind_speed: float, hour: int) -> int:
    """Return a coarse Pasquill-Gifford stability class (A=0 … F=5)."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if wind_speed < 0:
        raise ValueError("wind_speed cannot be negative")
    if 6 <= hour <= 18:
        if wind_speed < 2:
            return 0
        if wind_speed < 3:
            return 1
        if wind_speed < 5:
            return 2
        return 3
    if wind_speed < 2:
        return 5
    if wind_speed < 3:
        return 4
    return 3


def estimate_boundary_layer_height(hour: int, stability: int) -> float:
    """Return the fallback mixing height used when a forecast is unavailable."""
    if not 0 <= stability <= 5:
        raise ValueError("stability must be an integer between 0 and 5")
    daylight = max(0.1, 0.5 + 0.5 * math.sin(math.pi * (hour - 6) / 12))
    stability_factor = (5 - stability) / 5.0
    return 200.0 + 1600.0 * daylight * stability_factor


def diurnal_traffic_factor(hour: int, day_of_week: int) -> float:
    """Traffic proxy with pandas/Python weekday convention (Monday=0)."""
    if not 0 <= day_of_week <= 6:
        raise ValueError("day_of_week must use Monday=0 through Sunday=6")
    if day_of_week < 5:
        return 0.2 + 0.8 * (
            0.8 * math.exp(-0.5 * ((hour - 8.5) / 1.2) ** 2)
            + math.exp(-0.5 * ((hour - 18.0) / 1.3) ** 2)
        )
    return 0.3 + 0.5 * math.exp(-0.5 * ((hour - 11.5) / 2.0) ** 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    radius_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def circular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute separation of two headings."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
