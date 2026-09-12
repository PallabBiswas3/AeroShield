"""Small, cached Open-Meteo adapter for real forecast meteorology."""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from time import monotonic

import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE = {}
_LOCK = Lock()
_TTL_SECONDS = 15 * 60


def _parse_local_hour(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(minute=0, second=0, microsecond=0)


def get_forecast_weather(lat: float, lon: float, when: datetime) -> dict:
    """Return forecast wind and mixing-height values nearest to ``when``."""
    key = (round(lat, 3), round(lon, 3))
    now = monotonic()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _TTL_SECONDS:
            payload = cached[1]
        else:
            response = requests.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "wind_speed_10m,wind_direction_10m,boundary_layer_height",
                    "wind_speed_unit": "ms",
                    "timezone": "Asia/Kolkata",
                    "forecast_days": 16,
                },
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
            _CACHE[key] = (now, payload)

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError("forecast service returned no hourly data")
    target = when.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    parsed = [_parse_local_hour(value) for value in times]
    index = min(range(len(parsed)), key=lambda i: abs((parsed[i] - target).total_seconds()))
    if abs((parsed[index] - target).total_seconds()) > 3600:
        raise ValueError("requested time is outside the available forecast horizon")

    def value(name: str):
        values = hourly.get(name, [])
        if index >= len(values) or values[index] is None:
            return None
        return float(values[index])

    return {
        "forecast_time": times[index],
        "wind_speed": value("wind_speed_10m"),
        "wind_direction": value("wind_direction_10m"),
        "boundary_layer_height": value("boundary_layer_height"),
        "source": "Open-Meteo 16-day forecast",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
