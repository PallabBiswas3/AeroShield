"""Small, cached Open-Meteo adapter for real forecast meteorology."""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from time import monotonic
import math

import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE = {}
_LOCK = Lock()
_TTL_SECONDS = 15 * 60
_FAILURES = {}


def _parse_local_hour(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(minute=0, second=0, microsecond=0)


def get_forecast_weather(lat: float, lon: float, when: datetime) -> dict:
    """Return forecast wind and mixing height for the requested local hour."""
    if when.tzinfo is not None:
        # API timestamps are requested in Asia/Kolkata. Callers must provide a
        # local wall-clock value or an aware datetime already in that zone;
        # dropping another zone's offset would select the wrong forecast hour.
        offset = when.utcoffset()
        if offset is None or offset.total_seconds() != 19800:
            raise ValueError("forecast time must be naive Asia/Kolkata local time or use UTC+05:30")
    key = (round(lat, 3), round(lon, 3))
    now = monotonic()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _TTL_SECONDS:
            payload = cached[1]
            fetched_at = cached[2]
        else:
            if key in _FAILURES and now < _FAILURES[key]:
                raise RuntimeError('Weather provider unavailable; retry cooldown active')
            try:
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
            except requests.RequestException as exc:
                _FAILURES[key] = monotonic() + 60
                raise RuntimeError('Weather provider unavailable or rate-limited; using scenario wind') from exc
            fetched_at = datetime.now(timezone.utc).isoformat()
            _CACHE[key] = (monotonic(), payload, fetched_at)

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError("forecast service returned no hourly data")
    target = when.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    parsed = [_parse_local_hour(value) for value in times]
    index = min(range(len(parsed)), key=lambda i: abs((parsed[i] - target).total_seconds()))
    if parsed[index] != target:
        raise ValueError("requested time is outside the available forecast horizon")

    def value(name: str):
        values = hourly.get(name, [])
        if index >= len(values) or values[index] is None:
            return None
        return float(values[index])

    speed, direction = value('wind_speed_10m'), value('wind_direction_10m')
    if speed is None or direction is None or not math.isfinite(speed) or not math.isfinite(direction) or not 0 <= speed <= 75 or not 0 <= direction <= 360:
        raise RuntimeError('Incomplete or invalid forecast wind; using scenario wind')
    return {
        "forecast_time": times[index],
        "wind_speed": speed,
        "wind_direction": direction % 360,
        "boundary_layer_height": value("boundary_layer_height"),
        "source": "Open-Meteo 16-day forecast",
        "fetched_at": fetched_at,
    }
