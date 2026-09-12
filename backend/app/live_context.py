"""Resilient live air-quality context for the AeroShield demo.

Open-Meteo/CAMS is a coarse atmospheric-model product. It is exposed as an
external operational baseline, never as AeroShield's hyperlocal ground truth.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from time import monotonic

import requests


AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
_LOCK = Lock()
_CACHE: dict[tuple[float, float], tuple[float, dict]] = {}
_TTL_SECONDS = 15 * 60
_ROOT = Path(__file__).resolve().parents[2]
_DISK_CACHE = _ROOT / "data" / "runtime_cache" / "live_air_quality.json"


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_disk_cache(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_context(payload: dict, fetched_at: str) -> dict:
    current = payload.get("current") or {}
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    pm25 = hourly.get("pm2_5") or []
    outlook = []
    for index, timestamp in enumerate(times[:25]):
        value = pm25[index] if index < len(pm25) else None
        if value is not None:
            outlook.append({"time": timestamp, "pm25": round(float(value), 1)})
    return {
        "current": {
            "time": current.get("time"),
            "pm25": current.get("pm2_5"),
            "pm10": current.get("pm10"),
            "us_aqi": current.get("us_aqi"),
            "nitrogen_dioxide": current.get("nitrogen_dioxide"),
        },
        "outlook_24h": outlook,
        "fetched_at": fetched_at,
        "provider": "Open-Meteo / CAMS global",
        "source_kind": "coarse_external_model_baseline",
        "spatial_resolution_note": "CAMS global air-quality forecasts are approximately 45 km, not AeroShield's 1 km output.",
    }


def get_live_air_quality(lat: float = 28.628, lon: float = 77.209) -> dict:
    """Return live/coarse AQ context, falling back to the last valid response."""
    key = (round(float(lat), 3), round(float(lon), 3))
    now = monotonic()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _TTL_SECONDS:
            return {**cached[1], "mode": "memory_cache", "stale": False}

    try:
        response = requests.get(
            AIR_QUALITY_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "pm2_5,pm10,us_aqi,nitrogen_dioxide",
                "hourly": "pm2_5",
                "forecast_hours": 25,
                "timezone": "Asia/Kolkata",
            },
            timeout=12,
        )
        response.raise_for_status()
        fetched_at = datetime.now(timezone.utc).isoformat()
        context = _build_context(response.json(), fetched_at)
        if context["current"]["pm25"] is None or not context["outlook_24h"]:
            raise RuntimeError("air-quality provider returned incomplete PM2.5 data")
        _atomic_json_write(_DISK_CACHE, context)
        with _LOCK:
            _CACHE[key] = (now, context)
        return {**context, "mode": "live", "stale": False}
    except Exception as exc:
        disk = _read_disk_cache(_DISK_CACHE)
        if disk:
            return {**disk, "mode": "disk_cache", "stale": True, "fallback_reason": str(exc)}
        raise RuntimeError(f"live air-quality data unavailable and no cache exists: {exc}") from exc
