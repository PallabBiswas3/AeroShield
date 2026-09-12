"""NASA FIRMS active-fire evidence adapter with transparent fallback state."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import math

import requests

from app.ml.features import circular_difference_deg, haversine_km


FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{bbox}/2"
DELHI_REGION = (76.70, 28.20, 77.60, 29.10)
_ROOT = Path(__file__).resolve().parents[2]
_CACHE_PATH = _ROOT / "data" / "runtime_cache" / "firms_delhi.json"


def _safe_failure_reason(exc: Exception) -> str:
    """Return an operator-useful category without exposing the key-bearing URL."""
    if isinstance(exc, requests.Timeout):
        return "NASA FIRMS request timed out"
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        return f"NASA FIRMS returned HTTP {status}"
    if isinstance(exc, requests.RequestException):
        return "NASA FIRMS network request failed"
    return "NASA FIRMS response could not be processed"


def _read_cache() -> dict | None:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(payload: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _CACHE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, _CACHE_PATH)


def _parse_firms_csv(text: str) -> list[dict]:
    fires = []
    for row in csv.DictReader(StringIO(text)):
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not (DELHI_REGION[1] <= lat <= DELHI_REGION[3] and DELHI_REGION[0] <= lon <= DELHI_REGION[2]):
                continue
            fires.append({
                "id": f"{row.get('satellite', 'VIIRS')}-{row.get('acq_date')}-{row.get('acq_time')}-{lat:.4f}-{lon:.4f}",
                "lat": lat, "lon": lon,
                "detected_at": f"{row.get('acq_date', '')} {row.get('acq_time', '')} UTC".strip(),
                "satellite": row.get("satellite") or "VIIRS-SNPP",
                "confidence": row.get("confidence"),
                "frp": float(row["frp"]) if row.get("frp") else None,
                "daynight": row.get("daynight"),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(fires, key=lambda item: item.get("frp") or 0, reverse=True)[:250]


def get_satellite_fires() -> dict:
    """Fetch FIRMS evidence when configured; otherwise expose availability honestly."""
    key = os.getenv("NASA_FIRMS_MAP_KEY", "").strip()
    if key:
        try:
            bbox = ",".join(str(value) for value in DELHI_REGION)
            response = requests.get(FIRMS_URL.format(key=key, bbox=bbox), timeout=20)
            response.raise_for_status()
            fires = _parse_firms_csv(response.text)
            result = {
                "fires": fires,
                "mode": "live",
                "stale": False,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "provider": "NASA FIRMS VIIRS-SNPP NRT",
                "region": DELHI_REGION,
            }
            _write_cache(result)
            return result
        except Exception as exc:
            safe_reason = _safe_failure_reason(exc)
            cached = _read_cache()
            if cached:
                return {**cached, "mode": "disk_cache", "stale": True, "fallback_reason": safe_reason}
            return {"fires": [], "mode": "unavailable", "stale": True, "provider": "NASA FIRMS", "reason": safe_reason}

    cached = _read_cache()
    if cached:
        return {**cached, "mode": "disk_cache", "stale": True, "fallback_reason": "NASA_FIRMS_MAP_KEY is not configured"}
    return {
        "fires": [], "mode": "not_configured", "stale": True,
        "provider": "NASA FIRMS VIIRS-SNPP NRT",
        "reason": "Set NASA_FIRMS_MAP_KEY to enable live satellite thermal anomalies.",
    }


def rank_fire_evidence(hotspot_lat: float, hotspot_lon: float, wind_from_deg: float, fires: list[dict], max_distance_km: float = 150.0) -> list[dict]:
    """Rank thermal anomalies by distance and transport-direction compatibility."""
    wind_to_deg = (float(wind_from_deg) + 180.0) % 360.0
    ranked = []
    for fire in fires:
        distance = haversine_km(fire["lat"], fire["lon"], hotspot_lat, hotspot_lon)
        if distance > max_distance_km or distance == 0:
            continue
        lat1, lat2 = math.radians(fire["lat"]), math.radians(hotspot_lat)
        delta_lon = math.radians(hotspot_lon - fire["lon"])
        y = math.sin(delta_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        separation = circular_difference_deg(bearing, wind_to_deg)
        compatibility = max(0.0, math.cos(math.radians(separation)))
        distance_weight = math.exp(-distance / 75.0)
        score = compatibility * distance_weight
        ranked.append({
            **fire,
            "distance_km": round(distance, 1),
            "transport_separation_deg": round(separation, 1),
            "wind_compatibility": round(compatibility, 3),
            "screening_score": round(score, 4),
            "interpretation": "Satellite thermal anomaly screening evidence; not proof of a pollution contribution.",
        })
    return sorted(ranked, key=lambda item: item["screening_score"], reverse=True)
