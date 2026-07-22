# backend/app/ml/plume_math.py
import numpy as np

_PG_SY = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
_PG_SZ = [0.20, 0.12, 0.08, 0.06, 0.03, 0.016]

def _stability_class(wind_speed: float, hour: int) -> int:
    """Must stay identical to inference.py::_stability_class and
    step3_etl_and_train.py's inline equivalent — this determines the
    Pasquill-Gifford sigma coefficients used both when the hotspot was
    predicted and when we back-attribute it to a source. A mismatch here
    silently biases confidence scores.
    """
    is_day = 6 <= hour <= 18
    if is_day:
        if wind_speed < 2:   return 0
        elif wind_speed < 3: return 1
        elif wind_speed < 5: return 2
        else:                return 3
    else:
        if wind_speed < 2:   return 5
        elif wind_speed < 3: return 4
        else:                return 3

def calculate_source_attribution(hotspot_lat: float, hotspot_lon: float, wind_speed: float, wind_dir_deg: float, sources_list: list, hour: int = None) -> list:
    from datetime import datetime
    if hour is None:
        hour = datetime.now().hour
    wind_rad = np.radians((wind_dir_deg + 180) % 360)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    sc = _stability_class(wind_speed, hour)
    u  = max(wind_speed, 0.5)

    scored = []
    for src in sources_list:
        dx_km = (hotspot_lon - src["lon"]) * 95.0   
        dy_km = (hotspot_lat - src["lat"]) * 111.0  
        disp_vec = np.array([dx_km, dy_km])
        dist_km  = float(np.linalg.norm(disp_vec))

        if dist_km < 0.1:
            scored.append({**src, "confidence_score": 99.9, "distance_km": round(dist_km, 2)})
            continue

        x_down = float(np.dot(wind_vec, disp_vec))
        if x_down <= 0:
            scored.append({**src, "confidence_score": 0.0, "distance_km": round(dist_km, 2)})
            continue

        y_cross = float(np.linalg.norm(disp_vec - x_down * wind_vec))
        sigma_y = max(_PG_SY[sc] * (x_down ** 0.894), 0.5)
        sigma_z = max(_PG_SZ[sc] * (x_down ** 0.894), 0.3)

        intensity = src.get("intensity", 400)
        C = (intensity / (np.pi * u * sigma_y * sigma_z)) * np.exp(-0.5 * (y_cross / sigma_y) ** 2)

        scored.append({
            **src, "raw_concentration": round(float(C), 4), "distance_km": round(dist_km, 2),
        })

    raw_vals = [s.get("raw_concentration", 0.0) for s in scored]
    max_raw  = max(raw_vals) if max(raw_vals) > 0 else 1.0

    for s in scored:
        raw = s.pop("raw_concentration", 0.0)
        s["confidence_score"] = round(min(99.9, (raw / max_raw) * 99.9), 1)

    return sorted(scored, key=lambda x: x["confidence_score"], reverse=True)