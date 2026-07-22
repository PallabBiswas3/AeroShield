# backend/app/ml/inference.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ── Path resolution ──────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_BACKEND     = os.path.dirname(os.path.dirname(_HERE))
# Changed path resolution to point to root /data directory correctly
DATA_DIR     = os.path.abspath(os.path.join(_BACKEND, "..", "data"))
MODEL_PATH   = os.path.join(DATA_DIR, "surrogate_model.joblib")
SOURCES_PATH = os.path.join(DATA_DIR, "delhi_emission_sources.json")
ROAD_PATH    = os.path.join(DATA_DIR, "delhi_grid_road_density.csv")
META_PATH    = os.path.join(DATA_DIR, "model_meta.json")

# ── Grid constants (Delhi — Connaught Place centre) ──────────────────────────
GRID_SIZE    = 15
BASE_LAT     = 28.6280
BASE_LON     = 77.2090
DEG_LAT      = 0.009     
DEG_LON      = 0.0105    

# ── Delhi annual median fills for co-pollutants (CPCB 2024-25 averages) ──────
DELHI_MEDIANS = {
    "no2":  38.0,   
    "co":   1.1,    
    "pm10": 95.0,
    "pm25_median": 41.1,
}

# ── Pasquill-Gifford σ coefficients ─────────────────────────────────────────
_PG_SY = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
_PG_SZ = [0.20, 0.12, 0.08, 0.06, 0.03, 0.016]

# ── Lazy-loaded globals ──────────────────────────────────────────────────────
_model       = None
_sources     = None
_road_grid   = None   # dict of numpy arrays: {"lat", "lon", "density"}
_meta        = None

def _ensure_loaded():
    global _model, _sources, _road_grid, _meta
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        # NOTE: we deliberately do NOT silently auto-train a synthetic
        # fallback model anymore. train.py's synthetic generator uses a
        # different feature/grid schema than the real Delhi pipeline
        # (step1/step2/step3), so an auto-trained fallback would silently
        # serve predictions inconsistent with delhi_emission_sources.json
        # and delhi_grid_road_density.csv. Fail loudly instead.
        raise RuntimeError(
            "[inference] surrogate_model.joblib not found at "
            f"{MODEL_PATH}. Run the data pipeline first:\n"
            "  python pipeline/step1_download_openaq.py\n"
            "  python pipeline/step2_spatial_layers.py\n"
            "  python pipeline/step3_etl_and_train.py"
        )

    _model = joblib.load(MODEL_PATH)

    if os.path.exists(SOURCES_PATH):
        with open(SOURCES_PATH) as f:
            _sources = json.load(f)
    else:
        _sources = []

    if os.path.exists(ROAD_PATH):
        road_df = pd.read_csv(ROAD_PATH)
        # Keep raw lat/lon arrays for nearest-neighbour lookup instead of
        # indexing by cell_id — the road-density grid from step2 is built
        # at a different resolution (45x45) than the serving grid here
        # (15x15), so cell_id values are NOT comparable across the two.
        _road_grid = {
            "lat": road_df["lat"].to_numpy(dtype=float),
            "lon": road_df["lon"].to_numpy(dtype=float),
            "density": road_df["road_density_m"].to_numpy(dtype=float),
        }
    else:
        _road_grid = None

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta = json.load(f)

def _stability_class(wind_speed: float, hour: int) -> int:
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

def _blh(hour: int, sc: int) -> float:
    d = max(0.1, 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12))
    s = (5 - sc) / 5.0
    return 200 + 1600 * d * s

def _source_flux_log(cell_x, cell_y, wind_speed, wind_rad, sc, blh_val, sources) -> float:
    total = 0.0
    cell_lat = BASE_LAT + (cell_y - GRID_SIZE // 2) * DEG_LAT
    cell_lon = BASE_LON + (cell_x - GRID_SIZE // 2) * DEG_LON

    for s in sources:
        dx_km = (cell_lon - s["lon"]) * 95.0
        dy_km = (cell_lat - s["lat"]) * 111.0
        wind_vec  = np.array([np.cos(wind_rad), np.sin(wind_rad)])
        cell_vec  = np.array([dx_km, dy_km], dtype=float)
        x_down    = float(np.dot(wind_vec, cell_vec))

        if x_down < 1.0:
            continue

        y_cross = float(np.linalg.norm(cell_vec - x_down * wind_vec))
        sy = max(_PG_SY[sc] * (x_down ** 0.894), 0.5)
        sz = max(_PG_SZ[sc] * (x_down ** 0.894), 0.3)
        u  = max(wind_speed, 0.5)
        C  = (s["intensity"] / (np.pi * u * sy * sz)) * np.exp(-0.5 * (y_cross / sy) ** 2)
        total += C

    blh_factor = 300.0 / max(blh_val, 100.0)
    return float(np.log1p(total) * blh_factor * 5.0)

def _road_density_for_latlon(lat: float, lon: float) -> float:
    """Nearest-neighbour lookup into the step2 road-density grid.

    step2's grid is built at 45x45 resolution while this module's serving
    grid is 15x15 — the two `cell_id` numbering schemes are NOT comparable,
    so we must match by physical coordinates, not by index. This mirrors
    the approach already used correctly in step3_etl_and_train.py's
    merge_road_density().
    """
    if _road_grid is None:
        return 1500.0
    dlat = _road_grid["lat"] - lat
    dlon = _road_grid["lon"] - lon
    idx = int(np.argmin(dlat * dlat + dlon * dlon))
    return float(_road_grid["density"][idx])

def _diurnal_factor(hour: int, day_of_week: int) -> float:
    if day_of_week < 5:
        return (0.2 + 0.8 * (
            0.8 * np.exp(-0.5 * ((hour - 8.5) / 1.2) ** 2) +
            1.0 * np.exp(-0.5 * ((hour - 18.0) / 1.3) ** 2)
        ))
    return 0.3 + 0.5 * np.exp(-0.5 * ((hour - 11.5) / 2.0) ** 2)

def _aqi_category(pm25: float) -> dict:
    if pm25 < 30:   return {"label": "Good",          "color": "#00e676", "level": 1}
    elif pm25 < 60: return {"label": "Satisfactory",  "color": "#b2ff59", "level": 2}
    elif pm25 < 90: return {"label": "Moderate",      "color": "#ffee58", "level": 3}
    elif pm25 < 120:return {"label": "Poor",          "color": "#ff9800", "level": 4}
    elif pm25 < 250:return {"label": "Very Poor",     "color": "#f44336", "level": 5}
    else:           return {"label": "Severe",        "color": "#b71c1c", "level": 6}

def get_grid_predictions(hour: int = 12, day_of_week: int = 1, wind_speed: float = 4.0, wind_direction: float = 315.0, month: int = None, lag_pm25: float = None) -> list:
    _ensure_loaded()
    if month is None: month = datetime.now().month

    lat_mean, lon_mean = BASE_LAT, BASE_LON
    wind_rad  = np.radians((wind_direction + 180) % 360)
    sc        = _stability_class(wind_speed, hour)
    blh_val   = _blh(hour, sc)
    wind_sin  = float(np.sin(np.radians(wind_direction)))
    wind_cos  = float(np.cos(np.radians(wind_direction)))
    default_lag = lag_pm25 if lag_pm25 is not None else DELHI_MEDIANS.get("pm25_median", 41.1)

    records = []
    for cell_id in range(GRID_SIZE * GRID_SIZE):
        x = cell_id % GRID_SIZE
        y = cell_id // GRID_SIZE
        cell_lat = BASE_LAT + (y - GRID_SIZE // 2) * DEG_LAT
        cell_lon = BASE_LON + (x - GRID_SIZE // 2) * DEG_LON

        x_feat = round((cell_lon - lon_mean) / DEG_LON, 1)
        y_feat = round((cell_lat - lat_mean) / DEG_LAT, 1)
        road_m  = _road_density_for_latlon(cell_lat, cell_lon)
        diurnal = _diurnal_factor(hour, day_of_week)
        traffic = (road_m / 1000.0) * diurnal * 50
        flux = _source_flux_log(x, y, wind_speed, wind_rad, sc, blh_val, _sources or [])

        records.append({
            "x": x_feat, "y": y_feat, "hour": hour, "day_of_week": day_of_week, "month": month,
            "wind_speed": wind_speed, "wind_direction": wind_direction, "wind_sin": wind_sin, "wind_cos": wind_cos,
            "stability_class": sc, "boundary_layer_height": blh_val, "traffic_density": traffic, "source_flux_log": flux,
            "lag_1h_pm25": default_lag, "lag_3h_pm25": default_lag,
            "no2": DELHI_MEDIANS["no2"], "co": DELHI_MEDIANS["co"], "pm10": DELHI_MEDIANS["pm10"],
            "_cell_id": cell_id, "_lat": round(cell_lat, 6), "_lon": round(cell_lon, 6),
        })

    df = pd.DataFrame(records)
    if _meta and "features" in _meta: feature_cols = _meta["features"]
    else: feature_cols = ["x", "y", "hour", "day_of_week", "month", "wind_speed", "wind_direction", "wind_sin", "wind_cos", "stability_class", "boundary_layer_height", "traffic_density", "source_flux_log", "lag_1h_pm25", "lag_3h_pm25", "no2", "co", "pm10"]

    raw_preds = _model.predict(df[feature_cols])
    preds = np.clip(raw_preds, 0.0, 500.0)

    output = []
    for i, row in df.iterrows():
        pm25 = float(preds[i])
        output.append({
            "cell_id": int(row["_cell_id"]), "x": int(row["_cell_id"] % GRID_SIZE), "y": int(row["_cell_id"] // GRID_SIZE),
            "lat": float(row["_lat"]), "lon": float(row["_lon"]), "predicted_pm25": round(pm25, 1),
            "aqi_category": _aqi_category(pm25), "wind_speed": wind_speed, "wind_direction": wind_direction,
            "stability_class": sc, "boundary_layer_height": round(blh_val, 0), "source_flux_log": round(float(row["source_flux_log"]), 2),
        })
    return output

def get_point_prediction(lat: float, lon: float, wind_speed: float, wind_direction: float, hour: int = None, day_of_week: int = None, month: int = None, lag_pm25: float = None) -> dict:
    """Run the surrogate model for a single exact lat/lon (e.g. a clicked
    hotspot) instead of only the coarse 15x15 grid centres. Used by
    /api/analyze-hotspot so the enforcement mandate is grounded in a real
    prediction rather than a hardcoded placeholder value.
    """
    _ensure_loaded()
    now = datetime.now()
    if hour is None: hour = now.hour
    if day_of_week is None: day_of_week = now.weekday()
    if month is None: month = now.month

    wind_rad = np.radians((wind_direction + 180) % 360)
    sc       = _stability_class(wind_speed, hour)
    blh_val  = _blh(hour, sc)
    default_lag = lag_pm25 if lag_pm25 is not None else DELHI_MEDIANS.get("pm25_median", 41.1)

    # Reuse the same x/y grid-cell coordinate convention as the training
    # pipeline (offset from grid centre in DEG_LAT/DEG_LON units) so the
    # point falls on the same feature scale the model was trained on.
    x_feat = round((lon - BASE_LON) / DEG_LON, 1)
    y_feat = round((lat - BASE_LAT) / DEG_LAT, 1)

    # Approximate grid x,y indices purely for the Gaussian-plume source
    # flux calculation, which expects grid-index space.
    cell_x = int(round(x_feat)) + GRID_SIZE // 2
    cell_y = int(round(y_feat)) + GRID_SIZE // 2

    road_m  = _road_density_for_latlon(lat, lon)
    diurnal = _diurnal_factor(hour, day_of_week)
    traffic = (road_m / 1000.0) * diurnal * 50
    flux    = _source_flux_log(cell_x, cell_y, wind_speed, wind_rad, sc, blh_val, _sources or [])

    row = {
        "x": x_feat, "y": y_feat, "hour": hour, "day_of_week": day_of_week, "month": month,
        "wind_speed": wind_speed, "wind_direction": wind_direction,
        "wind_sin": float(np.sin(np.radians(wind_direction))), "wind_cos": float(np.cos(np.radians(wind_direction))),
        "stability_class": sc, "boundary_layer_height": blh_val, "traffic_density": traffic, "source_flux_log": flux,
        "lag_1h_pm25": default_lag, "lag_3h_pm25": default_lag,
        "no2": DELHI_MEDIANS["no2"], "co": DELHI_MEDIANS["co"], "pm10": DELHI_MEDIANS["pm10"],
    }
    feature_cols = _meta["features"] if _meta and "features" in _meta else list(row.keys())
    df = pd.DataFrame([row])
    pm25 = float(np.clip(_model.predict(df[feature_cols])[0], 0.0, 500.0))

    return {
        "lat": lat, "lon": lon, "predicted_pm25": round(pm25, 1), "aqi_category": _aqi_category(pm25),
        "wind_speed": wind_speed, "wind_direction": wind_direction,
        "stability_class": sc, "boundary_layer_height": round(blh_val, 0), "source_flux_log": round(flux, 2),
    }

def get_emission_sources() -> list:
    _ensure_loaded()
    return _sources or []

def get_model_meta() -> dict:
    _ensure_loaded()
    return _meta or {}
