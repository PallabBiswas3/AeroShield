# backend/app/ml/train.py
#
# ⚠️  SYNTHETIC DEMO TRAINER — NOT PART OF THE REAL DELHI PIPELINE  ⚠️
# This script generates a fully synthetic dataset (fake wind, fake traffic,
# fake seasonal baseline) around Bangalore coordinates for local dev/demo
# purposes, using a DIFFERENT feature/grid schema than the real Delhi model
# trained by pipeline/step1_download_openaq.py -> step2_spatial_layers.py
# -> step3_etl_and_train.py (which uses real OpenAQ + Open-Meteo + OSM data
# and includes no2/co/pm10 features that this script's model does not have).
#
# inference.py no longer auto-invokes this on missing model file — it now
# raises a clear error instead — specifically so this script can't
# accidentally end up silently serving "Delhi" predictions. Only run this
# file directly if you intentionally want a quick synthetic model to
# sanity-check the API/frontend wiring before your real data pipeline
# finishes.
import os
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

GRID_SIZE = 15
BASE_LAT, BASE_LON = 12.9716, 77.5946  # Bangalore — demo only, do not confuse with Delhi model

EMISSION_SOURCES = [
    {"source_id": 1, "name": "Whitefield Infrastructure Project", "type": "Construction", "x": 12, "y": 13, "intensity": 520, "lat": BASE_LAT + (13 - 7) * 0.009, "lon": BASE_LON + (12 - 7) * 0.010},
    {"source_id": 2, "name": "Peenya Metal Processing Grid", "type": "Industrial Stack", "x": 3, "y": 10, "intensity": 460, "lat": BASE_LAT + (10 - 7) * 0.009, "lon": BASE_LON + (3 - 7) * 0.010},
    {"source_id": 3, "name": "Hebbal Thermal Cluster", "type": "Industrial Stack", "x": 7, "y": 2, "intensity": 310, "lat": BASE_LAT + (2 - 7) * 0.009, "lon": BASE_LON + (7 - 7) * 0.010},
]

MODEL_NAME        = "surrogate_model.joblib"
SOURCES_JSON_NAME = "emission_sources.json"
META_JSON_NAME    = "model_meta.json"

_PG_SY = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
_PG_SZ = [0.20, 0.12, 0.08, 0.06, 0.03, 0.016]

def get_stability_class(wind_speed: float, hour_of_day: int) -> int:
    is_day = 6 <= hour_of_day <= 18
    if is_day:
        if wind_speed < 2:   return 0
        elif wind_speed < 3: return 1
        elif wind_speed < 5: return 2
        else:                return 3
    else:
        if wind_speed < 2:   return 5
        elif wind_speed < 3: return 4
        else:                return 3

def gaussian_plume_raw(src_x, src_y, cell_x, cell_y, wind_speed, wind_rad, stability_class, intensity) -> float:
    dx = cell_x - src_x
    dy = cell_y - src_y
    wind_vec  = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    cell_vec  = np.array([dx, dy], dtype=float)
    x_down    = np.dot(wind_vec, cell_vec)

    if x_down < 1.0: return 0.0

    y_cross = float(np.linalg.norm(cell_vec - x_down * wind_vec))
    sc      = stability_class
    sigma_y = max(_PG_SY[sc] * (x_down ** 0.894), 0.5)
    sigma_z = max(_PG_SZ[sc] * (x_down ** 0.894), 0.3)
    u       = max(wind_speed, 0.5)

    C = (intensity / (np.pi * u * sigma_y * sigma_z)) * np.exp(-0.5 * (y_cross / sigma_y) ** 2)
    return float(C)

def compute_source_flux(cell_x, cell_y, wind_speed, wind_rad, stability_class, blh) -> float:
    total_raw = sum(
        gaussian_plume_raw(s["x"], s["y"], cell_x, cell_y, wind_speed, wind_rad, stability_class, s["intensity"])
        for s in EMISSION_SOURCES
    )
    blh_factor = 300.0 / max(blh, 100.0)
    return float(np.log1p(total_raw) * blh_factor * 5.0)

def traffic_density(hour_of_day: int, day_of_week: int, x: int, y: int) -> float:
    in_centre = (4 <= x <= 10) and (4 <= y <= 10)
    scale = 1.4 if in_centre else 0.7
    if day_of_week < 5:
        density = (12.0 + 85 * np.exp(-0.5 * ((hour_of_day - 8.5) / 1.2) ** 2) + 90 * np.exp(-0.5 * ((hour_of_day - 18.0) / 1.3) ** 2))
    else:
        density = 15.0 + 55 * np.exp(-0.5 * ((hour_of_day - 11.5) / 2.0) ** 2)
    return float(max(8.0, density * scale + np.random.normal(0, 4)))

def seasonal_baseline(month: int) -> float:
    if month in (10, 11, 12, 1): return float(np.random.uniform(45, 70))
    elif month in (6, 7, 8, 9): return float(np.random.uniform(15, 28))
    else: return float(np.random.uniform(28, 45))

def boundary_layer_height(hour_of_day: int, stability_class: int) -> float:
    diurnal   = max(0.1, 0.5 + 0.5 * np.sin(np.pi * (hour_of_day - 6) / 12))
    stability = (5 - stability_class) / 5.0
    return float(200 + 1600 * diurnal * stability + np.random.normal(0, 30))

FEATURES = ['x', 'y', 'hour', 'day_of_week', 'month', 'wind_speed', 'wind_direction', 'wind_sin', 'wind_cos', 'stability_class', 'boundary_layer_height', 'traffic_density', 'source_flux_log', 'lag_1h_pm25', 'lag_3h_pm25']

def generate_spatial_dataset(num_days: int = 90) -> pd.DataFrame:
    np.random.seed(42)
    total_hours = num_days * 24
    records     = []
    pm25_buf    = {cid: [] for cid in range(GRID_SIZE * GRID_SIZE)}

    for hour_abs in range(total_hours):
        wind_speed = float(max(0.5, np.random.weibull(2.0) * 6.0))
        wind_dir   = float(np.random.uniform(0.0, 360.0))
        wind_rad   = np.radians((wind_dir + 180) % 360)
        month       = (hour_abs // (24 * 30)) % 12 + 1
        day_of_week = (hour_abs // 24) % 7
        hour_of_day = hour_abs % 24
        stability   = get_stability_class(wind_speed, hour_of_day)
        blh         = boundary_layer_height(hour_of_day, stability)
        baseline    = seasonal_baseline(month)

        for cell_id in range(GRID_SIZE * GRID_SIZE):
            x, y = cell_id % GRID_SIZE, cell_id // GRID_SIZE
            traf      = traffic_density(hour_of_day, day_of_week, x, y)
            flux_log  = compute_source_flux(x, y, wind_speed, wind_rad, stability, blh)
            base_pm25 = baseline + 0.35 * traf + flux_log
            if np.random.rand() < 0.01: base_pm25 *= np.random.uniform(1.3, 2.2)
            target_pm25 = float(max(5.0, base_pm25 + np.random.normal(0, 5)))

            buf  = pm25_buf[cell_id]
            lag1 = buf[-1] if len(buf) >= 1 else target_pm25
            lag3 = buf[-3] if len(buf) >= 3 else target_pm25
            buf.append(target_pm25)
            if len(buf) > 6: buf.pop(0)

            records.append({
                'x': x, 'y': y, 'hour': hour_of_day, 'day_of_week': day_of_week, 'month': month,
                'wind_speed': wind_speed, 'wind_direction': wind_dir, 'wind_sin': float(np.sin(np.radians(wind_dir))),
                'wind_cos': float(np.cos(np.radians(wind_dir))), 'stability_class': stability,
                'boundary_layer_height': blh, 'traffic_density': traf, 'source_flux_log': flux_log,
                'lag_1h_pm25': lag1, 'lag_3h_pm25': lag3, 'target_pm25': target_pm25,
            })
    return pd.DataFrame(records)

def train_core_surrogate():
    df = generate_spatial_dataset(num_days=90)
    X = df[FEATURES]
    y = df['target_pm25']
    split_idx = int(len(df) * 0.88)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=63, max_depth=8, min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2, objective='huber', alpha=0.9, n_jobs=-1, verbose=-1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])

    y_pred     = model.predict(X_val)
    rmse       = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae        = float(mean_absolute_error(y_val, y_pred))
    within_15  = float(np.mean(np.abs(y_val.values - y_pred) <= 15) * 100)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(data_dir, MODEL_NAME))
    with open(os.path.join(data_dir, SOURCES_JSON_NAME), 'w') as f: json.dump(EMISSION_SOURCES, f, indent=2)
    with open(os.path.join(data_dir, META_JSON_NAME), 'w') as f:
        json.dump({"features": FEATURES, "grid_size": GRID_SIZE, "base_lat": BASE_LAT, "base_lon": BASE_LON, "metrics": {"rmse": round(rmse, 3), "mae": round(mae, 3), "within_15pct": round(within_15, 1)}, "best_iteration": int(model.best_iteration_), "training_days": 90, "num_sources": len(EMISSION_SOURCES)}, f, indent=2)
    return model

if __name__ == "__main__":
    train_core_surrogate()