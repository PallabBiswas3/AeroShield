# backend/pipeline/step3_etl_and_train.py
import os, json, time, requests, warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAW_CSV_PATH      = os.path.join(DATA_DIR, "raw_openaq", "delhi_all_stations_2025.csv")
SOURCES_PATH      = os.path.join(DATA_DIR, "delhi_emission_sources.json")
ROAD_DENSITY_PATH = os.path.join(DATA_DIR, "delhi_grid_road_density.csv")
MODEL_PATH        = os.path.join(DATA_DIR, "surrogate_model.joblib")
META_PATH         = os.path.join(DATA_DIR, "model_meta.json")
MERGED_CSV_PATH   = os.path.join(DATA_DIR, "merged_training_data.csv")

_PG_SY = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
_PG_SZ = [0.20, 0.12, 0.08, 0.06, 0.03, 0.016]

def load_and_pivot(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"], utc=True)
    df["datetime_hour"] = df["datetimeUtc"].dt.floor("h")
    df = df[(df["value"] >= 0) & (df["value"] < 10_000)]
    mask = (df["parameter"] == "pm25") & (df["unit"].str.lower() == "ppm")
    df.loc[mask, "value"] *= 1_000
    pivot = df.pivot_table(index=["location_id", "location_name", "latitude", "longitude", "datetime_hour"], columns="parameter", values="value", aggfunc="mean").reset_index()
    pivot.columns.name = None
    pivot = pivot.dropna(subset=["pm25"])
    return pivot[(pivot["pm25"] > 0) & (pivot["pm25"] <= 500)]

def fetch_wind(lat, lon, start, end) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "hourly": "wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,boundary_layer_height", "wind_speed_unit": "ms", "timezone": "UTC"}
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"datetime_hour": pd.to_datetime(h["time"], utc=True), "wind_speed_met": h["wind_speed_10m"], "wind_dir_met": h["wind_direction_10m"], "temp_met": h["temperature_2m"], "rh_met": h["relative_humidity_2m"], "blh_met": h["boundary_layer_height"]})

def enrich_with_wind(df: pd.DataFrame) -> pd.DataFrame:
    start, end = df["datetime_hour"].min().strftime("%Y-%m-%d"), df["datetime_hour"].max().strftime("%Y-%m-%d")
    met_frames = []
    for loc_id, grp in df.groupby("location_id"):
        lat, lon = grp["latitude"].iloc[0], grp["longitude"].iloc[0]
        try:
            met = fetch_wind(lat, lon, start, end)
            sub = grp[["location_id", "datetime_hour"]].copy().merge(met, on="datetime_hour", how="left")
            if "wind_speed" in grp.columns and grp["wind_speed"].notna().mean() > 0.5:
                sub["wind_speed_met"], sub["wind_dir_met"] = grp["wind_speed"].values, grp["wind_direction"].values
        except Exception:
            sub = grp[["location_id", "datetime_hour"]].copy()
            sub["wind_speed_met"] = grp["wind_speed"].values if "wind_speed" in grp.columns else np.nan
            sub["wind_dir_met"] = grp["wind_direction"].values if "wind_direction" in grp.columns else np.nan
            sub["blh_met"], sub["temp_met"], sub["rh_met"] = np.nan, np.nan, np.nan
        met_frames.append(sub)
        time.sleep(0.5)

    df = df.merge(pd.concat(met_frames, ignore_index=True), on=["location_id", "datetime_hour"], how="left").sort_values(["location_id", "datetime_hour"])
    df["wind_speed_met"] = df.groupby("location_id")["wind_speed_met"].transform(lambda s: s.ffill(limit=6).bfill(limit=6))
    df["wind_dir_met"] = df.groupby("location_id")["wind_dir_met"].transform(lambda s: s.ffill(limit=6).bfill(limit=6))
    
    month = df["datetime_hour"].dt.tz_convert("Asia/Kolkata").dt.month
    default_ws = np.where(month.isin([10,11,12,1,2]), 4.0, np.where(month.isin([6,7,8,9]), 3.0, 3.5))
    default_wd = np.where(month.isin([10,11,12,1,2]), 315.0, np.where(month.isin([6,7,8,9]), 220.0, 270.0))
    df["wind_speed_met"] = df["wind_speed_met"].fillna(pd.Series(default_ws, index=df.index))
    df["wind_dir_met"] = df["wind_dir_met"].fillna(pd.Series(default_wd, index=df.index))
    return df

def merge_road_density(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(ROAD_DENSITY_PATH):
        df["road_density_m"] = 1500.0
        return df
    grid = pd.read_csv(ROAD_DENSITY_PATH)
    def nearest(lat, lon): return float(grid.loc[np.sqrt((grid["lat"] - lat)**2 + (grid["lon"] - lon)**2).idxmin(), "road_density_m"])
    df["road_density_m"] = df["location_id"].map({loc_id: nearest(grp["latitude"].iloc[0], grp["longitude"].iloc[0]) for loc_id, grp in df.groupby("location_id")})
    return df

def engineer_features(df: pd.DataFrame, sources: list) -> pd.DataFrame:
    out = df.copy().sort_values(["location_id", "datetime_hour"])
    local = out["datetime_hour"].dt.tz_convert("Asia/Kolkata")
    out["hour"], out["day_of_week"], out["month"] = local.dt.hour, local.dt.dayofweek, local.dt.month
    out["wind_speed"], out["wind_direction"] = out["wind_speed_met"], out["wind_dir_met"]
    out["wind_sin"], out["wind_cos"] = np.sin(np.radians(out["wind_direction"])), np.cos(np.radians(out["wind_direction"]))
    
    out["stability_class"] = [0 if ws<2 else 1 if ws<3 else 2 if ws<5 else 3 if 6<=h<=18 else 5 if ws<2 else 4 if ws<3 else 3 for ws, h in zip(out["wind_speed"], out["hour"])]
    out["boundary_layer_height"] = out.apply(lambda r: float(r["blh_met"]) if pd.notna(r.get("blh_met")) else 200 + 1600 * max(0.1, 0.5 + 0.5 * np.sin(np.pi * (r["hour"] - 6) / 12)) * (5 - r["stability_class"]) / 5.0, axis=1)

    wind_rad_arr = np.radians((out["wind_direction"].values + 180) % 360)
    flux = np.zeros(len(out))
    for s in sources:
        for i in range(len(out)):
            dx, dy = (out["longitude"].iloc[i] - s["lon"]) * 95.0, (out["latitude"].iloc[i] - s["lat"]) * 111.0
            wv, cv = np.array([np.cos(wind_rad_arr[i]), np.sin(wind_rad_arr[i])]), np.array([dx, dy])
            xd = np.dot(wv, cv)
            if xd >= 1.0:
                yc = float(np.linalg.norm(cv - xd * wv))
                sc, u = int(out["stability_class"].iloc[i]), max(out["wind_speed"].iloc[i], 0.5)
                sy, sz = max(_PG_SY[sc] * (xd ** 0.894), 0.5), max(_PG_SZ[sc] * (xd ** 0.894), 0.3)
                flux[i] += float((s["intensity"] / (np.pi * u * sy * sz)) * np.exp(-0.5 * (yc / sy) ** 2))
                
    out["source_flux_log"] = np.log1p(flux) * (300.0 / np.maximum(out["boundary_layer_height"].values, 100.0)) * 5.0
    out["traffic_density"] = (out["road_density_m"] / 1000.0) * out.apply(lambda r: (0.2 + 0.8 * (0.8 * np.exp(-0.5 * ((r["hour"] - 8.5) / 1.2) ** 2) + 1.0 * np.exp(-0.5 * ((r["hour"] - 18.0) / 1.3) ** 2))) if r["day_of_week"] < 5 else 0.3 + 0.5 * np.exp(-0.5 * ((r["hour"] - 11.5) / 2.0) ** 2), axis=1) * 50
    out["x"], out["y"] = ((out["longitude"] - out["longitude"].mean()) / 0.0105).round(1), ((out["latitude"] - out["latitude"].mean()) / 0.009).round(1)

    out["pm25_filled"] = out.groupby("location_id")["pm25"].transform(lambda s: s.ffill(limit=3).bfill(limit=1))
    out["lag_1h_pm25"], out["lag_3h_pm25"] = out.groupby("location_id")["pm25_filled"].shift(1), out.groupby("location_id")["pm25_filled"].shift(3)
    for col in ["lag_1h_pm25", "lag_3h_pm25"]: out[col] = out.groupby("location_id")[col].transform(lambda s: s.fillna(s.median()))
    for col in ["no2", "co", "pm10"]:
        if col in out.columns: out[col] = out.groupby("location_id")[col].transform(lambda s: s.ffill(limit=3).bfill(limit=3).fillna(s.median()))
    return out

def main():
    if not os.path.exists(RAW_CSV_PATH): return
    df = engineer_features(merge_road_density(enrich_with_wind(load_and_pivot(RAW_CSV_PATH))), json.load(open(SOURCES_PATH)) if os.path.exists(SOURCES_PATH) else [])
    df.to_csv(MERGED_CSV_PATH, index=False)
    
    features = ["x", "y", "hour", "day_of_week", "month", "wind_speed", "wind_direction", "wind_sin", "wind_cos", "stability_class", "boundary_layer_height", "traffic_density", "source_flux_log", "lag_1h_pm25", "lag_3h_pm25"] + [f for f in ["no2", "co", "pm10"] if f in df.columns]
    clean = df.dropna(subset=features + ["pm25"])
    split = int(len(clean) * 0.80)
    X_tr, y_tr = clean[features].iloc[:split], clean["pm25"].iloc[:split]
    X_val, y_val = clean[features].iloc[split:], clean["pm25"].iloc[split:]

    model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=8, min_child_samples=max(5, len(clean)//300), subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2, objective="huber", alpha=0.9, n_jobs=-1, verbose=-1, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    # ── Validation metrics (Feature #2: drives the "Model: RMSE ..." UI badge) ──
    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    within_15pct = float(np.mean(np.abs(y_val.values - y_pred) <= 15) * 100)
    print(f"[step3] Validation — RMSE: {rmse:.2f} µg/m³  MAE: {mae:.2f} µg/m³  Within ±15 µg/m³: {within_15pct:.1f}%")

    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "features": features,
            "grid": {"size": 15, "center_lat": 28.6280, "center_lon": 77.2090, "city": "Delhi"},
            "metrics": {"rmse": round(rmse, 3), "mae": round(mae, 3), "within_15pct": round(within_15pct, 1)},
            "best_iteration": int(model.best_iteration_) if getattr(model, "best_iteration_", None) else None,
            "training_rows": int(len(clean)),
            "train_rows": int(len(X_tr)),
            "val_rows": int(len(X_val)),
            "trained_at": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

if __name__ == "__main__": main()