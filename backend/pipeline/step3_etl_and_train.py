# backend/pipeline/step3_etl_and_train.py
import os, json, time, requests, warnings, hashlib, platform, io, glob
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.ml.features import (
    PG_SIGMA_Y,
    PG_SIGMA_Z,
    diurnal_traffic_factor,
    estimate_boundary_layer_height,
    stability_class,
)
from app.ml.uncertainty import apply_conformal, calibrate_interval, interval_metrics

warnings.filterwarnings("ignore")

DATA_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAW_CSV_PATH      = os.path.join(DATA_DIR, "raw_openaq", "delhi_all_stations_2025.csv")
SOURCES_PATH      = os.path.join(DATA_DIR, "delhi_emission_sources.json")
ROAD_DENSITY_PATH = os.path.join(DATA_DIR, "delhi_grid_road_density.csv")
MODEL_PATH        = os.path.join(DATA_DIR, "surrogate_model.joblib")
META_PATH         = os.path.join(DATA_DIR, "model_meta.json")
UQ_MODEL_PATH     = os.path.join(DATA_DIR, "uncertainty_models.joblib")
MERGED_CSV_PATH   = os.path.join(DATA_DIR, "merged_training_data.csv")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_chunked_artifact(value, path: str, chunk_bytes: int = 500_000) -> list[str]:
    """Write a compressed joblib as GitHub-API-friendly deterministic chunks."""
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=9)
    payload = buffer.getvalue()
    for stale_path in glob.glob(f"{path}.part[0-9][0-9][0-9]"):
        os.unlink(stale_path)
    paths = []
    for index, offset in enumerate(range(0, len(payload), chunk_bytes)):
        part_path = f"{path}.part{index:03d}"
        with open(part_path, "wb") as part_file:
            part_file.write(payload[offset:offset + chunk_bytes])
        paths.append(part_path)
    return paths

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
            sub["weather_fallback"] = sub[["wind_speed_met", "wind_dir_met"]].isna().any(axis=1)
        except Exception:
            sub = grp[["location_id", "datetime_hour"]].copy()
            sub["wind_speed_met"] = grp["wind_speed"].values if "wind_speed" in grp.columns else np.nan
            sub["wind_dir_met"] = grp["wind_direction"].values if "wind_direction" in grp.columns else np.nan
            sub["blh_met"], sub["temp_met"], sub["rh_met"] = np.nan, np.nan, np.nan
            sub["weather_fallback"] = True
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
    
    out["stability_class"] = [stability_class(float(ws), int(h)) for ws, h in zip(out["wind_speed"], out["hour"])]
    out["boundary_layer_height"] = out.apply(lambda r: float(r["blh_met"]) if pd.notna(r.get("blh_met")) else estimate_boundary_layer_height(int(r["hour"]), int(r["stability_class"])), axis=1)

    wind_rad_arr = np.radians((out["wind_direction"].values + 180) % 360)
    flux = np.zeros(len(out))
    for s in sources:
        for i in range(len(out)):
            dx, dy = (out["longitude"].iloc[i] - s["lon"]) * 95.0, (out["latitude"].iloc[i] - s["lat"]) * 111.0
            wv, cv = np.array([np.sin(wind_rad_arr[i]), np.cos(wind_rad_arr[i])]), np.array([dx, dy])
            xd = np.dot(wv, cv)
            if xd >= 1.0:
                yc = float(np.linalg.norm(cv - xd * wv))
                sc, u = int(out["stability_class"].iloc[i]), max(out["wind_speed"].iloc[i], 0.5)
                sy, sz = max(PG_SIGMA_Y[sc] * (xd ** 0.894), 0.5), max(PG_SIGMA_Z[sc] * (xd ** 0.894), 0.3)
                flux[i] += float((s["intensity"] / (np.pi * u * sy * sz)) * np.exp(-0.5 * (yc / sy) ** 2))
                
    out["source_flux_log"] = np.log1p(flux) * (300.0 / np.maximum(out["boundary_layer_height"].values, 100.0)) * 5.0
    out["traffic_density"] = (out["road_density_m"] / 1000.0) * out.apply(lambda r: diurnal_traffic_factor(int(r["hour"]), int(r["day_of_week"])), axis=1) * 50
    # Fixed serving-grid origin; dataset means can change between downloads and
    # would otherwise shift the feature coordinate system after every retrain.
    out["x"] = ((out["longitude"] - 77.2090) / 0.0105).round(1)
    out["y"] = ((out["latitude"] - 28.6280) / 0.009).round(1)

    # Past-only fill: backward filling here would leak a future measurement.
    out["pm25_filled"] = out.groupby("location_id")["pm25"].transform(lambda s: s.ffill(limit=3))
    grouped = out.groupby("location_id", sort=False)
    previous_value_1 = grouped["pm25_filled"].shift(1)
    previous_time_1 = grouped["datetime_hour"].shift(1)
    previous_value_3 = grouped["pm25_filled"].shift(3)
    previous_time_3 = grouped["datetime_hour"].shift(3)
    out["lag_1h_pm25"] = previous_value_1.where(out["datetime_hour"] - previous_time_1 == pd.Timedelta(hours=1))
    out["lag_3h_pm25"] = previous_value_3.where(out["datetime_hour"] - previous_time_3 == pd.Timedelta(hours=3))
    for col in ["no2", "co", "pm10"]:
        if col in out.columns: out[col] = out.groupby("location_id")[col].transform(lambda s: s.ffill(limit=3))
    return out

def _chronological_blocks(df: pd.DataFrame):
    """70/15/15 split by timestamp, never by row or station ordering."""
    timestamps = np.array(sorted(df["datetime_hour"].dropna().unique()))
    if len(timestamps) < 20:
        raise RuntimeError("not enough unique timestamps for chronological train/calibration/test blocks")
    train_end = timestamps[max(1, int(len(timestamps) * 0.70)) - 1]
    calibration_end = timestamps[max(2, int(len(timestamps) * 0.85)) - 1]
    train = df[df["datetime_hour"] <= train_end].copy()
    calibration = df[(df["datetime_hour"] > train_end) & (df["datetime_hour"] <= calibration_end)].copy()
    test = df[df["datetime_hour"] > calibration_end].copy()
    return train, calibration, test, train_end, calibration_end


def _quantile_model(alpha: float, rows: int):
    return lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.035, num_leaves=47, max_depth=8,
        min_child_samples=max(10, rows // 300), subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.25,
        objective="quantile", alpha=alpha, n_jobs=-1, verbose=-1, random_state=42,
    )


def main():
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"raw OpenAQ data not found at {RAW_CSV_PATH}")
    if os.path.exists(SOURCES_PATH):
        with open(SOURCES_PATH, encoding="utf-8") as source_file:
            sources = json.load(source_file)
    else:
        sources = []
    df = engineer_features(merge_road_density(enrich_with_wind(load_and_pivot(RAW_CSV_PATH))), sources)
    df.to_csv(MERGED_CSV_PATH, index=False)

    # Exclude contemporaneous co-pollutants. They are unavailable in the UI at
    # prediction time (serving previously substituted constants), which inflated
    # offline performance and created train/serve skew.
    features = ["x", "y", "hour", "day_of_week", "month", "wind_speed", "wind_direction", "wind_sin", "wind_cos", "stability_class", "boundary_layer_height", "traffic_density", "source_flux_log", "lag_1h_pm25", "lag_3h_pm25"]
    candidate = df.dropna(subset=["datetime_hour", "pm25"]).sort_values("datetime_hour").copy()
    train, calibration, test, train_end, calibration_end = _chronological_blocks(candidate)

    # Fit every missing-value statistic on the training block only.
    fill_values = {}
    for feature in features:
        median = float(train[feature].median())
        if not np.isfinite(median):
            raise RuntimeError(f"feature {feature} has no finite training median")
        fill_values[feature] = median
        train[feature] = train[feature].fillna(median)
        calibration[feature] = calibration[feature].fillna(median)
        test[feature] = test[feature].fillna(median)

    X_train, y_train = train[features], train["pm25"]
    X_cal, y_cal = calibration[features], calibration["pm25"]
    X_test, y_test = test[features], test["pm25"]

    lower_model, median_model, upper_model = (_quantile_model(alpha, len(train)) for alpha in (0.10, 0.50, 0.90))
    for quantile_model in (lower_model, median_model, upper_model):
        quantile_model.fit(X_train, y_train)
    qhat = calibrate_interval(y_cal.values, lower_model.predict(X_cal), upper_model.predict(X_cal), coverage=0.90)
    test_lower, test_upper = apply_conformal(lower_model.predict(X_test), upper_model.predict(X_test), qhat)

    # The quantile median is both the served point estimate and the centre of
    # the reported band. A separate Huber model previously produced points far
    # outside its own intervals and weakened the persistence comparison.
    model = median_model
    predictions = np.clip(median_model.predict(X_test), 0.0, 500.0)
    persistence = np.clip(X_test["lag_1h_pm25"].to_numpy(), 0.0, 500.0)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, persistence)))
    baseline_mae = float(mean_absolute_error(y_test, persistence))
    skill = 1.0 - rmse / baseline_rmse if baseline_rmse else 0.0
    within_15 = float(np.mean(np.abs(y_test.values - predictions) <= 15) * 100)
    uq_metrics = interval_metrics(y_test.values, test_lower, test_upper)

    print(f"[step3] Chronological test — RMSE {rmse:.2f}, MAE {mae:.2f}, persistence RMSE {baseline_rmse:.2f}")
    print(f"[step3] 90% interval — coverage {uq_metrics['empirical_coverage']:.1%}, mean width {uq_metrics['mean_interval_width']:.2f}")

    training_data_sha256 = _sha256(RAW_CSV_PATH)
    source_inventory_sha256 = _sha256(SOURCES_PATH) if os.path.exists(SOURCES_PATH) else None
    artifact_provenance = {
        "random_seed": 42,
        "training_data_sha256": training_data_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "scikit_learn": version("scikit-learn"),
        "joblib": joblib.__version__, "requests": requests.__version__,
    }

    _dump_chunked_artifact(model, MODEL_PATH)
    _dump_chunked_artifact({
        "lower_model": lower_model, "median_model": median_model, "upper_model": upper_model,
        "qhat": qhat, "coverage": 0.90, "features": features,
        "provenance": artifact_provenance,
    }, UQ_MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "features": features,
            "feature_fill_values": fill_values,
            "grid": {"size": 15, "center_lat": 28.6280, "center_lon": 77.2090, "city": "Delhi"},
            "metrics": {
                "rmse": round(rmse, 3), "mae": round(mae, 3), "within_15pct": round(within_15, 1),
                "persistence_rmse": round(baseline_rmse, 3), "persistence_mae": round(baseline_mae, 3),
                "skill_vs_persistence": round(skill, 4),
                "interval_coverage": round(uq_metrics["empirical_coverage"], 4),
                "mean_interval_width": round(uq_metrics["mean_interval_width"], 3),
            },
            "evaluation": {
                "split": "chronological_70_train_15_calibration_15_test",
                "train_end": str(train_end), "calibration_end": str(calibration_end),
                "target_interval_coverage": 0.90, "conformal_qhat": round(float(qhat), 4),
                "product_scope": "conditional PM2.5 estimate; not a validated multi-horizon forecast",
            },
            "provenance": artifact_provenance,
            "data_quality": {
                "weather_fallback_fraction": round(float(df.get("weather_fallback", pd.Series(False, index=df.index)).mean()), 4),
                "serving_uses_default_lags_unless_explicitly_supplied": True,
                "contemporaneous_copollutants_excluded": True,
            },
            "best_iteration": int(model.best_iteration_) if getattr(model, "best_iteration_", None) else None,
            "train_rows": int(len(train)), "calibration_rows": int(len(calibration)), "test_rows": int(len(test)),
            "trained_at": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)

if __name__ == "__main__": main()
