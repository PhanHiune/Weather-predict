# src/predict_hourly.py
import os
import joblib
import pandas as pd
import zoneinfo
from datetime import datetime, timedelta, time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_hourly_rain.pkl")

LAT, LON = 16.0471, 108.2068
VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

FEATURES_HOURLY = ["temperature", "humidity", "cloud", "pressure", "wind", "hour_of_day"]

def fetch_forecast_hourly_48h():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relativehumidity_2m,cloudcover,pressure_msl,precipitation,wind_speed_10m"
        "&timezone=Asia%2FHo_Chi_Minh&forecast_days=3"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
    "time": pd.to_datetime(data["hourly"]["time"]),
    "temperature": data["hourly"]["temperature_2m"],
    "humidity": data["hourly"]["relative_humidity_2m"],
    "cloud": data["hourly"]["cloudcover"],
    "pressure": data["hourly"]["pressure_msl"],
    "wind": data["hourly"]["wind_speed_10m"],
    })
    df["hour_of_day"] = df["time"].dt.hour
    X = df[loaded["features"]]
    prob = loaded["model"].predict_proba(X)[:, 1]
    df["rain_prob"] = prob
    df["will_rain"] = (df["rain_prob"] >= 0.5).astype(int)


    # Lấy từ giờ hiện tại đến +48h (cắt chặt)
    now = datetime.now(VN_TZ)
    end = now + timedelta(hours=48)
    mask = (df["time"] >= now.replace(minute=0, second=0, microsecond=0)) & (df["time"] < end)
    df = df.loc[mask].copy()

    df["hour_of_day"] = df["time"].dt.hour
    return df

def predict_hourly_next_48h():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chưa có hourly model. Hãy train_hourly_model trước.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    trained_feats = bundle.get("features") if isinstance(bundle, dict) else FEATURES_HOURLY
    if list(trained_feats) != FEATURES_HOURLY:
        raise ValueError(f"Feature mismatch! trained={trained_feats} current={FEATURES_HOURLY}")

    df = fetch_forecast_hourly_48h()
    X = df[FEATURES_HOURLY]
    proba = model.predict_proba(X)[:, 1]
    df_out = df[["time","precip_forecast"]].copy()
    df_out["rain_prob"] = proba
    df_out["will_rain"] = (df_out["rain_prob"] >= threshold).astype(int)
    return df_out
