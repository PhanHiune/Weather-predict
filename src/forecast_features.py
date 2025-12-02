# src/forecast_features.py — build tomorrow daily features from forecast
import requests
import pandas as pd
import zoneinfo
from datetime import datetime, timedelta, time


LAT, LON = 16.0471, 108.2068
VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

def get_tomorrow_hourly_and_features():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,cloudcover,pressure_msl,precipitation,wind_speed_10m"
        "&timezone=Asia%2FHo_Chi_Minh&forecast_days=3"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Forecast API invalid response: {data}")

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "cloud": data["hourly"]["cloudcover"],
        "pressure": data["hourly"]["pressure_msl"],
        "precip": data["hourly"]["precipitation"],
        "wind": data["hourly"]["wind_speed_10m"],
    })
    # Các timestamp do API đã trả về theo timezone VN (vì ta set timezone),
    # nhưng chưa có tzinfo -> ta "gắn" VN_TZ vào:
    df["time"] = df["time"].dt.tz_localize(VN_TZ)

    today = datetime.now(VN_TZ).date()
    dates = sorted(df["time"].dt.date.unique())

    # Chọn ngày > hôm nay (thường là ngày mai)
    target_date = None
    for d in dates:
        if d > today:
            target_date = d
            break
    if target_date is None:
        target_date = dates[-1]  # fallback

    d = df[df["time"].dt.date == target_date].copy()
    if d.empty:
        raise RuntimeError("Không tìm thấy dữ liệu hourly của ngày mục tiêu trong forecast.")

    feats = pd.Series({
        "temp_mean": d["temperature"].mean(),
        "temp_max": d["temperature"].max(),
        "temp_min": d["temperature"].min(),
        "cloud_mean": d["cloud"].mean(),
        "cloud_max": d["cloud"].max(),
        "humidity_mean": d["humidity"].mean(),
        "pressure_mean": d["pressure"].mean(),
        "wind_mean": d["wind"].mean(),
        "wind_max": d["wind"].max(),
        "precip_sum": d["precip"].sum(),
    }).to_frame().T

    return feats, d, target_date
def _fetch_forecast_hourly_3days():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,cloudcover,pressure_msl,precipitation,wind_speed_10m"
        "&timezone=Asia%2FHo_Chi_Minh&forecast_days=3"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Forecast API invalid response: {data}")
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "cloud": data["hourly"]["cloudcover"],
        "pressure": data["hourly"]["pressure_msl"],
        "precip": data["hourly"]["precipitation"],
        "wind": data["hourly"]["wind_speed_10m"],
    })
    df["time"] = df["time"].dt.tz_localize(VN_TZ)
    return df

def get_next_3_days_hourly_and_features():
    df = _fetch_forecast_hourly_3days()
    today = datetime.now(VN_TZ).date()
    days = [today + timedelta(days=i) for i in range(3)]

    out = []
    for d in days:
        start = datetime.combine(d, time(0, 0)).replace(tzinfo=VN_TZ)
        end   = start + timedelta(days=1)
        mask = (df["time"] >= start) & (df["time"] < end)
        h = df.loc[mask].copy()
        if h.empty:
            continue
        feats = pd.Series({
            "temp_mean": h["temperature"].mean(),
            "temp_max": h["temperature"].max(),
            "temp_min": h["temperature"].min(),
            "cloud_mean": h["cloud"].mean(),
            "cloud_max": h["cloud"].max(),
            "humidity_mean": h["humidity"].mean(),
            "pressure_mean": h["pressure"].mean(),
            "wind_mean": h["wind"].mean(),
            "wind_max": h["wind"].max(),
            "precip_sum": h["precip"].sum(),
        }).to_frame().T
        out.append({"date": d, "hourly": h, "features": feats})
    return out
