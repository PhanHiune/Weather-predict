# src/build_hourly_dataset.py
import os
import pandas as pd
from datetime import datetime, timedelta
import zoneinfo
from src.api import fetch_archive_hourly

VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

def build_hourly_csv(csv_out: str, start_date: str = "2020-01-01", rain_threshold_mm: float = 0.1, chunk_days: int = 30):
    end_date = (datetime.now(VN_TZ) - timedelta(days=1)).date().isoformat()
    try:
        dfh = fetch_archive_hourly(start_date, end_date, chunk_days=chunk_days)
    except Exception as e:
        # fallback: thử giảm chunk_days nếu lỗi timeout kéo dài
        print(f"⚠️ Lỗi khi tải với chunk={chunk_days}: {e}\n→ Thử lại với chunk_days=15")
        dfh = fetch_archive_hourly(start_date, end_date, chunk_days=15)

    dfh["hour_of_day"] = dfh["time"].dt.hour
    dfh["rain_hour"] = (dfh["precip"] > rain_threshold_mm).astype("int8")

    keep = ["time", "temperature", "humidity", "cloud", "pressure", "wind", "hour_of_day", "rain_hour"]
    dfo = dfh[keep].copy()

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    dfo.to_csv(csv_out, index=False)
    print(f"✔ Đã lưu hourly dataset: {csv_out}  ({len(dfo)} rows)")
