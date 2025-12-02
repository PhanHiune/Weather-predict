# src/api.py — data access layer for Open-Meteo (robust)
import os
import math
import time as _time
import requests
import pandas as pd
from datetime import datetime, timedelta, date
import zoneinfo
from requests.adapters import HTTPAdapter, Retry

# ĐÀ NẴNG — toạ độ cố định
LAT, LON = 16.0471, 108.2068
VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

HOURLY = "temperature_2m,relative_humidity_2m,cloudcover,pressure_msl,precipitation,wind_speed_10m"

# ---------- Robust HTTP session ----------
def _requests_session(total=5, backoff=1.0):
    sess = requests.Session()
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

# ---------- Archive: tải 1 chunk ----------
def _fetch_archive_hourly_chunk(start_date: str, end_date: str, timeout=90, session=None) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={HOURLY}&timezone=Asia%2FHo_Chi_Minh"
    )
    s = session or _requests_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Archive API invalid response: {data}")

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

# ---------- Archive: tải cả dải theo nhiều chunk ----------
def fetch_archive_hourly(start_date: str, end_date: str, chunk_days: int = 30, pause_sec: float = 0.3) -> pd.DataFrame:
    s = _requests_session()
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if start > end:
        raise ValueError("start_date > end_date")

    out = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        tries = 0
        last_err = None
        while tries < 3:
            try:
                df = _fetch_archive_hourly_chunk(cur.isoformat(), chunk_end.isoformat(), timeout=120, session=s)
                out.append(df)
                break
            except requests.exceptions.ReadTimeout as e:
                last_err = e
                tries += 1
                # backoff nhẹ
                _time.sleep(1.5 * tries)
            except Exception as e:
                last_err = e
                tries += 1
                _time.sleep(1.0 * tries)
        if tries >= 3:
            raise RuntimeError(f"Failed to fetch chunk {cur}..{chunk_end}: {last_err}")

        # dịch tới chunk tiếp theo
        cur = chunk_end + timedelta(days=1)
        _time.sleep(pause_sec)

    if not out:
        raise RuntimeError("No data returned.")
    df_all = pd.concat(out, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return df_all

# ---------- Convenience: tải nhanh tới hôm qua và lưu CSV ----------
def download_historical_hourly_csv(csv_path: str, start_date: str = "2020-01-01", chunk_days: int = 30):
    end_date = (datetime.now(VN_TZ) - timedelta(days=1)).date().isoformat()
    df = fetch_archive_hourly(start_date, end_date, chunk_days=chunk_days)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✔ Đã lưu hourly archive: {csv_path}  ({len(df)} rows)")
