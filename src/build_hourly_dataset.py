# src/build_hourly_dataset.py
import os, time, typing as t, requests
import pandas as pd
from datetime import datetime, timedelta

DEFAULT_LAT = 16.0471
DEFAULT_LON = 108.2068
DEFAULT_TZ  = "Asia/Ho_Chi_Minh"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
]

KEEP_COLS = [
    "time",
    "temperature", "humidity", "cloud", "pressure", "wind",
    "hour_of_day",
    "precip",
    "label3", "label3_id",
    "rain_hour",
]

CLASS_ORDER = ["sunny", "cloudy", "rain"]

def _fetch_openmeteo_hourly_chunk(lat, lon, start_date, end_date, tz):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "timezone": tz, "hourly": ",".join(HOURLY_VARS)
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"time": h["time"]})
    mapping = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "cloud_cover": "cloud",
        "pressure_msl": "pressure",
        "wind_speed_10m": "wind",
        "precipitation": "precip",
    }
    for src, dst in mapping.items():
        df[dst] = pd.to_numeric(h.get(src, []), errors="coerce")
    df["time"] = pd.to_datetime(df["time"])
    df["hour_of_day"] = df["time"].dt.hour
    return df

def _label3(precip: float, cloud: float, p_thr=0.2, c_thr=60.0) -> str:
    if pd.isna(precip): precip = 0.0
    if pd.isna(cloud):  cloud  = 0.0
    if precip > p_thr:  return "rain"
    if cloud  > c_thr:  return "cloudy"
    return "sunny"

def _ensure_labels(df: pd.DataFrame, rain_threshold_mm=0.2, cloud_threshold=60.0) -> pd.DataFrame:
    # Tính thiếu cột nào thì bổ sung cột đó
    if "rain_hour" not in df.columns:
        if "precip" not in df.columns:
            raise ValueError("Thiếu cột 'precip' trong dữ liệu, không thể gán nhãn.")
        df["rain_hour"] = (pd.to_numeric(df["precip"], errors="coerce") > float(rain_threshold_mm)).astype("int8")

    if "label3" not in df.columns or "label3_id" not in df.columns:
        if "precip" not in df.columns or "cloud" not in df.columns:
            raise ValueError("Thiếu 'precip' hoặc 'cloud' trong dữ liệu, không thể gán nhãn 3 lớp.")
        labels = [_label3(p, c, rain_threshold_mm, cloud_threshold) for p, c in zip(df["precip"], df["cloud"])]
        df["label3"] = labels
        id_map = {c: i for i, c in enumerate(["sunny", "cloudy", "rain"])}
        df["label3_id"] = df["label3"].map(id_map).astype("int8")

    # Chuẩn schema & sắp xếp
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[KEEP_COLS]

def build_hourly_csv(
    out_csv_path: str,
    start_date: str = "2020-01-01",
    end_date: t.Optional[str] = None,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    tz: str = DEFAULT_TZ,
    rain_threshold_mm: float = 0.2,
    cloud_threshold: float = 60.0,
    chunk_days: int = 30,
    relabel_if_exists: bool = True,
) -> None:
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    # Nếu file đã tồn tại, chỉ cần bổ sung nhãn cho chắc chắn
    if os.path.exists(out_csv_path) and relabel_if_exists:
        df = pd.read_csv(out_csv_path)
        df = _ensure_labels(df, rain_threshold_mm, cloud_threshold)
        df.to_csv(out_csv_path, index=False)
        print(f"✅ Đã bổ sung nhãn vào {out_csv_path} ({len(df)} dòng)")
        return

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cur  = datetime.strptime(start_date, "%Y-%m-%d")
    endd = datetime.strptime(end_date,   "%Y-%m-%d")
    parts = []
    while cur <= endd:
        chunk_end = min(cur + timedelta(days=chunk_days-1), endd)
        s, e = cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        print(f"⏳ Fetch {s} → {e} ...")
        try:
            parts.append(_fetch_openmeteo_hourly_chunk(lat, lon, s, e, tz))
        except Exception as ex:
            print("⚠️  Lỗi chunk:", ex)
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.25)

    if not parts:
        raise RuntimeError("Không có dữ liệu hourly.")

    df = pd.concat(parts, ignore_index=True)
    df = _ensure_labels(df, rain_threshold_mm, cloud_threshold)
    df.to_csv(out_csv_path, index=False)
    print(f"✅ Đã lưu {out_csv_path} ({len(df)} dòng)")
