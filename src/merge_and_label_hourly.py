# src/merge_and_label_hourly.py
import requests
import pandas as pd
from pathlib import Path

# ========= CẤU HÌNH (đổi theo dữ liệu của bạn) =========
LAT, LON = 10.8231, 106.6297             # HCM mặc định
START, END = "2020-01-01", "2020-12-31"  # khoảng thời gian trùng với CSV gốc
RAIN_THRESHOLD_MM = 0.1                  # >0.0 là có mưa (bạn có thể chỉnh)

IN  = Path("data/historical_hourly.csv")           # CSV gốc của bạn
OUT = Path("data/historical_hourly_labeled.csv")   # CSV sau khi gán nhãn
URL = "https://archive-api.open-meteo.com/v1/archive"

# ========= 1) Đọc CSV gốc =========
df = pd.read_csv(IN)

# Ép về datetime + bóc timezone nếu có (fix lỗi bạn gặp)
df["time"] = pd.to_datetime(df["time"], errors="coerce")
if getattr(df["time"].dt, "tz", None) is not None:
    # BÓC timezone về dạng naive, giữ nguyên “giờ treo tường”
    df["time"] = df["time"].dt.tz_localize(None)

# Làm tròn về đầu giờ để khớp map theo giờ
df["time"] = df["time"].dt.floor("H")

# ========= 2) Gọi Open-Meteo lấy precipitation / weathercode =========
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START,
    "end_date": END,
    "hourly": "precipitation,weathercode",
    "timezone": "auto",   # trả về theo local time
}

r = requests.get(URL, params=params, timeout=60)
r.raise_for_status()
data = r.json()

lab = pd.DataFrame(data["hourly"])
lab["time"] = pd.to_datetime(lab["time"], errors="coerce")
# Thông thường cột này là NAIVE (không tz). Đưa về chuẩn giờ (floor) giống df:
lab["time"] = lab["time"].dt.floor("H")

# ========= 3) Merge an toàn theo time =========
# Thử merge trực tiếp (nhanh, nếu timestamps trùng 1-1)
try:
    merged = df.merge(lab, on="time", how="left", validate="m:1")
except Exception as e:
    # Nếu vẫn có lệch phút/giây, dùng merge_asof với tolerance 1h
    df_sorted  = df.sort_values("time")
    lab_sorted = lab.sort_values("time")
    merged = pd.merge_asof(
        df_sorted, lab_sorted,
        on="time", direction="nearest",
        tolerance=pd.Timedelta("1H")   # cho phép lệch trong 1 giờ
    )

# ========= 4) Tạo nhãn rain_hour =========
if "precipitation" in merged.columns:
    merged["rain_hour"] = (merged["precipitation"] > RAIN_THRESHOLD_MM).astype("int8")
elif "weathercode" in merged.columns:
    rainy_codes = {51,53,55,56,57,61,63,65,66,67,80,81,82,95,96,99}
    merged["rain_hour"] = merged["weathercode"].isin(rainy_codes).astype("int8")
else:
    raise SystemExit("Không có precipitation/weathercode sau khi merge.")

# ========= 5) Tạo feature giờ & CHỌN CỘT =========
merged["hour_of_day"] = pd.to_datetime(merged["time"]).dt.hour

# Không dùng precipitation làm feature nếu label = (precipitation>0) cùng giờ
keep = [
    "time",
    "temperature", "humidity", "cloud", "pressure", "wind",
    "hour_of_day",
    "rain_hour"
]

# Nếu CSV gốc của bạn đặt tên cột khác, map về chuẩn ở đây:
rename_map = {
    # ví dụ: "temperature_2m": "temperature",
    #        "relative_humidity_2m": "humidity",
    #        "cloudcover": "cloud",
    #        "pressure_msl": "pressure",
    #        "wind_speed_10m": "wind",
}
merged = merged.rename(columns=rename_map)

missing = [c for c in keep if c not in merged.columns]
if missing:
    raise SystemExit(f"Thiếu cột sau khi merge/rename: {missing}.\n"
                     f"Hãy cập nhật rename_map cho đúng tên cột của bạn.")

# ========= 6) Ghi file =========
OUT.parent.mkdir(parents=True, exist_ok=True)
merged[keep].to_csv(OUT, index=False, date_format="%Y-%m-%d %H:%M:%S")
print("✔ Labeled file saved:", OUT)
