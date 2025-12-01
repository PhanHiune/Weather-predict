# src/build_labels_hourly.py
import pandas as pd
from pathlib import Path

IN  = Path("data/historical_hourly.csv")
OUT = Path("data/historical_hourly_labeled.csv")
RAIN_THRESHOLD_MM = 0.1   # > 0.0 là có mưa; bạn chỉnh tùy ý

df = pd.read_csv(IN, parse_dates=["time"])

if "precipitation" in df.columns:
    df["rain_hour"] = (df["precipitation"] > RAIN_THRESHOLD_MM).astype("int8")
elif "weathercode" in df.columns:
    rainy_codes = {51,53,55,56,57,61,63,65,66,67,80,81,82,95,96,99}
    df["rain_hour"] = df["weathercode"].isin(rainy_codes).astype("int8")
else:
    raise SystemExit(
        "CSV chưa có precipitation/weathercode. "
        "Hãy dùng script merge_and_label_hourly.py ở Bước 3 (Trường hợp B)."
    )

df["hour_of_day"] = df["time"].dt.hour

# KHÔNG đưa precipitation làm feature nếu label = (precip > 0) cùng giờ
keep = [
    "time", "temperature", "humidity", "cloud", "pressure", "wind",
    "hour_of_day", "rain_hour"
]
df[keep].to_csv(OUT, index=False)
print("✔ Labeled file saved:", OUT)
