# src/build_daily_dataset.py — build daily-level dataset (features + label)
import os
import pandas as pd
from datetime import datetime
import zoneinfo
from src.api import fetch_archive_hourly

VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

def _label_day(row) -> str:
    # Quy tắc nhãn cơ bản; bạn có thể tinh chỉnh ngưỡng cho phù hợp thực tế Đà Nẵng
    if row["precip_sum"] > 0.2:
        return "rain"
    elif row["cloud_max"] > 60:
        return "cloudy"
    else:
        return "sunny"

def build_daily_csv(csv_out: str, start_date: str = "2020-01-01"):
    # Lấy hourly quá khứ đến hôm nay
    end_date = datetime.now(VN_TZ).date().isoformat()
    dfh = fetch_archive_hourly(start_date, end_date)  # 'time' đã tz-aware

    # Lấy phần ngày (date) theo VN
    dfh["date"] = dfh["time"].dt.date

    # Tổng hợp theo ngày bằng groupby().agg() (tránh .dt sau groupby)
    dfd = dfh.groupby("date", as_index=False).agg(
        temp_mean=("temperature", "mean"),
        temp_max=("temperature", "max"),
        temp_min=("temperature", "min"),
        cloud_mean=("cloud", "mean"),
        cloud_max=("cloud", "max"),
        humidity_mean=("humidity", "mean"),
        pressure_mean=("pressure", "mean"),
        wind_mean=("wind", "mean"),
        wind_max=("wind", "max"),
        precip_sum=("precip", "sum"),
    )

    # Gán nhãn
    dfd["label"] = dfd.apply(_label_day, axis=1)

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    dfd.to_csv(csv_out, index=False)
    print(f"✔ Đã lưu daily dataset: {csv_out}  ({len(dfd)} ngày)")
