# src/relabel_from_rain.py
from pathlib import Path
import pandas as pd

IN  = Path("data/historical_weather.csv")
OUT = Path("data/historical_hourly_labeled.csv")

def relabel():
    df = pd.read_csv(IN, parse_dates=["time"])

    # thêm cột giờ trong ngày
    df["hour_of_day"] = df["time"].dt.hour

    # ---------- 1) Nhãn nhị phân: mưa / không mưa ----------
    # 1 = mưa, 0 = không mưa
    df["rain_hour"] = (df["rain"] > 0).astype("int8")

    # ---------- 2) Nhãn 3 lớp ----------
    def make_weather_label(row):
        if row["rain"] > 0:
            return 1                     # mưa
        elif row["rain"] == 0 and row["temperature"] < 27:
            return 2                     # mát mẻ
        else:
            return 0                     # nắng / nóng

    df["weather_label"] = df.apply(make_weather_label, axis=1)

    # thêm cột text cho dễ đọc (không dùng train)
    def classify_condition(row):
        if row["weather_label"] == 1:
            return "mua"
        elif row["weather_label"] == 2:
            return "mat_me"
        else:
            return "nang"

    df["condition"] = df.apply(classify_condition, axis=1)

    keep = [
        "time",
        "temperature",
        "humidity",
        "cloud",
        "pressure",
        "wind",
        "hour_of_day",
        "rain_hour",       # nhãn 0/1 mưa / không mưa
        "weather_label",   # nhãn 0/1/2 cho bài toán 3 lớp
        "condition",       # mô tả text
    ]

    df_out = df[keep]
    df_out.to_csv(OUT, index=False)

    print(f"✔ Đã gán nhãn xong, lưu vào: {OUT}")
    print("📊 rain_hour (1=mưa,0=không mưa):")
    print(df_out["rain_hour"].value_counts())
    print("\n📊 weather_label (0=nắng,1=mưa,2=mát):")
    print(df_out["weather_label"].value_counts())

if __name__ == "__main__":
    relabel()
