# src/preprocess_daily.py
import pandas as pd

def make_daily_features(df_hourly: pd.DataFrame):
    """
    df_hourly: DataFrame từ file data/historical_weather.csv
      cột: time, temperature, humidity, cloud, pressure, rain, wind (giờ)
    Trả về: X, y, daily (y là nhãn NGÀY T+1)
    """
    df = df_hourly.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    daily = df.groupby("date").agg({
        "temperature": "mean",
        "humidity": "mean",
        "cloud": "mean",
        "pressure": "mean",
        "wind": "mean",
        "rain": "sum",   # tổng mưa trong ngày
    }).reset_index()

    # Quy tắc nhãn theo ngày hiện tại (t)
    def label_today(row):
        if row["rain"] > 2.0:     # >2mm coi là mưa (bạn có thể chỉnh ngưỡng)
            return 2              # Mưa
        elif row["cloud"] > 60:   # >60% coi là có mây
            return 1              # Có mây
        else:
            return 0              # Nắng

    daily["label_today"] = daily.apply(label_today, axis=1)

    # Target là nhãn NGÀY T+1
    daily["label_next_day"] = daily["label_today"].shift(-1)
    daily = daily.dropna(subset=["label_next_day"]).copy()
    daily["label_next_day"] = daily["label_next_day"].astype(int)

    X = daily[["temperature", "humidity", "cloud", "pressure", "wind", "rain"]]
    y = daily["label_next_day"]
    return X, y, daily
