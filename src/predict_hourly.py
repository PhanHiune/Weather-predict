# src/predict_hourly.py
import os
import joblib
import pandas as pd
import zoneinfo
from datetime import datetime, timedelta
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model nhị phân: 1 = mưa, 0 = không mưa
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_hourly_rain.pkl")

LAT, LON = 16.0471, 108.2068
VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")

FEATURES_HOURLY = [
    "temperature",
    "humidity",
    "cloud",
    "pressure",
    "wind",
    "hour_of_day",
]

# ========= 1) HÀM LẤY DỮ LIỆU DỰ BÁO 48H TỚI =========
def fetch_forecast_hourly_48h():
    """
    Gọi API open-meteo, lấy dự báo theo giờ cho 2 ngày tới,
    cắt đoạn [now, now + 48h), và trả về dataframe với các feature cần thiết.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relativehumidity_2m,cloudcover,pressure_msl,precipitation,wind_speed_10m"
        "&timezone=Asia%2FHo_Chi_Minh&forecast_days=3"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    # CHÚ Ý: key trong JSON là "relativehumidity_2m" (không có dấu "_")
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relativehumidity_2m"],
            "cloud": data["hourly"]["cloudcover"],
            "pressure": data["hourly"]["pressure_msl"],
            "wind": data["hourly"]["wind_speed_10m"],
            "precipitation": data["hourly"]["precipitation"],
        }
    )

    # thêm giờ trong ngày
    df["hour_of_day"] = df["time"].dt.hour

    # Lấy từ giờ hiện tại đến +48h (cắt chặt)
    now = datetime.now(VN_TZ)
    end = now + timedelta(hours=48)
    start_hour = now.replace(minute=0, second=0, microsecond=0)

    mask = (df["time"] >= start_hour) & (df["time"] < end)
    df = df.loc[mask].copy()

    # đảm bảo lại hour_of_day sau khi cắt
    df["hour_of_day"] = df["time"].dt.hour

    return df


# ========= 2) DỰ ĐOÁN + GÁN NHÃN 0/1/2 =========
def predict_hourly_next_48h(threshold: float = 0.5) -> pd.DataFrame:
    """
    Dự đoán mưa 48h tới, sau đó gán nhãn 3 lớp:

        weather_label:
            1 = Trời mưa
            2 = Trời mát mẻ  (không mưa & nhiệt độ < 27)
            0 = Trời nắng   (không mưa & nhiệt độ >= 27)

        weather_text: mô tả tiếng Việt tương ứng.

    threshold: ngưỡng xác suất để coi là "sẽ mưa".
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chưa có hourly model. Hãy train_hourly_model trước.")

    bundle = joblib.load(MODEL_PATH)

    # Hỗ trợ cả trường hợp bạn lưu model dạng thuần và dạng dict {"model":..., "features":...}
    if isinstance(bundle, dict):
        model = bundle["model"]
        trained_feats = bundle.get("features", FEATURES_HOURLY)
    else:
        model = bundle
        trained_feats = FEATURES_HOURLY

    if list(trained_feats) != FEATURES_HOURLY:
        raise ValueError(f"Feature mismatch! trained={trained_feats} current={FEATURES_HOURLY}")

    # Lấy forecast 48h tới
    df = fetch_forecast_hourly_48h()

    # Chuẩn bị dữ liệu đầu vào cho model
    X = df[FEATURES_HOURLY]

    # Model nhị phân: predict_proba trả về [P(class=0), P(class=1)]
    proba = model.predict_proba(X)[:, 1]  # xác suất mưa (class 1)

    # Chuẩn bị output
    df_out = df[
        [
            "time",
            "temperature",
            "humidity",
            "cloud",
            "pressure",
            "wind",
            "precipitation",
        ]
    ].copy()
    df_out = df_out.rename(columns={"precipitation": "precip_forecast"})

    df_out["rain_prob"] = proba
    df_out["will_rain"] = (df_out["rain_prob"] >= threshold).astype(int)

    # =========== GÁN NHÃN 0/1/2 =============
    # Rule:
    #   - Nếu will_rain == 1 → 1 (mưa)
    #   - Nếu will_rain == 0 & temperature < 27 → 2 (mát mẻ)
    #   - Nếu will_rain == 0 & temperature >= 27 → 0 (nắng)
    def _weather_label_row(row):
        if row["will_rain"] == 1:
            return 1  # mưa
        elif row["temperature"] < 27:
            return 2  # mát mẻ
        else:
            return 0  # nắng / nóng

    df_out["weather_label"] = df_out.apply(_weather_label_row, axis=1)
    df_out["weather_text"] = df_out["weather_label"].map(
        {
            0: "Trời nắng / khô, nóng",
            1: "Trời mưa",
            2: "Trời mát mẻ",
        }
    )

    return df_out
