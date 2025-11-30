# src/predict_hourly.py
import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_JSON = os.path.join(BASE_DIR, "models", "xgb_hourly.json")  # load JSON, không dùng .pkl

DEFAULT_LAT = 16.0471
DEFAULT_LON = 108.2068
DEFAULT_TZ  = "Asia/Ho_Chi_Minh"

# Trật tự lớp để hiển thị
CLASS_ORDER = ["sunny","cloudy","rain"]

def _fetch_forecast_48h(lat=DEFAULT_LAT, lon=DEFAULT_LON, tz=DEFAULT_TZ) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "forecast_hours": 48,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,pressure_msl,wind_speed_10m",
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
    }
    for src, dst in mapping.items():
        df[dst] = pd.to_numeric(h.get(src, []), errors="coerce")
    df["time"] = pd.to_datetime(df["time"])
    df["hour_of_day"] = df["time"].dt.hour
    return df

class _BoosterMultiWrapper:
    def __init__(self, booster: xgb.Booster, class_order):
        self.booster = booster
        # Lấy danh sách features đúng lúc train từ chính model
        self.feature_names = booster.feature_names
        # Một số bản JSON không có best_iteration – dùng full khi không có
        self.best_iteration = getattr(booster, "best_iteration", None)
        self.class_order = class_order

    def _dm(self, X: pd.DataFrame):
        # Chọn đúng các cột model đã train
        return xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame):
        dm = self._dm(X)
        if self.best_iteration is None:
            proba = self.booster.predict(dm)
        else:
            # một số phiên bản không hỗ trợ iteration_range – khi đó cũng dùng full
            try:
                proba = self.booster.predict(dm, iteration_range=(0, self.best_iteration + 1))
            except TypeError:
                proba = self.booster.predict(dm)
        return np.asarray(proba, dtype=float)  # (N, K)

    def predict_label(self, X: pd.DataFrame):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return [self.class_order[i] for i in idx], proba.max(axis=1)

def _load_model_from_json() -> _BoosterMultiWrapper:
    if not os.path.exists(MODEL_JSON):
        raise FileNotFoundError(f"Không thấy model JSON: {MODEL_JSON}. Hãy train trước.")
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    return _BoosterMultiWrapper(booster, CLASS_ORDER)

def predict_hourly_next_48h() -> pd.DataFrame:
    model = _load_model_from_json()
    df = _fetch_forecast_48h()

    proba = model.predict_proba(df)   # wrapper sẽ tự chọn đúng cột theo model
    # Nếu model là nhị phân (K=2) thì thêm cột cho đủ, nhưng mặc định ta train 3 lớp
    if proba.shape[1] == 2:
        # giả sử class_order = [no_rain, rain] -> map tạm về sunny/rain
        p_rain = proba[:, 1]
        proba = np.c_[1-p_rain, np.zeros_like(p_rain), p_rain]

    df["prob_sunny"]  = proba[:, 0]
    df["prob_cloudy"] = proba[:, 1]
    df["prob_rain"]   = proba[:, 2]

    labels, maxp = model.predict_label(df)
    df["pred_label"] = labels
    df["pred_prob"]  = maxp

    return df[["time","temperature","humidity","cloud","pressure","wind",
               "hour_of_day","prob_sunny","prob_cloudy","prob_rain",
               "pred_label","pred_prob"]]
