# src/predict.py
import os
import joblib
import numpy as np
from src.forecast_features import get_tomorrow_hourly_and_features
from src.visualize import plot_hourly
from src.forecast_features import get_next_3_days_hourly_and_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

FEATURES = [
    "temp_mean", "temp_max", "temp_min",
    "cloud_mean", "cloud_max",
    "humidity_mean",
    "pressure_mean",
    "wind_mean", "wind_max",
    "precip_sum",
]

ID2LABEL = {0: "Nắng", 1: "Có mây", 2: "Mưa"}
RAIN_CLASS_ID = 2  # mapping theo train

def predict_tomorrow():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chưa có model. Hãy train trước.")

    bundle = joblib.load(MODEL_PATH)
    # Cho phép cả 2 kiểu lưu: model trực tiếp hoặc dict {"model":..., "features":[...]}
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        trained_feats = bundle.get("features")
        if trained_feats and list(trained_feats) != FEATURES:
            raise ValueError(f"Feature mismatch! trained={trained_feats} current={FEATURES}")
    else:
        model = bundle

    feats_df, hourly_df, target_date = get_tomorrow_hourly_and_features()
    X = feats_df[FEATURES]

    # Dự đoán lớp + xác suất mưa
    pred_id = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]  # array theo thứ tự lớp đã fit
        rain_prob = float(proba[RAIN_CLASS_ID])  # 0..1
    else:
        rain_prob = float(pred_id == RAIN_CLASS_ID)

    label_vi = ID2LABEL.get(pred_id, str(pred_id))
    rain_prob_percent = round(rain_prob * 100, 1)

    # Vẽ biểu đồ hourly ngày mục tiêu, gắn kèm kết quả dự đoán
    charts_dir = os.path.join(OUTPUT_DIR, "charts")
    out_imgs = plot_hourly(hourly_df, charts_dir, target_date, label_vi, rain_prob_percent)

    result = {
        "target_date": target_date.isoformat(),
        "label": label_vi,
        "rain_probability_percent": rain_prob_percent,
        "charts": out_imgs
    }
    return result
def predict_tomorrow_details():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chưa có model. Hãy train trước.")

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        trained_feats = bundle.get("features")
        if trained_feats and list(trained_feats) != FEATURES:
            raise ValueError(f"Feature mismatch! trained={trained_feats} current={FEATURES}")
    else:
        model = bundle

    feats_df, hourly_df, target_date = get_tomorrow_hourly_and_features()
    X = feats_df[FEATURES]

    pred_id = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        rain_prob = float(proba[2])  # lớp 'Mưa' = id 2
    else:
        rain_prob = float(pred_id == 2)

    label_vi = ID2LABEL.get(pred_id, str(pred_id))
    rain_prob_percent = round(rain_prob * 100, 1)

    result = {
        "target_date": target_date.isoformat(),
        "label": label_vi,
        "rain_probability_percent": rain_prob_percent,
    }
    return result, hourly_df, target_date
def predict_next_3_days():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chưa có model. Hãy train trước.")

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        trained_feats = bundle.get("features")
        if trained_feats and list(trained_feats) != FEATURES:
            raise ValueError(f"Feature mismatch! trained={trained_feats} current={FEATURES}")
    else:
        model = bundle

    days = get_next_3_days_hourly_and_features()
    results = []
    target_display_idx = None  # chọn ngày mai nếu có

    for i, d in enumerate(days):
        X = d["features"][FEATURES]
        pred_id = int(model.predict(X)[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            rain_prob = float(proba[2])
        else:
            rain_prob = float(pred_id == 2)
        results.append({
            "date": d["date"],
            "label": ID2LABEL.get(pred_id, str(pred_id)),
            "rain_prob": rain_prob
        })
        # pick tomorrow for display
        if target_display_idx is None and d["date"] > days[0]["date"]:
            target_display_idx = i

    if target_display_idx is None:
        target_display_idx = 0  # fallback: hôm nay

    hourly_for_display = days[target_display_idx]["hourly"]
    display_date = days[target_display_idx]["date"]
    return results, hourly_for_display, display_date