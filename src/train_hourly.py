# src/train_hourly.py
import os, joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HOURLY = os.path.join(BASE_DIR, "data", "historical_hourly.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_hourly_rain.pkl")
MODEL_RAW  = os.path.join(MODEL_DIR, "xgb_hourly.json")

# KHÔNG dùng cloud/precip để tránh leakage
FEATURES_HOURLY = ["temperature", "humidity", "pressure", "wind", "hour_of_day"]
CLASS_ORDER = ["sunny","cloudy","rain"]

def _ensure_label3(df: pd.DataFrame, rain_thr=0.2, cloud_thr=60.0) -> pd.DataFrame:
    if "label3_id" in df.columns and "label3" in df.columns:
        return df
    if "precip" not in df.columns or "cloud" not in df.columns:
        raise ValueError("CSV thiếu 'precip'/'cloud' nên không thể gán nhãn 3 lớp.")
    def _lab(p, c):
        p = 0.0 if pd.isna(p) else p
        c = 0.0 if pd.isna(c) else c
        if p > rain_thr: return "rain"
        if c > cloud_thr: return "cloudy"
        return "sunny"
    df["label3"] = [_lab(p, c) for p, c in zip(df["precip"], df["cloud"])]
    id_map = {c:i for i,c in enumerate(CLASS_ORDER)}
    df["label3_id"] = df["label3"].map(id_map).astype("int8")
    return df

def train_hourly_model():
    if not os.path.exists(DATA_HOURLY):
        raise FileNotFoundError(f"Chưa có {DATA_HOURLY}. Hãy chạy build_hourly_csv trước.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_HOURLY)
    df = _ensure_label3(df)  # dự phòng nếu file cũ chưa có nhãn

    X = df[FEATURES_HOURLY]
    y = df["label3_id"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES_HOURLY)
    dvalid = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURES_HOURLY)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.5,
        "tree_method": "hist",
        "nthread": 0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    proba = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    y_pred = np.asarray(proba).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_ORDER, digits=3, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    booster.save_model(MODEL_RAW)
    joblib.dump({"model_json": MODEL_RAW, "feature_names": FEATURES_HOURLY, "class_order": CLASS_ORDER}, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}\nRaw booster: {MODEL_RAW}")
