# src/train_hourly.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HOURLY = os.path.join(BASE_DIR, "data", "historical_hourly.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_hourly_rain.pkl")

FEATURES_HOURLY = ["temperature", "humidity", "cloud", "pressure", "wind", "hour_of_day"]

def train_hourly_model():
    if not os.path.exists(DATA_HOURLY):
        raise FileNotFoundError(f"Chưa có {DATA_HOURLY}. Hãy chạy build_hourly_csv trước.")

    df = pd.read_csv(DATA_HOURLY)
    X = df[FEATURES_HOURLY]
    y = df["rain_hour"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=0
    )
    model.fit(X_train, y_train)

    print("Hourly Train acc:", model.score(X_train, y_train))
    print("Hourly Test  acc:", model.score(X_test, y_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Lưu kèm metadata features để tránh mismatch
    joblib.dump({"model": model, "features": FEATURES_HOURLY}, MODEL_PATH)
    print(f"✔ Hourly model saved: {MODEL_PATH}")

if __name__ == "__main__":
    train_hourly_model()
