# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.preprocess import add_label_id

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DAILY = os.path.join(BASE_DIR, "data", "historical_daily.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

FEATURES = [
    "temp_mean", "temp_max", "temp_min",
    "cloud_mean", "cloud_max",
    "humidity_mean",
    "pressure_mean",
    "wind_mean", "wind_max",
    "precip_sum",
]

def train_model():
    if not os.path.exists(DATA_DAILY):
        raise FileNotFoundError(
            f"Chưa có {DATA_DAILY}. Hãy chạy build_daily_dataset trước."
        )

    df = pd.read_csv(DATA_DAILY)
    df = add_label_id(df)

    X = df[FEATURES]
    y = df["label_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Train acc:", model.score(X_train, y_train))
    print("Test  acc:", model.score(X_test, y_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"✔ Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
