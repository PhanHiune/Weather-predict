# src/train_weather_label.py
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

DATA = Path("data/historical_hourly_labeled.csv")
MODEL_OUT = Path("models/xgb_weather_label.pkl")

FEATURES = ["temperature", "humidity", "cloud", "pressure", "wind", "hour_of_day"]
TARGET = "weather_label"

def main():
    df = pd.read_csv(DATA, parse_dates=["time"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBClassifier cho bài toán multi-class (3 lớp)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✔ Accuracy (test): {acc:.4f}")
    print("🔎 Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["nắng (0)", "mưa (1)", "mát mẻ (2)"],
        )
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"💾 Đã lưu model vào: {MODEL_OUT}")

if __name__ == "__main__":
    main()
