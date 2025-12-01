# src/train_hourly_rain.py
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

CSV   = Path("data/historical_hourly_labeled.csv")
MODEL = Path("models/xgb_hourly_rain.pkl")

df = pd.read_csv(CSV, parse_dates=["time"])

FEATURES = ["temperature","humidity","cloud","pressure","wind","hour_of_day"]
TARGET   = "rain_hour"

X = df[FEATURES]
y = df[TARGET].astype("int8")

# Cân bằng lớp (mưa thường ít)
pos = int(y.sum()); neg = int((y == 0).sum())
scale_pos_weight = max(1.0, neg / max(1, pos))

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# eval_metric để trong constructor (ổn cho cả nhiều phiên bản)
clf = XGBClassifier(
    n_estimators=1200,          # giảm một chút nếu không dùng early stopping
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
)

# ---- Train: thử nhiều chế độ để tương thích mọi phiên bản XGB ----
trained_with = None
try:
    # Cách 1: dùng callbacks (xgboost >= 2.1)
    import xgboost as xgb
    callbacks = [
        xgb.callback.EarlyStopping(rounds=100, save_best=True, maximize=True)
    ]
    clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], callbacks=callbacks, verbose=False)
    trained_with = "callbacks"
except TypeError:
    try:
        # Cách 2: dùng early_stopping_rounds trong fit (xgboost 1.x)
        clf.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                early_stopping_rounds=100, verbose=False)
        trained_with = "early_stopping_rounds"
    except TypeError:
        # Cách 3: không early stopping (mọi bản đều chạy được)
        clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        trained_with = "no_early_stopping"

print(f"Trained with: {trained_with}")

# ---- Đánh giá ----
proba = clf.predict_proba(Xte)[:, 1]
print("AUC:", roc_auc_score(yte, proba))
print(classification_report(yte, (proba >= 0.5).astype(int), digits=3))

best_iter = getattr(getattr(clf, "best_iteration", None), "__int__", lambda: None)()
if best_iter is not None:
    print("Best iteration:", best_iter)

MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": clf, "features": FEATURES}, MODEL)
print("Model saved:", MODEL)
