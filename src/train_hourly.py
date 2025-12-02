# train_hourly_rain.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

CSV = "historical_hourly_labeled.csv"

df = pd.read_csv(CSV, parse_dates=["time"])
FEATURES = ["temperature","humidity","cloud","pressure","wind","hour_of_day"]
X = df[FEATURES]
y = df["rain_hour"]

# cân bằng lớp (tuỳ dữ liệu)
pos = y.sum()
neg = (y==0).sum()
scale_pos_weight = max(1.0, neg / max(1, pos))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.9, colsample_bytree=0.9,
    reg_lambda=1.0, n_jobs=-1, random_state=42, tree_method="hist",
    scale_pos_weight=scale_pos_weight
)
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    early_stopping_rounds=100,
    verbose=False
)

proba = clf.predict_proba(X_test)[:,1]
print("AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, (proba>=0.5).astype(int), digits=3))
