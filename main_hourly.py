# main_hourly.py
import os, sys
from src.build_hourly_dataset import build_hourly_csv
from src.train_hourly import train_hourly_model
from src.predict_hourly import predict_hourly_next_48h
from src.visualize import plot_hourly_rain_dashboard

def parse_args():
    # ví dụ: python main_hourly.py --start 2022-01-01 --chunk 20
    start = "2020-01-01"
    chunk = 30
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--start" and i+1 < len(sys.argv):
            start = sys.argv[i+1]; i += 2
        elif sys.argv[i] == "--chunk" and i+1 < len(sys.argv):
            chunk = int(sys.argv[i+1]); i += 2
        else:
            i += 1
    return start, chunk

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    hourly_csv = os.path.join(base, "data", "historical_hourly.csv")
    model_path = os.path.join(base, "models", "xgboost_hourly_rain.pkl")

    start_date, chunk_days = parse_args()

    if not os.path.exists(hourly_csv):
        print(f"🔄 Chưa có historical_hourly.csv → đang tạo (start={start_date}, chunk={chunk_days})...")
        build_hourly_csv(hourly_csv, start_date=start_date, rain_threshold_mm=0.1, chunk_days=chunk_days)
    else:
        print("✅ Đã có historical_hourly.csv")

    if not os.path.exists(model_path):
        print("🔄 Chưa có hourly model → đang train...")
        train_hourly_model()
    else:
        print("✅ Đã có hourly model")

    print("🔮 Dự đoán xác suất mưa theo GIỜ (48h tới)...")
    df_pred = predict_hourly_next_48h()

    # In top giờ có xác suất mưa cao nhất
    top = df_pred.sort_values("rain_prob", ascending=False).head(8)
    print("👉 Các giờ dễ mưa nhất (top 8):")
    for _, r in top.iterrows():
        print(f"   - {r['time']}  ~  {r['rain_prob']*100:.1f}%  | precip_forecast≈{r['precip_forecast']}mm")

    # Vẽ dashboard 1 khung
    save_png = os.path.join(base, "output", "charts", "hourly_rain_48h.png")
    plot_hourly_rain_dashboard(df_pred, threshold=0.5, save_path=save_png, show=True)
    print("👉 Dashboard đã lưu tại:", save_png)
