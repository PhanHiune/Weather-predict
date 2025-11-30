# main_hourly.py
import os, sys
import pandas as pd
from src.build_hourly_dataset import build_hourly_csv
from src.train_hourly import train_hourly_model
from src.predict_hourly import predict_hourly_next_48h
from src.visualize import save_hourly_report_csv, plot_three_figures_once

def parse_args():
    start = "2020-01-01"; chunk = 30
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

    # Đảm bảo CSV có nhãn 3 lớp
    if not os.path.exists(hourly_csv):
        print("🔄 Chưa có historical_hourly.csv → đang tạo...")
    else:
        print("✅ Đã có historical_hourly.csv")
    build_hourly_csv(
        hourly_csv, start_date=start_date,
        rain_threshold_mm=0.2, cloud_threshold=60.0,
        chunk_days=chunk_days, relabel_if_exists=True
    )

    # Train nếu cần
    if not os.path.exists(model_path):
        print("🔄 Chưa có hourly model → đang train (3 lớp)...")
        train_hourly_model()
    else:
        print("✅ Đã có hourly model (3 lớp)")

    # Dự đoán
    print("🔮 Dự đoán 48h tới (sunny/cloudy/rain)...")
    df_pred = predict_hourly_next_48h()

    # Xuất report CSV
    csv_path = os.path.join(base, "output", "reports", "hourly_report.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    save_hourly_report_csv(df_pred, csv_path)
    print("✔ Đã lưu báo cáo CSV:", csv_path)

    # TẠO & LƯU & HIỂN THỊ CÙNG LÚC 3 FIGURE
    out_dir = os.path.join(base, "output", "charts")
    result = plot_three_figures_once(df_pred, out_dir=out_dir, threshold=0.5)
    print("✔ Đã lưu hình:", result["dashboard_headline_png"])
    print("✔ Đã lưu hình:", result["condition_bands_png"])
    print("✔ Đã lưu hình:", result["rain_prob_png"])
    print("📝", result["headline_text"])
