import os
from src.build_daily_dataset import build_daily_csv
from src.train import train_model
from src.predict import predict_tomorrow
from src.predict import predict_tomorrow_details
from src.visualize import plot_hourly_show
from src.visualize import plot_hourly_dashboard

def ensure_data_and_model():
    base = os.path.dirname(os.path.abspath(__file__))
    daily_csv = os.path.join(base, "data", "historical_daily.csv")
    model_path = os.path.join(base, "models", "xgboost_model.pkl")

    if not os.path.exists(daily_csv):
        print("🔄 Chưa có historical_daily.csv → đang tạo từ archive (Open-Meteo)...")
        build_daily_csv(daily_csv, start_date="2020-01-01")
    else:
        print("✅ Đã có historical_daily.csv")

    if not os.path.exists(model_path):
        print("🔄 Chưa có model → đang train...")
        train_model()
    else:
        print("✅ Đã có model")

if __name__ == "__main__":
    ensure_data_and_model()
    print("🔮 Dự đoán 3 ngày (Đà Nẵng, VN):")

    from src.predict import predict_next_3_days
    from src.visualize import plot_hourly_dashboard_with_probs
    import os

    results, hourly_df, display_date = predict_next_3_days()

    # In console
    for r in results:
        print(f" - {r['date'].isoformat()}: {r['label']} | mưa ~ {round(r['rain_prob']*100,1)}%")

    # Build list (date, rain_prob_percent)
    three_day_probs = [(r["date"], round(r["rain_prob"]*100, 1)) for r in results]
    # Tìm kết quả của ngày hiển thị (mặc định: ngày mai nếu có)
    disp = next((r for r in results if r["date"] == display_date), results[0])
    rain_percent = round(disp["rain_prob"]*100, 1)
    label = disp["label"]

    # Vẽ 1 khung hình gồm 6 line + 1 bar 3 ngày
    save_png = os.path.join("output", "charts", f"dashboard_3days_{display_date}.png")
    plot_hourly_dashboard_with_probs(
        hourly_df=hourly_df,
        target_date=display_date,
        predicted_label=label,
        rain_prob_percent=rain_percent,
        three_day_probs=three_day_probs,
        save_path=save_png,
        show=True
    )
    print("👉 Dashboard đã lưu tại:", save_png)
