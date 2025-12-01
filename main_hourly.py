# src/main_hourly.py
# ------------------------------------------------------------
# MAIN: dự đoán mưa theo giờ bằng model đã train + vẽ heatmap
# ------------------------------------------------------------
import argparse
from pathlib import Path
import json
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
MODEL_PATH = Path("models/xgb_hourly_rain.pkl")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ngưỡng phân loại mưa
RAIN_PROB_THRESHOLD = 0.5

# Quy tắc phân loại "nắng / mát mẻ" (khi KHÔNG mưa)
# - Nắng: nhiệt độ >= HOT_TEMP_C và mây (cloud) < CLOUD_SUNNY_MAX
# - Mát mẻ: còn lại
HOT_TEMP_C = 32.0
CLOUD_SUNNY_MAX = 30  # %

# ===================== UTILS =====================
def fetch_forecast_hourly(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    """Lấy dự báo giờ từ Open-Meteo trong ~48-72h tới."""
    import requests, numpy as np, pandas as pd

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        # KHÔNG đưa 'time' vào đây; Open-Meteo tự trả 'time'
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "cloudcover",
            "pressure_msl",
            "windspeed_10m",      # <-- tên đúng (không có dấu _ giữa wind và speed)
            "weathercode",
            "precipitation",
        ]),
        "forecast_days": 3,       # ~72h, ta sẽ cắt còn 'hours' phía dưới
        "timezone": "auto",
    }

    r = requests.get(url, params=params, timeout=60)
    # Nếu lỗi, in thông điệp để dễ debug
    try:
        r.raise_for_status()
    except Exception as e:
        print(">>> URL gọi:", r.url)
        print(">>> Status:", r.status_code)
        print(">>> Nội dung trả về:", r.text[:1000])
        raise

    data = r.json()
    h = pd.DataFrame(data["hourly"])

    # Chuẩn hoá tên cột về schema train
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
        "cloud": h["cloudcover"],
        "pressure": h["pressure_msl"],
        "wind": h["windspeed_10m"],         # <-- map đúng tên mới
        "weathercode": h.get("weathercode", [np.nan]*len(h)),
        "precipitation": h.get("precipitation", [np.nan]*len(h)),
    })
    df["hour_of_day"] = df["time"].dt.hour
    df = df.sort_values("time").head(hours).reset_index(drop=True)
    return df


def load_model():
    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    features = pack["features"]
    return model, features

def predict_rain_prob(df: pd.DataFrame, model, features) -> pd.DataFrame:
    # Kiểm tra đủ cột features
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột feature trong forecast: {missing}")
    X = df[features]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["rain_prob"] = proba
    out["will_rain"] = (out["rain_prob"] >= RAIN_PROB_THRESHOLD).astype(int)
    return out

def classify_condition(row) -> str:
    """Trả về 'Mưa' | 'Nắng' | 'Mát mẻ' cho từng giờ."""
    if row.get("will_rain", 0) == 1:
        return "Mưa"
    # Không mưa: xét nắng / mát mẻ theo ngưỡng
    if (row.get("temperature", np.nan) >= HOT_TEMP_C) and (row.get("cloud", 100) < CLOUD_SUNNY_MAX):
        return "Nắng"
    return "Mát mẻ"

def plot_heatmaps(df_pred: pd.DataFrame, out_prefix: Path):
    """
    Vẽ 3 biểu đồ:
     - Heatmap nhiệt độ theo (Ngày x Giờ)
     - Heatmap độ ẩm theo (Ngày x Giờ)
     - Bản đồ điều kiện thời tiết (Mưa / Nắng / Mát mẻ) theo (Ngày x Giờ)
    """
    df = df_pred.copy()
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    # Pivot
    temp_pv = df.pivot(index="date", columns="hour", values="temperature")
    hum_pv  = df.pivot(index="date", columns="hour", values="humidity")

    # Vẽ heatmap Nhiệt độ
    fig1, ax1 = plt.subplots(figsize=(12, 3 + 0.6*len(temp_pv)))
    im1 = ax1.imshow(temp_pv.values, aspect="auto")
    ax1.set_title("Bản đồ nhiệt độ theo giờ")
    ax1.set_yticks(range(len(temp_pv.index)))
    ax1.set_yticklabels([str(d) for d in temp_pv.index])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([str(h) for h in range(0, 24, 2)])
    cbar1 = fig1.colorbar(im1, ax=ax1)
    cbar1.set_label("°C")
    ax1.set_xlabel("Giờ trong ngày")
    ax1.set_ylabel("Ngày")
    fig1.tight_layout()
    fig1.savefig(out_prefix.with_name(out_prefix.name + "_temp_heatmap.png"), dpi=150)

    # Vẽ heatmap Độ ẩm
    fig2, ax2 = plt.subplots(figsize=(12, 3 + 0.6*len(hum_pv)))
    im2 = ax2.imshow(hum_pv.values, aspect="auto")
    ax2.set_title("Bản đồ độ ẩm theo giờ")
    ax2.set_yticks(range(len(hum_pv.index)))
    ax2.set_yticklabels([str(d) for d in hum_pv.index])
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([str(h) for h in range(0, 24, 2)])
    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label("%")
    ax2.set_xlabel("Giờ trong ngày")
    ax2.set_ylabel("Ngày")
    fig2.tight_layout()
    fig2.savefig(out_prefix.with_name(out_prefix.name + "_humidity_heatmap.png"), dpi=150)

    # Lưới điều kiện (categorical)
    # Map điều kiện thành mã màu
    cond_map = {"Mưa": 0, "Nắng": 1, "Mát mẻ": 2}
    colors = np.array([
        [0.25, 0.5, 1.0],   # Mưa - xanh
        [1.0, 0.8, 0.2],    # Nắng - vàng
        [0.6, 0.9, 0.6],    # Mát mẻ - xanh nhạt
    ])
    df["condition_id"] = df["condition"].map(cond_map)
    cond_pv = df.pivot(index="date", columns="hour", values="condition_id")
    # Điền giá trị mặc định nếu NaN
    cond_pv = cond_pv.fillna(2)  # mặc định "Mát mẻ"
    cond_img = colors[cond_pv.values.astype(int)]

    fig3, ax3 = plt.subplots(figsize=(12, 3 + 0.6*len(cond_pv)))
    ax3.imshow(cond_img, aspect="auto")
    ax3.set_title("Tình trạng thời tiết theo giờ")
    ax3.set_yticks(range(len(cond_pv.index)))
    ax3.set_yticklabels([str(d) for d in cond_pv.index])
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([str(h) for h in range(0, 24, 2)])
    ax3.set_xlabel("Giờ trong ngày")
    ax3.set_ylabel("Ngày")

    # Legend thủ công
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=colors[0], edgecolor='k', label='Mưa'),
        Patch(facecolor=colors[1], edgecolor='k', label='Nắng'),
        Patch(facecolor=colors[2], edgecolor='k', label='Mát mẻ'),
    ]
    ax3.legend(handles=legend_patches, loc="upper right", frameon=True)
    fig3.tight_layout()
    fig3.savefig(out_prefix.with_name(out_prefix.name + "_condition_grid.png"), dpi=150)

    plt.close('all')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=10.8231, help="Vĩ độ")
    parser.add_argument("--lon", type=float, default=106.6297, help="Kinh độ")
    parser.add_argument("--hours", type=int, default=48, help="Số giờ dự báo (<=72)")
    args = parser.parse_args()

    # 1) Nạp model
    model, features = load_model()

    # 2) Lấy forecast
    df_fc = fetch_forecast_hourly(args.lat, args.lon, hours=args.hours)

    # 3) Dự đoán mưa (xác suất + nhị phân)
    df_pred = predict_rain_prob(df_fc, model, features)

    # 4) Gán nhãn điều kiện (Mưa / Nắng / Mát mẻ)
    df_pred["condition"] = df_pred.apply(classify_condition, axis=1)

    # 5) In bảng tóm tắt ra console (10 dòng đầu)
    show = df_pred[["time", "temperature", "humidity", "cloud",
                    "rain_prob", "will_rain", "condition"]].head(10)
    print("\n=== Dự đoán 10 giờ đầu ===")
    print(show.to_string(index=False, justify="left", col_space=10))

    # 6) Lưu CSV + Metadata
    out_csv = OUT_DIR / "hourly_prediction.csv"
    df_pred.to_csv(out_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")
    meta = {
        "lat": args.lat, "lon": args.lon, "hours": args.hours,
        "rain_prob_threshold": RAIN_PROB_THRESHOLD,
        "hot_temp_c": HOT_TEMP_C, "cloud_sunny_max": CLOUD_SUNNY_MAX,
        "features_used": features,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✔ Đã lưu dự đoán: {out_csv}")
    print(f"✔ Đã lưu cấu hình: {OUT_DIR/'config.json'}")

    # 7) Vẽ heatmap/bản đồ điều kiện
    plot_heatmaps(df_pred, OUT_DIR / "hourly")
    print(f"✔ Đã vẽ: {OUT_DIR/'hourly_temp_heatmap.png'}, "
          f"{OUT_DIR/'hourly_humidity_heatmap.png'}, "
          f"{OUT_DIR/'hourly_condition_grid.png'}")

if __name__ == "__main__":
    main()
