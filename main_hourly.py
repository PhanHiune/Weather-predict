# src/main_hourly.py
# ------------------------------------------------------------
# MAIN: dự đoán mưa / thời tiết theo giờ bằng model đã train + vẽ heatmap
# ------------------------------------------------------------
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
# Nếu bạn train theo weather_label (0=nắng,1=mưa,2=mát mẻ)
# thì đường dẫn nên là:
MODEL_PATH = Path("models/xgb_weather_label.pkl")

# Nếu bạn vẫn đang dùng model nhị phân (rain_hour 0/1) thì
# sửa lại thành:
# MODEL_PATH = Path("models/xgb_hourly_rain.pkl")

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ngưỡng phân loại mưa (cho model nhị phân)
RAIN_PROB_THRESHOLD = 0.5

# Quy tắc phân loại "nắng / mát mẻ" (chỉ dùng fallback khi không có weather_label)
HOT_TEMP_C = 32.0
CLOUD_SUNNY_MAX = 30  # %

# Các feature mà model đã sử dụng để train
FEATURES_DEFAULT = [
    "temperature",
    "humidity",
    "cloud",
    "pressure",
    "wind",
    "hour_of_day",
]

# ===================== UTILS =====================
def fetch_forecast_hourly(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    """Lấy dự báo giờ từ Open-Meteo trong ~48-72h tới."""
    import requests, numpy as np, pandas as pd

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "cloudcover",
                "pressure_msl",
                "windspeed_10m",  # theo schema bạn đang dùng
                "weathercode",
                "precipitation",
            ]
        ),
        "forecast_days": 3,  # ~72h, ta sẽ cắt còn 'hours' phía dưới
        "timezone": "auto",
    }

    r = requests.get(url, params=params, timeout=60)
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
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(h["time"]),
            "temperature": h["temperature_2m"],
            "humidity": h["relative_humidity_2m"],
            "cloud": h["cloudcover"],
            "pressure": h["pressure_msl"],
            "wind": h["windspeed_10m"],
            "weathercode": h.get("weathercode", [np.nan] * len(h)),
            "precipitation": h.get("precipitation", [np.nan] * len(h)),
        }
    )
    df["hour_of_day"] = df["time"].dt.hour
    df = df.sort_values("time").head(hours).reset_index(drop=True)
    return df


def load_model():
    """
    Hỗ trợ cả 2 kiểu lưu:
        - joblib.dump(model, path)
        - joblib.dump({"model": model, "features": [...]}, path)
    """
    pack = joblib.load(MODEL_PATH)
    if isinstance(pack, dict):
        model = pack["model"]
        features = pack.get("features", FEATURES_DEFAULT)
    else:
        model = pack
        features = FEATURES_DEFAULT
    return model, features


def predict_rain_prob(df: pd.DataFrame, model, features) -> pd.DataFrame:
    """
    - Nếu model là nhị phân (2 lớp):
        + rain_prob = P(class=1)
        + will_rain = rain_prob >= threshold
        + weather_label được gán bằng rule: 1=mưa, 2=mát mẻ, 0=nắng
    - Nếu model là multi-class (3 lớp: 0=nắng,1=mưa,2=mát mẻ):
        + dùng argmax để lấy weather_label
        + rain_prob = P(class=1)
        + will_rain = (weather_label==1)
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột feature trong forecast: {missing}")

    X = df[features]
    proba = model.predict_proba(X)
    out = df.copy()

    n_classes = proba.shape[1]

    if n_classes == 2:
        # ===== MODEL NHỊ PHÂN: 0 = không mưa, 1 = mưa =====
        rain_prob = proba[:, 1]
        out["rain_prob"] = rain_prob
        out["will_rain"] = (out["rain_prob"] >= RAIN_PROB_THRESHOLD).astype(int)

        # Gán weather_label từ rule:
        # will_rain == 1 -> 1 (Mưa)
        # will_rain == 0 & temp < 27 -> 2 (Mát mẻ)
        # will_rain == 0 & temp >= 27 -> 0 (Nắng)
        def _rule_weather(row):
            if row["will_rain"] == 1:
                return 1
            elif row["temperature"] < 27:
                return 2
            else:
                return 0

        out["weather_label"] = out.apply(_rule_weather, axis=1)

    else:
        # ===== MODEL 3 LỚP: 0 = nắng, 1 = mưa, 2 = mát mẻ =====
        rain_prob = proba[:, 1]
        labels = proba.argmax(axis=1).astype(int)

        out["rain_prob"] = rain_prob
        out["weather_label"] = labels
        out["will_rain"] = (labels == 1).astype(int)

    return out


def classify_condition(row) -> str:
    lbl = row.get("weather_label", None)
    if lbl == 1:
        return "Mưa"
    elif lbl == 2:
        return "Mát mẻ"
    elif lbl == 0:
        return "Nắng"

    # Fallback: dùng logic cũ nếu không có weather_label
    if row.get("will_rain", 0) == 1:
        return "Mưa"
    if (row.get("temperature", np.nan) >= HOT_TEMP_C) and (
        row.get("cloud", 100) < CLOUD_SUNNY_MAX
    ):
        return "Nắng"
    return "Mát mẻ"


def plot_heatmaps(df_pred: pd.DataFrame, out_prefix: Path):
    df = df_pred.copy()
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour

    temp_pv = df.pivot(index="date", columns="hour", values="temperature")
    hum_pv = df.pivot(index="date", columns="hour", values="humidity")

    # Heatmap Nhiệt độ
    fig1, ax1 = plt.subplots(figsize=(12, 3 + 0.6 * len(temp_pv)))
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

    # Heatmap Độ ẩm
    fig2, ax2 = plt.subplots(figsize=(12, 3 + 0.6 * len(hum_pv)))
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
    fig2.savefig(
        out_prefix.with_name(out_prefix.name + "_humidity_heatmap.png"), dpi=150
    )

    # Bản đồ điều kiện thời tiết (categorical)
    cond_map = {"Mưa": 0, "Nắng": 1, "Mát mẻ": 2}
    colors = np.array(
        [
            [0.25, 0.5, 1.0],  # Mưa - xanh
            [1.0, 0.8, 0.2],   # Nắng - vàng
            [0.6, 0.9, 0.6],   # Mát mẻ - xanh nhạt
        ]
    )
    df["condition_id"] = df["condition"].map(cond_map)
    cond_pv = df.pivot(index="date", columns="hour", values="condition_id")
    cond_pv = cond_pv.fillna(2)  # mặc định "Mát mẻ"
    cond_img = colors[cond_pv.values.astype(int)]

    fig3, ax3 = plt.subplots(figsize=(12, 3 + 0.6 * len(cond_pv)))
    ax3.imshow(cond_img, aspect="auto")
    ax3.set_title("Tình trạng thời tiết theo giờ")
    ax3.set_yticks(range(len(cond_pv.index)))
    ax3.set_yticklabels([str(d) for d in cond_pv.index])
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([str(h) for h in range(0, 24, 2)])
    ax3.set_xlabel("Giờ trong ngày")
    ax3.set_ylabel("Ngày")

    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=colors[0], edgecolor="k", label="Mưa"),
        Patch(facecolor=colors[1], edgecolor="k", label="Nắng"),
        Patch(facecolor=colors[2], edgecolor="k", label="Mát mẻ"),
    ]
    ax3.legend(handles=legend_patches, loc="upper right", frameon=True)
    fig3.tight_layout()
    fig3.savefig(
        out_prefix.with_name(out_prefix.name + "_condition_grid.png"), dpi=150
    )

    plt.close("all")


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

    # 3) Dự đoán (mưa / thời tiết) + xác suất
    df_pred = predict_rain_prob(df_fc, model, features)

    # 4) Gán nhãn điều kiện (Mưa / Nắng / Mát mẻ) từ weather_label/will_rain
    df_pred["condition"] = df_pred.apply(classify_condition, axis=1)

    # 5) In bảng tóm tắt ra console (10 dòng đầu)
    show = df_pred[
        [
            "time",
            "temperature",
            "humidity",
            "cloud",
            "rain_prob",
            "will_rain",
            "weather_label",
            "condition",
        ]
    ].head(10)
    print("\n=== Dự đoán 10 giờ đầu ===")
    print(show.to_string(index=False, justify="left", col_space=10))

    # 6) Lưu CSV + Metadata
    out_csv = OUT_DIR / "hourly_prediction.csv"
    df_pred.to_csv(out_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")
    meta = {
        "lat": args.lat,
        "lon": args.lon,
        "hours": args.hours,
        "rain_prob_threshold": RAIN_PROB_THRESHOLD,
        "hot_temp_c": HOT_TEMP_C,
        "cloud_sunny_max": CLOUD_SUNNY_MAX,
        "features_used": features,
        "model_path": str(MODEL_PATH),
    }
    (OUT_DIR / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✔ Đã lưu dự đoán: {out_csv}")
    print(f"✔ Đã lưu cấu hình: {OUT_DIR/'config.json'}")

    # 7) Vẽ heatmap/bản đồ điều kiện
    plot_heatmaps(df_pred, OUT_DIR / "hourly")
    print(
        f"✔ Đã vẽ: {OUT_DIR/'hourly_temp_heatmap.png'}, "
        f"{OUT_DIR/'hourly_humidity_heatmap.png'}, "
        f"{OUT_DIR/'hourly_condition_grid.png'}"
    )


if __name__ == "__main__":
    main()
