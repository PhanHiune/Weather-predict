# src/visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== tiện ích =====================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _format_time_axis(ax, tick_every_hours=3):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=tick_every_hours))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
    ax.grid(True, alpha=0.3)

# ===================== báo cáo (bảng) =====================

def build_hourly_report(df_pred: pd.DataFrame) -> pd.DataFrame:
    if not {"time","pred_label","temperature","humidity","prob_rain"}.issubset(df_pred.columns):
        raise ValueError("df_pred phải có các cột: time, pred_label, temperature, humidity, prob_rain")
    rep = pd.DataFrame({
        "time": df_pred["time"],
        "condition": df_pred["pred_label"],
        "temperature_C": df_pred["temperature"],
        "humidity_%": df_pred["humidity"],
        "rain_prob_%": (df_pred["prob_rain"] * 100.0).round(1)
    })
    return rep

def save_hourly_report_csv(df_pred: pd.DataFrame, out_csv_path: str) -> str:
    ensure_dir(os.path.dirname(out_csv_path))
    rep = build_hourly_report(df_pred)
    rep.to_csv(out_csv_path, index=False)
    return out_csv_path

# ===================== các plot đơn lẻ =====================

def plot_hourly_rain_prob(df_pred: pd.DataFrame, title_prefix="Xác suất mưa theo giờ (XGBoost)",
                          save_path: str | None = None, show: bool = False, threshold: float = 0.5):
    if not {"time","prob_rain"}.issubset(df_pred.columns):
        raise ValueError("Thiếu cột time/prob_rain trong df_pred")
    t = df_pred["time"]
    prob_pct = df_pred["prob_rain"] * 100.0
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_title(title_prefix)
    bars = ax.bar(t, prob_pct, width=0.03)
    ax.set_ylabel("Xác suất mưa (%)")
    ax.set_ylim(0, 100)
    _format_time_axis(ax, tick_every_hours=3)
    ax.axhline(threshold * 100, linestyle="--", linewidth=1)
    for b, p in zip(bars, prob_pct):
        b.set_alpha(0.8 if p >= threshold * 100 else 0.4)
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    return fig  # không close, để show cùng lúc

def plot_hourly_temperature(df_pred: pd.DataFrame, title="Nhiệt độ theo giờ",
                            save_path: str | None = None, show: bool = False):
    if not {"time","temperature"}.issubset(df_pred.columns):
        raise ValueError("Thiếu cột time/temperature")
    t = df_pred["time"]; y = df_pred["temperature"]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_ylabel("°C")
    ax.set_xlabel("Thời gian (VN)")
    _format_time_axis(ax, tick_every_hours=3)
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    return fig

def plot_hourly_humidity(df_pred: pd.DataFrame, title="Độ ẩm theo giờ",
                         save_path: str | None = None, show: bool = False):
    if not {"time","humidity"}.issubset(df_pred.columns):
        raise ValueError("Thiếu cột time/humidity")
    t = df_pred["time"]; y = df_pred["humidity"]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_ylabel("%")
    ax.set_xlabel("Thời gian (VN)")
    _format_time_axis(ax, tick_every_hours=3)
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    return fig

# ===================== condition bands + dashboard có headline =====================

def conclude_day(df_pred: pd.DataFrame, date_hint=None):
    if df_pred.empty:
        return "Kết luận: Không có dữ liệu", "unknown"
    counts = df_pred["pred_label"].value_counts(dropna=False)
    day_label = counts.index[0]
    conf = df_pred["pred_prob"].mean() if "pred_prob" in df_pred.columns else None
    if day_label == "rain":
        txt = "Kết luận: Hôm nay KHẢ NĂNG MƯA cao"
    elif day_label == "cloudy":
        txt = "Kết luận: Hôm nay NHIỀU MÂY"
    else:
        txt = "Kết luận: Hôm nay NẮNG"
    if conf is not None:
        txt += f" · độ tự tin trung bình {conf*100:.0f}%"
    return txt, day_label

def plot_hourly_dashboard_xgb_with_headline(
    df_pred: pd.DataFrame,
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
    title_prefix: str = "Dự báo theo giờ (XGBoost — sunny/cloudy/rain)",
    headline_text: str | None = None
):
    required = {"time","temperature","humidity","prob_rain","pred_label"}
    if not required.issubset(df_pred.columns):
        raise ValueError(f"df_pred phải có các cột: {sorted(required)}")
    t = df_pred["time"]
    prob_pct = df_pred["prob_rain"] * 100.0
    temp = df_pred["temperature"]
    hum  = df_pred["humidity"]
    if headline_text is None:
        headline_text, _ = conclude_day(df_pred)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    # (1) Rain prob
    ax1 = axes[0]
    ax1.set_title(f"{title_prefix}\n{headline_text}")
    bars = ax1.bar(t, prob_pct, width=0.03)
    ax1.set_ylabel("%"); ax1.set_ylim(0, 100)
    ax1.axhline(threshold * 100, linestyle="--", linewidth=1)
    _format_time_axis(ax1, tick_every_hours=3)
    for b, p in zip(bars, prob_pct):
        b.set_alpha(0.8 if p >= threshold * 100 else 0.4)
    # (2) Temp
    ax2 = axes[1]; ax2.plot(t, temp)
    ax2.set_title("Nhiệt độ"); ax2.set_ylabel("°C")
    _format_time_axis(ax2, tick_every_hours=3)
    # (3) Humidity
    ax3 = axes[2]; ax3.plot(t, hum)
    ax3.set_title("Độ ẩm"); ax3.set_ylabel("%"); ax3.set_xlabel("Thời gian (VN)")
    _format_time_axis(ax3, tick_every_hours=3)
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    return fig

def plot_hourly_condition_bands(df_pred: pd.DataFrame,
                                save_path: str | None = None, show: bool = False,
                                title: str = "Điều kiện theo giờ (sunny/cloudy/rain) + nhiệt độ & độ ẩm"):
    required = {"time","pred_label","temperature","humidity"}
    if not required.issubset(df_pred.columns):
        raise ValueError(f"df_pred phải có các cột: {sorted(required)}")
    t = df_pred["time"]
    temp = df_pred["temperature"]
    hum  = df_pred["humidity"]
    labels = df_pred["pred_label"].astype(str)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title(title)
    alpha_map = {"sunny": 0.15, "cloudy": 0.35, "rain": 0.6}
    alphas = [alpha_map.get(lbl, 0.35) for lbl in labels]
    ax1.bar(t, [1]*len(t), width=0.03, alpha=0.0)
    for ti, a in zip(t, alphas):
        ax1.bar(ti, 1, width=0.03, alpha=a, align="center", edgecolor="none")
    ax1.plot(t, temp, linewidth=2)
    ax1.set_ylabel("Nhiệt độ (°C)")
    _format_time_axis(ax1, tick_every_hours=3)
    ax2 = ax1.twinx()
    ax2.plot(t, hum, linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Độ ẩm (%)")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    return fig

# ===================== HIỆN 3 FIGURE CÙNG LÚC =====================

def plot_three_figures_once(df_pred: pd.DataFrame, out_dir: str, threshold: float = 0.5):
    """
    Tạo & LƯU & HIỂN THỊ cùng lúc 3 figure:
      1) Dashboard 3-ô (rain%, temp, humidity) + headline kết luận.
      2) Condition bands (nền theo nhãn) + temp & humidity.
      3) Rain probability bar (độc lập).
    Không set màu cụ thể, mỗi chart là một figure riêng.
    """
    ensure_dir(out_dir)
    headline_text, _ = conclude_day(df_pred)

    fig1 = plot_hourly_dashboard_xgb_with_headline(
        df_pred, threshold=threshold,
        save_path=os.path.join(out_dir, "fig_dashboard_headline.png"),
        show=False, headline_text=headline_text
    )
    fig2 = plot_hourly_condition_bands(
        df_pred,
        save_path=os.path.join(out_dir, "fig_condition_bands.png"),
        show=False
    )
    fig3 = plot_hourly_rain_prob(
        df_pred,
        save_path=os.path.join(out_dir, "fig_rain_prob.png"),
        show=False, threshold=threshold
    )

    # Hiển thị CÙNG LÚC cả 3 figure đã tạo
    plt.show()

    return {
        "dashboard_headline_png": os.path.join(out_dir, "fig_dashboard_headline.png"),
        "condition_bands_png": os.path.join(out_dir, "fig_condition_bands.png"),
        "rain_prob_png": os.path.join(out_dir, "fig_rain_prob.png"),
        "headline_text": headline_text
    }
