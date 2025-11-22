# src/visualize.py
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_hourly(df_hourly, out_dir: str, target_date, predicted_label: str, rain_prob_percent: float):
    ensure_dir(out_dir)
    t = df_hourly["time"]

    header = f"Ngày {target_date.isoformat()} – Dự đoán: {predicted_label} – Xác suất mưa: {rain_prob_percent:.1f}%"

    def save_plot(y, title, filename, ylabel=None):
        plt.figure()
        plt.plot(t, y)  # không set style/color để theo mặc định
        plt.title(f"{title}\n{header}")
        if ylabel:
            plt.ylabel(ylabel)
        plt.xlabel("Thời gian (VN)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=130)
        plt.close()
        return out_path

    paths = []
    paths.append(save_plot(df_hourly["temperature"], "Nhiệt độ theo giờ", f"temperature_{target_date}.png", "°C"))
    paths.append(save_plot(df_hourly["humidity"],    "Độ ẩm theo giờ",    f"humidity_{target_date}.png", "%"))
    paths.append(save_plot(df_hourly["cloud"],       "Mây che phủ theo giờ", f"cloud_{target_date}.png", "%"))
    paths.append(save_plot(df_hourly["pressure"],    "Áp suất theo giờ",  f"pressure_{target_date}.png", "hPa"))
    paths.append(save_plot(df_hourly["wind"],        "Tốc độ gió theo giờ", f"wind_{target_date}.png", "m/s"))
    paths.append(save_plot(df_hourly["precip"],      "Lượng mưa theo giờ", f"precip_{target_date}.png", "mm"))

    return paths

def plot_hourly_show(df_hourly, target_date, predicted_label: str, rain_prob_percent: float):
    t = df_hourly["time"]
    header = f"Ngày {target_date.isoformat()} – Dự đoán: {predicted_label} – Xác suất mưa: {rain_prob_percent:.1f}%"

    def make_plot(y, title, ylabel=None):
        fig, ax = plt.subplots()
        ax.plot(t, y)  # không set màu/style theo yêu cầu
        ax.set_title(f"{title}\n{header}")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel("Thời gian (VN)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()

    make_plot(df_hourly["temperature"], "Nhiệt độ theo giờ", "°C")
    make_plot(df_hourly["humidity"],    "Độ ẩm theo giờ", "%")
    make_plot(df_hourly["cloud"],       "Mây che phủ theo giờ", "%")
    make_plot(df_hourly["pressure"],    "Áp suất theo giờ", "hPa")
    make_plot(df_hourly["wind"],        "Tốc độ gió theo giờ", "m/s")
    make_plot(df_hourly["precip"],      "Lượng mưa theo giờ", "mm")

    # Hiển thị tất cả figure đã tạo
    plt.show()
def plot_hourly_dashboard(hourly_df, target_date, predicted_label: str, rain_prob_percent: float,
                          save_path: str | None = None, show: bool = True):
    t = hourly_df["time"]
    header = f"Ngày {target_date.isoformat()} – Dự đoán: {predicted_label} – Xác suất mưa: {rain_prob_percent:.1f}%"

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    series = [
        ("Nhiệt độ (°C)", hourly_df["temperature"], "°C"),
        ("Độ ẩm (%)",     hourly_df["humidity"],    "%"),
        ("Mây che phủ (%)", hourly_df["cloud"],     "%"),
        ("Áp suất (hPa)", hourly_df["pressure"],    "hPa"),
        ("Tốc độ gió (m/s)", hourly_df["wind"],     "m/s"),
        ("Lượng mưa (mm)",  hourly_df["precip"],    "mm"),
    ]

    for ax, (title, y, ylabel) in zip(axes, series):
        ax.plot(t, y)  # để màu/style mặc định
        ax.set_title(title)
        ax.set_xlabel("Thời gian (VN)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    # Tiêu đề lớn cho toàn figure
    fig.suptitle(header, fontsize=12, y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130)

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_hourly_dashboard_with_probs(hourly_df, target_date, predicted_label: str, rain_prob_percent: float,
                                     three_day_probs: list,  # list[(date, rain_prob_percent_float)]
                                     save_path: str | None = None, show: bool = True):
    t = hourly_df["time"]
    header = f"Ngày hiển thị: {target_date.isoformat()} – Dự đoán: {predicted_label} – Xác suất mưa: {rain_prob_percent:.1f}%"

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)  # 4 hàng x 2 cột

    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0]); ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, :])  # hàng cuối: full width cho bar chart

    def line(ax, y, title, ylabel):
        ax.plot(t, y)
        ax.set_title(title)
        ax.set_xlabel("Thời gian (VN)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)

    line(ax1, hourly_df["temperature"], "Nhiệt độ", "°C")
    line(ax2, hourly_df["humidity"],    "Độ ẩm",    "%")
    line(ax3, hourly_df["cloud"],       "Mây che phủ", "%")
    line(ax4, hourly_df["pressure"],    "Áp suất",  "hPa")
    line(ax5, hourly_df["wind"],        "Tốc độ gió", "m/s")
    line(ax6, hourly_df["precip"],      "Lượng mưa", "mm")

    # Bar chart 3 ngày
    labels = [d.strftime("%d/%m") for d, _ in three_day_probs]
    vals   = [p for _, p in three_day_probs]
    bars = ax7.bar(labels, vals)
    ax7.set_title("Xác suất mưa (%) – Hôm nay / Ngày mai / Ngày kia")
    ax7.set_ylabel("%")
    ax7.set_ylim(0, 100)
    ax7.grid(True, axis="y", alpha=0.3)

    # Highlight cột của ngày đang hiển thị
    for i, (d, v) in enumerate(three_day_probs):
        if d == target_date:
            bars[i].set_alpha(0.7)
            # Không set màu cụ thể theo yêu cầu—chỉ đổi alpha
        ax7.text(i, v + 2, f"{v:.0f}%", ha="center", va="bottom")

    fig.suptitle(header, fontsize=12, y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130)

    if show:
        plt.show()
    else:
        plt.close(fig)
def plot_hourly_rain_dashboard(df_hourly_probs, threshold: float = 0.5,
                               title_prefix: str = "Xác suất mưa theo giờ (48h) -- Đà Nẵng",
                               save_path: str | None = None, show: bool = True):
    t = df_hourly_probs["time"]
    prob_pct = (df_hourly_probs["rain_prob"] * 100.0)
    precip = df_hourly_probs["precip_forecast"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)

    # Bar xác suất mưa
    bars = ax1.bar(t, prob_pct, width=0.03)  # width nhỏ để giống cột giờ
    ax1.set_title(f"{title_prefix}")
    ax1.set_ylabel("Xác suất mưa (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.axhline(threshold*100, linestyle="--", linewidth=1)  # vạch ngưỡng

    # Highlight bars > threshold
    for b, p in zip(bars, prob_pct):
        if p >= threshold * 100:
            b.set_alpha(0.8)  # không set màu cụ thể, chỉ đổi alpha
        else:
            b.set_alpha(0.4)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(45)

    # Line lượng mưa dự báo
    ax2.plot(t, precip)
    ax2.set_title("Lượng mưa dự báo (mm)")
    ax2.set_ylabel("mm")
    ax2.set_xlabel("Thời gian (VN)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(45)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130)

    if show:
        plt.show()
    else:
        plt.close(fig)