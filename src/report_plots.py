# src/report_plots.py
"""
Sinh các biểu đồ phục vụ báo cáo:
- Trước / sau xử lý dữ liệu
- Sau khi xây dựng & huấn luyện mô hình (Data Visualization after Model Building)

Chạy:
    python -m src.report_plots
hoặc:
    python src/report_plots.py
(từ thư mục gốc của project)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)

from xgboost import XGBClassifier

# ==== đường dẫn & cấu hình chung ====
DATA_HOURLY_LABELED = Path("data/historical_hourly_labeled.csv")
DATA_WEATHER_RAW = Path("data/historical_weather.csv")
OUT_DIR = Path("output/report_figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# các cột đặc trưng + nhãn (tuỳ project của bạn mà sửa lại)
FEATURES = ["temperature", "humidity", "cloud", "pressure", "wind", "hour_of_day"]
TARGET = "rain_hour"  # 0 = không mưa, 1 = có mưa


# ---------------------------------------------------------------------
# 0. HÀM TIỆN ÍCH
# ---------------------------------------------------------------------
def _load_hourly_labeled_basic():
    """Dùng riêng cho hình 1 (trees vs accuracy)."""
    df = pd.read_csv(DATA_HOURLY_LABELED, parse_dates=["time"])
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_evaluate_model():
    """
    Train 1 model XGBoost & trả về mọi thứ cần cho các hình 'sau khi có model'.

    Trả về:
        df      : full dataframe (có cột time)
        X_train, X_test, y_train, y_test
        model   : XGBClassifier đã train
        y_pred  : nhãn dự đoán (0/1) trên test
        y_proba : xác suất dự đoán lớp 1 trên test
    """
    df = pd.read_csv(DATA_HOURLY_LABELED, parse_dates=["time"])

    X = df[FEATURES]
    y = df[TARGET]

    # Không shuffle để giữ thứ tự thời gian cho biểu đồ theo time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return df, X_train, X_test, y_train, y_test, model, y_pred, y_proba


# ---------------------------------------------------------------------
# 1. ẢNH TRƯỚC / SAU XỬ LÝ DỮ LIỆU
# ---------------------------------------------------------------------
def plot_trees_vs_accuracy():
    """
    Ảnh 1: số lượng cây vs độ chính xác (train / test) – minh hoạ:
    'càng nhiều cây thì càng có độ chính xác cao' (đến một mức nào đó).
    """
    Xtr, Xte, ytr, yte = _load_hourly_labeled_basic()

    n_trees_list = [50, 100, 200, 400, 800]
    train_acc = []
    test_acc = []

    for n in n_trees_list:
        clf = XGBClassifier(
            n_estimators=n,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(Xtr, ytr)
        train_acc.append(accuracy_score(ytr, clf.predict(Xtr)))
        test_acc.append(accuracy_score(yte, clf.predict(Xte)))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(n_trees_list))

    ax.bar(x - width / 2, train_acc, width, label="Train accuracy")
    ax.bar(x + width / 2, test_acc, width, label="Test accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_trees_list])
    ax.set_xlabel("Số lượng cây (n_estimators)")
    ax.set_ylabel("Độ chính xác")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Ảnh 1 – Ảnh 2 cột: Ảnh hưởng số lượng cây tới độ chính xác")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out_path = OUT_DIR / "fig1_trees_vs_accuracy.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


def plot_gpu_time_tft_vs_xgb():
    """
    Ảnh 2: so sánh thời gian train & GPU giữa TFT và XGBoost.

    Ở đây dùng số liệu minh hoạ. Khi có số liệu thực, sửa 4 con số time_minutes, gpu_gb.
    """
    models = ["XGBoost", "TFT"]

    # TODO: thay bằng số liệu thực
    time_minutes = [5, 60]  # thời gian train (phút)
    gpu_gb = [1.0, 8.0]     # bộ nhớ GPU max (GB)

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Trục trái: thời gian train
    ax1.bar(x - width / 2, time_minutes, width, label="Thời gian train (phút)")
    ax1.set_ylabel("Thời gian train (phút)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title("Ảnh 2 – TFT train lâu và tốn nhiều GPU hơn XGBoost")
    ax1.grid(True, axis="y", alpha=0.3)

    # Trục phải: bộ nhớ GPU
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, gpu_gb, width, label="Bộ nhớ GPU (GB)", alpha=0.6)
    ax2.set_ylabel("Bộ nhớ GPU tối đa (GB)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    out_path = OUT_DIR / "fig2_tft_gpu_vs_xgb.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


def plot_accuracy_two_models():
    """
    Ảnh 3: so sánh độ chính xác 2 mô hình (XGBoost vs TFT).
    Bạn thay accuracy bằng số liệu thực tế.
    """
    model_names = ["XGBoost", "TFT"]

    # TODO: thay bằng metric thực (accuracy/AUC/F1...) trên tập test
    accuracy = [0.87, 0.90]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(model_names))
    ax.bar(x, accuracy)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Độ chính xác")
    ax.set_title("Ảnh 3 – So sánh độ chính xác giữa hai mô hình")

    for i, v in enumerate(accuracy):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")

    ax.grid(True, axis="y", alpha=0.3)

    out_path = OUT_DIR / "fig3_model_accuracy_compare.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


def plot_before_after_preprocess():
    """
    Ảnh 4: trực quan hoá dữ liệu trước và sau xử lý.

    - Trước: phân bố lượng mưa (mm) trong dữ liệu thô từ API.
    - Sau: phân bố nhãn rain_hour (0/1) sau khi chuyển sang bài toán classification.
    """
    raw = pd.read_csv(DATA_WEATHER_RAW)
    processed = pd.read_csv(DATA_HOURLY_LABELED)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Trước xử lý: histogram lượng mưa mm
    axes[0].hist(raw["rain"], bins=50)
    axes[0].set_title("Trước xử lý – Phân bố lượng mưa (mm)")
    axes[0].set_xlabel("Lượng mưa (mm)")
    axes[0].set_ylabel("Số giờ")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Sau xử lý: số mẫu không mưa / có mưa
    counts = processed["rain_hour"].value_counts().sort_index()
    labels = ["Không mưa (0)", "Có mưa (1)"]
    axes[1].bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    axes[1].set_title("Sau xử lý – Phân bố nhãn rain_hour")
    axes[1].set_ylabel("Số giờ")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Ảnh 4 – Trực quan hóa dữ liệu trước và sau khi xử lý")
    out_path = OUT_DIR / "fig4_before_after_preprocess.png"
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


# ---------------------------------------------------------------------
# 2. CÁC HÌNH “SAU KHI CÓ MODEL” – DATA VISUALIZATION AFTER MODEL BUILDING
# ---------------------------------------------------------------------
def plot_pred_vs_true_over_time(df, X_test, y_test, y_pred, y_proba):
    """
    Hình 5: Hiển thị kết quả dự đoán vs giá trị thực theo thời gian.
    (line/scatter plot).

    Giải thích:
        - Giúp xem mô hình dự đoán đúng – sai ở từng thời điểm.
        - Phù hợp với dữ liệu chuỗi thời gian (weather).
    """
    idx_test = X_test.index
    time_test = df.loc[idx_test, "time"]

    # Lấy tối đa N điểm đầu cho đỡ rối
    N = min(300, len(time_test))
    time_plot = time_test.iloc[:N]
    y_true_plot = y_test.iloc[:N]
    y_pred_plot = y_pred[:N]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(time_plot, y_true_plot, label="Thực tế (rain_hour)", marker="o")
    ax.scatter(time_plot, y_pred_plot, label="Dự đoán (label)", marker="x", alpha=0.7)

    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Nhãn mưa (0 = không mưa, 1 = có mưa)")
    ax.set_title("Ảnh 5 – So sánh giá trị thực và dự đoán theo thời gian")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()

    out_path = OUT_DIR / "fig5_pred_vs_true_over_time.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


def plot_confusion_matrix_roc_pr(y_test, y_pred, y_proba):
    """
    Hình 6 & 7: Trực quan hoá
        - Confusion matrix
        - ROC curve + Precision–Recall curve

    Giải thích:
        - Confusion matrix: xem mô hình hay nhầm lẫn kiểu nào.
        - ROC/PR: đánh giá độ phân biệt giữa 2 lớp, đặc biệt khi dữ liệu mất cân bằng.
    """
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Không mưa", "Có mưa"]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Ảnh 6 – Confusion Matrix mô hình XGBoost")
    out_path = OUT_DIR / "fig6_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")

    # --- ROC & Precision–Recall ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Ảnh 7a – ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision–Recall
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Ảnh 7b – Precision–Recall Curve")
    axes[1].grid(True, alpha=0.3)

    out_path = OUT_DIR / "fig7_roc_pr_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


def plot_feature_importance(model):
    """
    Hình 8: Feature Importance (bar chart).

    Giải thích:
        - Cho biết đặc trưng nào đóng góp nhiều cho quyết định của mô hình.
        - Rất phù hợp với các mô hình cây quyết định / XGBoost / Random Forest.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_sorted = [FEATURES[i] for i in indices]
    importances_sorted = importances[indices]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features_sorted[::-1], importances_sorted[::-1])
    ax.set_xlabel("Độ quan trọng (feature importance)")
    ax.set_title("Ảnh 8 – Tầm quan trọng của các đặc trưng (XGBoost)")
    ax.grid(True, axis="x", alpha=0.3)

    out_path = OUT_DIR / "fig8_feature_importance.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


def plot_error_distribution(y_test, y_proba):
    """
    Hình 9: Phân phối lỗi (residuals) – histogram + boxplot.

    Ở bài toán classification:
        residual = y_true - P(y=1)
    Giúp:
        - Xem mô hình đang bias về phía lớp nào.
        - Nhận biết outlier / mẫu dự đoán rất tệ.
    """
    residuals = y_test.values - y_proba

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram
    axes[0].hist(residuals, bins=40)
    axes[0].set_title("Ảnh 9a – Histogram của residuals")
    axes[0].set_xlabel("Residual = y_true - P(y=1)")
    axes[0].set_ylabel("Số mẫu")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Boxplot
    axes[1].boxplot(residuals, vert=True)
    axes[1].set_title("Ảnh 9b – Boxplot của residuals")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, axis="y", alpha=0.3)

    out_path = OUT_DIR / "fig9_error_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✔ Đã lưu {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # 1. Các hình về số lượng cây, GPU, accuracy, trước/sau xử lý dữ liệu
    plot_trees_vs_accuracy()
    plot_gpu_time_tft_vs_xgb()
    plot_accuracy_two_models()
    plot_before_after_preprocess()

    # 2. Train model 1 lần cho tất cả hình "sau khi có model"
    (
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        model,
        y_pred,
        y_proba,
    ) = train_and_evaluate_model()

    plot_pred_vs_true_over_time(df, X_test, y_test, y_pred, y_proba)
    plot_confusion_matrix_roc_pr(y_test, y_pred, y_proba)
    plot_feature_importance(model)
    plot_error_distribution(y_test, y_proba)


if __name__ == "__main__":
    main()
