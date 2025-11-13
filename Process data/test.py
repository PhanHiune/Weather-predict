import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === 1. Đọc dữ liệu từ file Excel ===
# Thay đường dẫn bằng file của bạn
file_path = "weather_prediction_bbq_labels.csv"
df = pd.read_csv(file_path)

print("Dữ liệu đọc được:")
print(df.head())
# đã có dữ liệu từ file csv
