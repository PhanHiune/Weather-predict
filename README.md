cài thư viện:
->pip install -r requirements.txt
chạy chương trình tải dữ liệu:
-> python src/download_data.py
chạy chương trình huấn luyện mô hình:
-> python -m src.relabel_from_rain
python -m src.train_weather_label

chạy dự đoán và vẽ hình:
python -m src.main_hourly --lat 16.0471 --lon 108.2068 --hours 48

*Lưu ý: nếu như chương trình bị lỗi, thì khả năng cao là do chưa xoá model đã train, vậy nên hãy chạy câu lệnh:
-> del models/xgboost_model.pkl
nếu như chạy lần đầu, cần dùng lệnh:
->python main_hourly.py --start 2023-01-01 --chunk 30
để có thay đổi khoảng chạy, có thể thay đổi năm

cách tạo venv
-> python -m venv venv
