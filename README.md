cài thư viện:
->pip install -r requirements.txt
chạy chương trình tải dữ liệu:
-> python src/download_data.py
chạy chương trình huấn luyện mô hình:
-> python src/train.py
chạy dự đoán:
->python main.py
*Lưu ý: nếu như chương trình bị lỗi, thì khả năng cao là do chưa xoá model đã train, vậy nên hãy chạy câu lệnh:
-> del models/xgboost_model.pkl
nếu như chạy lần đầu, cần dùng lệnh:
->python main_hourly.py --start 2023-01-01 --chunk 30
để có thay đổi khoảng chạy, có thể thay đổi năm

cách tạo venv
-> python -m venv venv
