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

