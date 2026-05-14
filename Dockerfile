FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

# Cài đặt các thư viện cần thiết
RUN pip install -r requirements.txt

# Chạy ứng dụng (Giả sử bạn sẽ tạo file app.py cho giao diện Web)
CMD ["python3", "app.py"]

