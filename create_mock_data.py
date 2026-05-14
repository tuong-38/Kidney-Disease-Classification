import os
from pathlib import Path

# Định nghĩa cấu trúc
data_path = Path("artifacts/data_ingestion/kidney-ct-scan-image")
categories = ['Adenocarcinoma', 'Normal', 'Cyst', 'Tumor']

# Tạo thư mục và file ảnh giả (Dummy images)
for category in categories:
    os.makedirs(data_path / category, exist_ok=True)
    for i in range(5): # Tạo mỗi loại 5 ảnh để test
        with open(data_path / category / f"image_{i}.jpg", "w") as f:
            f.write("dummy data")

print(f"Đã tạo bộ dữ liệu giả lập tại: {data_path}")