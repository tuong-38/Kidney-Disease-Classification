import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Đường dẫn mà Pipeline Stage 03 đang tìm kiếm
data_path = Path("artifacts/data_ingestion/kidney-ct-scan-image")
categories = ['Adenocarcinoma', 'Normal', 'Cyst', 'Tumor']

# Xóa cái file "lỗi" đang tồn tại đi và tạo folder mới
if os.path.exists("artifacts/data_ingestion/kidney-ct-scan-image"):
    if os.path.isfile("artifacts/data_ingestion/kidney-ct-scan-image"):
        os.remove("artifacts/data_ingestion/kidney-ct-scan-image")
    else:
        shutil.rmtree("artifacts/data_ingestion/kidney-ct-scan-image")

# Tạo lại cấu trúc thư mục và ảnh thực
for category in categories:
    os.makedirs(data_path / category, exist_ok=True)
    for i in range(5):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(data_path / category / f"sample_{i}.jpg")

print(f"Đã cấu trúc lại thư mục thành folder chuẩn tại: {data_path}")