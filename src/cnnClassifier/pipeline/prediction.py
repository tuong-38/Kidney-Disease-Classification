import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torchvision.models as models

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.device = torch.device("cpu") # Chạy bằng CPU cho nhẹ

    def predict(self):
        # 1. Load mô hình
        model = models.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 4)
        model.load_state_dict(torch.load(os.path.join("artifacts", "training", "model.pth")))
        model.eval()

        # 2. Xử lý ảnh đầu vào
        imagename = self.filename
        test_image = Image.open(imagename).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(test_image).unsqueeze(0)

        # 3. Dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            result = predicted.item()

        # 4. Giải mã kết quả
        classes = ['Adenocarcinoma', 'Cyst', 'Normal', 'Tumor']
        prediction = classes[result]
        return [{"image": prediction}]
    