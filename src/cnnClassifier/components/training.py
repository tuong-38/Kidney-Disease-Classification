import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        # Khởi tạo lại cấu trúc mô hình (VGG16)
        import torchvision.models as models
        self.model = models.vgg16()
        
        # Sửa lại lớp cuối tương tự như Stage 02 để load được file weights
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 4) # 4 là số classes
        
        # Load trọng số đã lưu từ Stage 02
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path))
        self.model.to(self.device)

    def train_dataloader(self):
        # Chuẩn bị Data Augmentation và Normalization
        transform_train = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.RandomHorizontalFlip() if self.config.params_is_augmentation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform_train
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True
        )

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01) # Bạn có thể chỉnh LR trong params.yaml

        self.model.train()
        
        print(f"Bắt đầu huấn luyện trên thiết bị: {self.device}")
        
        for epoch in range(self.config.params_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.config.params_epochs}, Loss: {running_loss/len(self.train_loader):.4f}")

        # Lưu mô hình sau khi huấn luyện xong
        torch.save(self.model.state_dict(), self.config.trained_model_path)