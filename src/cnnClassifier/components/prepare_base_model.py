import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        # Tải mô hình VGG16 với trọng số ImageNet
        if self.config.params_weights == "imagenet":
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            self.model = models.vgg16()
            
        self.model.to(self.device)
        self.save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        # Đóng băng toàn bộ các lớp (Freeze All)
        for param in self.model.parameters():
            param.requires_grad = False

        # Thay thế lớp Classifier cuối cùng (Include Top = False logic trong PyTorch)
        # VGG16 của torchvision có phần 'classifier' là một Sequential
        # Chúng ta thay đổi lớp Linear cuối cùng để khớp với số lượng classes của bạn (4 lớp)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.config.params_classes)
        
        # Đưa lớp mới này vào thiết bị (GPU) và bật tính năng học tập cho nó
        self.model.classifier[6].requires_grad = True
        self.model.to(self.device)

        print(self.model) # Xem cấu trúc mô hình
        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)
    