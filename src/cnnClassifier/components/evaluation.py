import torch
from pathlib import Path
from cnnClassifier.utils.common import save_json
from cnnClassifier.entity.config_entity import EvaluationConfig
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        self.model = models.vgg16()
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 4)
        self.model.load_state_dict(torch.load(self.config.path_of_model))
        self.model.to(self.device)
        self.model.eval()

    def evaluation(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.ImageFolder(root=self.config.training_data, transform=transform)
        loader = DataLoader(dataset, batch_size=self.config.params_batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.score = {"loss": 0, "accuracy": correct / total} # Tạm thời lấy accuracy
        save_json(path=Path("scores.json"), data=self.score)