# phase4_finetuning/models/baseline.py
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from config import NUM_CLASSES

class InceptionV3Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = inception_v3(pretrained=True)
        self.base_model.aux_logits = False  # Disable auxiliary logits for simplicity

        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, rgb):
        return self.base_model(rgb)