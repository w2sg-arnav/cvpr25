# phase4_finetuning/models/efficientnet.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights  # Updated to B7
from config import NUM_CLASSES

class EfficientNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        
        # Add dropout before the final fully connected layer
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout with 50% probability
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, rgb):
        return self.base_model(rgb)