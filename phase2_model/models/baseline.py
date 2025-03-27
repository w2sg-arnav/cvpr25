# models/baseline.py
import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, SPECTRAL_CHANNELS

class InceptionV3Baseline(nn.Module):
    """Inception V3 baseline for comparison, adapted for multi-modal input."""
    
    def __init__(self, num_classes: int = NUM_CLASSES, spectral_channels: int = SPECTRAL_CHANNELS):
        super().__init__()
        # Load pretrained Inception V3
        self.inception = models.inception_v3(pretrained=True)
        
        # Modify the first layer to accept 3 + spectral_channels
        self.inception.Conv2d_1a_3x3 = nn.Conv2d(3 + spectral_channels, 32, kernel_size=3, stride=2, bias=False)
        
        # Modify the final layer for the number of classes
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb (torch.Tensor): RGB input [batch, 3, H, W].
            spectral (torch.Tensor): Spectral input [batch, spectral_channels, H, W].
        
        Returns:
            torch.Tensor: Class logits [batch, num_classes].
        """
        # Concatenate RGB and spectral along the channel dimension
        x = torch.cat((rgb, spectral), dim=1)
        return self.inception(x)