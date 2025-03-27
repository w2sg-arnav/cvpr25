# models/baseline.py
import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, SPECTRAL_CHANNELS
import logging

class InceptionV3Baseline(nn.Module):
    """Inception V3 baseline for comparison, adapted for multi-modal input."""
    
    def __init__(self, num_classes: int = NUM_CLASSES, spectral_channels: int = SPECTRAL_CHANNELS):
        super().__init__()
        self.spectral_channels = spectral_channels
        
        # Load pretrained Inception V3 with weights
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # Store the original first layer parameters
        original_conv = self.inception.Conv2d_1a_3x3.conv
        original_params = {
            'out_channels': original_conv.out_channels,
            'kernel_size': original_conv.kernel_size,
            'stride': original_conv.stride,
            'padding': original_conv.padding,
            'bias': original_conv.bias is not None
        }
        
        # Create a new first convolutional layer that processes RGB and spectral separately
        self.rgb_conv = nn.Conv2d(3, original_params['out_channels'], 
                                 kernel_size=original_params['kernel_size'],
                                 stride=original_params['stride'],
                                 padding=original_params['padding'],
                                 bias=original_params['bias'])
        
        self.spectral_conv = nn.Conv2d(spectral_channels, original_params['out_channels'],
                                      kernel_size=original_params['kernel_size'],
                                      stride=original_params['stride'],
                                      padding=original_params['padding'],
                                      bias=original_params['bias'])
        
        # Initialize the RGB conv with pretrained weights
        with torch.no_grad():
            self.rgb_conv.weight = nn.Parameter(original_conv.weight.clone())
            if original_conv.bias is not None:
                self.rgb_conv.bias = nn.Parameter(original_conv.bias.clone())
            
            # Initialize spectral conv with mean of RGB weights
            self.spectral_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True).repeat(1, spectral_channels, 1, 1)
            )
            if original_conv.bias is not None:
                self.spectral_conv.bias = nn.Parameter(original_conv.bias.clone())
        
        # Modify the final layer for the number of classes
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)
        
        # Modify the auxiliary classifier's final layer as well
        if self.inception.AuxLogits is not None:
            in_features_aux = self.inception.AuxLogits.fc.in_features
            self.inception.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb (torch.Tensor): RGB input [batch, 3, H, W].
            spectral (torch.Tensor): Spectral input [batch, spectral_channels, H, W].
        
        Returns:
            torch.Tensor: Class logits [batch, num_classes].
        """
        # Log input shapes for debugging
        logging.info(f"RGB shape: {rgb.shape}")
        logging.info(f"Spectral shape: {spectral.shape}")
        
        # Process RGB and spectral separately
        rgb_features = self.rgb_conv(rgb)
        spectral_features = self.spectral_conv(spectral)
        
        # Combine the features
        x = rgb_features + spectral_features
        
        # Pass through the rest of the Inception network
        self.inception.eval()  # Ensure eval mode to avoid aux_logits
        return self.inception(x)