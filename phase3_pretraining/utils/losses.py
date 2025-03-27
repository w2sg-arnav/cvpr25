# phase3_pretraining/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TEMPERATURE

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features1 (torch.Tensor): Features from the first view [batch, projection_dim].
            features2 (torch.Tensor): Features from the second view [batch, projection_dim].
        
        Returns:
            torch.Tensor: InfoNCE loss.
        """
        batch_size = features1.shape[0]
        
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(features1, features2.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss