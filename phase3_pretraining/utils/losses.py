import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TEMPERATURE

class InfoNCELoss(nn.Module):
    """InfoNCE Loss for contrastive learning."""
    
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss between two sets of features.
        
        Args:
            features1 (torch.Tensor): Features from the first view [batch_size, feature_dim].
            features2 (torch.Tensor): Features from the second view [batch_size, feature_dim].
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        batch_size = features1.shape[0]
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss