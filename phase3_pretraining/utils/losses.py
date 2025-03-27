import torch
import torch.nn.functional as F
import logging
from config import TEMPERATURE

# Use a module-specific logger
logger = logging.getLogger(__name__)

class InfoNCELoss:
    def __init__(self):
        self.temperature = TEMPERATURE
    
    def __call__(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        batch_size = features1.shape[0]
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Labels for contrastive loss (diagonal elements are positives)
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss