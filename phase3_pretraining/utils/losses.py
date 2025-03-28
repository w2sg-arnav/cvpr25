import torch.nn as nn
import torch
import torch.nn.functional as F
import logging
from config import TEMPERATURE

logger = logging.getLogger(__name__)

class InfoNCELoss(nn.Module):  # Make it a nn.Module
    def __init__(self, temperature=TEMPERATURE):
        super().__init__() # add super().__init__()
        self.temperature = temperature

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        batch_size = features1.shape[0]
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        labels = torch.arange(batch_size, device=features1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss