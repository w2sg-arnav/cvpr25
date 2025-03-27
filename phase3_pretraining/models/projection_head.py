# phase3_pretraining/models/projection_head.py
import torch.nn as nn
from config import PROJECTION_DIM

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = PROJECTION_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)