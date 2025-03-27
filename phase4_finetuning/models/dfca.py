# phase3_pretraining/models/dfca.py
import torch
import torch.nn as nn

class DiseaseFocusedCrossAttention(nn.Module):
    """Cross-attention mechanism for multi-modal fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        # Cross-attention: RGB attends to spectral
        attn_out, _ = self.attention(rgb, spectral, spectral)
        rgb = self.norm(rgb + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(rgb)
        rgb = self.norm(rgb + ffn_out)
        return rgb