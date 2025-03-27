# models/dfca.py
import torch
import torch.nn as nn

class DiseaseFocusedCrossAttention(nn.Module):
    """Disease-Focused Cross-Attention (DFCA) for multi-modal fusion of RGB and spectral features."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim (int): Embedding dimension for both RGB and spectral features.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Learnable disease prior (e.g., to focus on lesion regions)
        self.disease_mask = nn.Parameter(torch.ones(1, 1, embed_dim))
        
        # Cross-attention module
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, rgb_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_features (torch.Tensor): RGB features [seq_len, batch, embed_dim].
            spectral_features (torch.Tensor): Spectral features [seq_len, batch, embed_dim].
        
        Returns:
            torch.Tensor: Fused features [seq_len, batch, embed_dim].
        """
        # Apply disease mask to spectral features to focus on disease-relevant regions
        spectral_features = spectral_features + self.disease_mask
        
        # Cross-attention: RGB queries, spectral keys/values
        fused_features, _ = self.cross_attention(
            query=rgb_features,
            key=spectral_features,
            value=spectral_features
        )
        fused_features = self.norm(rgb_features + self.dropout(fused_features))
        
        # Feed-forward network
        ffn_out = self.ffn(fused_features)
        fused_features = self.norm2(fused_features + self.dropout(ffn_out))
        
        return fused_features