import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)

class HVTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size: tuple, embed_dim: int = 256, num_heads: int = 8, num_blocks: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // 4) * (img_size[1] // 4)
        
        # RGB patch embedding
        self.rgb_patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Spectral patch embedding (disabled for pretraining if unused)
        self.spectral_patch_embed = nn.Conv2d(10, embed_dim, kernel_size=4, stride=4)
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HVTBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])
        
        # Projection head for SimCLR
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 256),
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor = None, pretrain: bool = True):
        # RGB path
        rgb = self.rgb_patch_embed(rgb)  # [batch, embed_dim, H/4, W/4]
        rgb = rgb.flatten(2).transpose(1, 2)  # [batch, (H/4)*(W/4), embed_dim]
        x = rgb + self.rgb_pos_embed  # Add positional embedding
        
        # Spectral path (only if provided)
        if spectral is not None:
            spectral = self.spectral_patch_embed(spectral)
            spectral = spectral.flatten(2).transpose(1, 2) + self.spectral_pos_embed
            x = torch.cat((x, spectral), dim=1)
        
        # Full Transformer loop
        for i, block in enumerate(self.blocks):
            x = block(x)
            logger.debug(f"After block {i}, x requires grad: {x.requires_grad}")
        
        if pretrain:
            x = x.mean(dim=1)  # [batch, embed_dim]
            x = self.projection_head(x)  # [batch, 256]
            logger.debug(f"Projection output requires grad: {x.requires_grad}")
        
        return x