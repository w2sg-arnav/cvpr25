import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
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
    def __init__(self, img_size: tuple, embed_dim: int = 128, num_heads: int = 4, num_blocks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // 8) * (img_size[1] // 8)

        self.rgb_patch_embed = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            HVTBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 256),
        )

        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)

    def forward(self, rgb: torch.Tensor, pretrain: bool = True):
        rgb = self.rgb_patch_embed(rgb)
        rgb = rgb.flatten(2).transpose(1, 2)
        x = rgb + self.rgb_pos_embed

        for i, block in enumerate(self.blocks):
            x = checkpoint(block, x, use_reentrant=False)

        if pretrain:
            x = x.mean(dim=1)
            x = self.projection_head(x)

        return x