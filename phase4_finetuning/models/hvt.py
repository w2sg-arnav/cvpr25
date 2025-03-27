# phase4_finetuning/models/hvt.py
import torch
import torch.nn as nn
import logging  # Add this import
from config import NUM_CLASSES

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim=112, window_size=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size
        logging.info(f"Initialized SwinTransformerBlock with window_size={window_size}")

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 4
        self.num_patches = (img_size[0] // self.patch_size) * (img_size[1] // self.patch_size)
        self.embed_dim = 112

        # Patch embedding for RGB
        self.patch_embed_rgb = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=self.embed_dim, window_size=4) for _ in range(12)
        ])

        # Global average pooling and classifier
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, NUM_CLASSES)

    def forward(self, rgb):
        # Patch embedding for RGB
        x = self.patch_embed_rgb(rgb)  # (B, embed_dim, H/patch_size, W/patch_size)

        # Apply Swin Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (B, embed_dim)
        x = self.norm(x)
        x = self.head(x)
        return x