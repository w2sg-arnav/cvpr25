# models/hvt.py
import torch
import torch.nn as nn
import math
from typing import List, Tuple
from .dfca import DiseaseFocusedCrossAttention
from config import PATCH_SIZE, EMBED_DIM, DEPTHS, NUM_HEADS, WINDOW_SIZE, NUM_CLASSES, SPECTRAL_CHANNELS

class PatchEmbed(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, img_size: Tuple[int, int], patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [batch, embed_dim, h', w']
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with windowed attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.window_size = window_size
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        batch, num_patches, embed_dim = x.shape
        # Reshape for windowed attention
        num_windows = (h // self.window_size) * (w // self.window_size)
        x = x.view(batch, h // self.window_size, self.window_size, w // self.window_size, self.window_size, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch * num_windows, self.window_size * self.window_size, embed_dim)
        
        # Apply attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Reshape back
        x = x.view(batch, h // self.window_size, w // self.window_size, self.window_size, self.window_size, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, num_patches, embed_dim)
        return x

class DiseaseAwareHVT(nn.Module):
    """Hierarchical Vision Transformer with Disease-Focused Cross-Attention."""
    
    def __init__(self, img_size: Tuple[int, int], patch_size: int = PATCH_SIZE, embed_dim: int = EMBED_DIM,
                 depths: List[int] = DEPTHS, num_heads: List[int] = NUM_HEADS, window_size: int = WINDOW_SIZE,
                 num_classes: int = NUM_CLASSES, spectral_channels: int = SPECTRAL_CHANNELS):
        super().__init__()
        self.img_size = img_size
        self.num_stages = len(depths)
        
        # Patch embedding for RGB and spectral
        self.rgb_patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.spectral_patch_embed = PatchEmbed(img_size, patch_size, spectral_channels, embed_dim)
        
        # Positional embeddings
        self.pos_embed_rgb = nn.Parameter(torch.zeros(1, self.rgb_patch_embed.num_patches, embed_dim))
        self.pos_embed_spectral = nn.Parameter(torch.zeros(1, self.spectral_patch_embed.num_patches, embed_dim))
        
        # Transformer stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.ModuleList([
                SwinTransformerBlock(embed_dim, num_heads[i], window_size)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            # Patch merging after each stage (except the last)
            if i < self.num_stages - 1:
                embed_dim *= 2
                self.stages.append(nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2))
        
        # Disease-Focused Cross-Attention for multi-modal fusion
        self.dfca = DiseaseFocusedCrossAttention(embed_dim, num_heads[-1])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb (torch.Tensor): RGB input [batch, 3, H, W].
            spectral (torch.Tensor): Spectral input [batch, spectral_channels, H, W].
        
        Returns:
            torch.Tensor: Class logits [batch, num_classes].
        """
        # Patch embedding
        h, w = self.img_size
        rgb = self.rgb_patch_embed(rgb) + self.pos_embed_rgb
        spectral = self.spectral_patch_embed(spectral) + self.pos_embed_spectral
        
        # Hierarchical stages
        for i, stage in enumerate(self.stages):
            for block in stage:
                if isinstance(block, SwinTransformerBlock):
                    h = h // (2 if i > 0 else 1)
                    w = w // (2 if i > 0 else 1)
                    rgb = block(rgb, h, w)
                    spectral = block(spectral, h, w)
                else:
                    # Patch merging
                    rgb = rgb.transpose(1, 2).view(rgb.shape[0], -1, h, w)
                    rgb = block(rgb).flatten(2).transpose(1, 2)
                    spectral = spectral.transpose(1, 2).view(spectral.shape[0], -1, h, w)
                    spectral = block(spectral).flatten(2).transpose(1, 2)
        
        # Multi-modal fusion with DFCA
        fused_features = self.dfca(rgb, spectral)
        
        # Classification
        fused_features = self.norm(fused_features.mean(dim=1))  # Global average pooling
        logits = self.head(fused_features)
        return logits