# phase3_pretraining/models/hvt.py
import torch
import torch.nn as nn
import math
from typing import List, Tuple
from .dfca import DiseaseFocusedCrossAttention
from .projection_head import ProjectionHead
from config import PATCH_SIZE, EMBED_DIM, DEPTHS, NUM_HEADS, WINDOW_SIZE, NUM_CLASSES, SPECTRAL_CHANNELS
import logging

class PatchEmbed(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, img_size: Tuple[int, int], patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
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
        logging.info(f"Initialized SwinTransformerBlock with window_size={self.window_size}")
    
    def forward(self, x: torch.Tensor, num_patches_h: int, num_patches_w: int) -> torch.Tensor:
        batch, num_patches, embed_dim = x.shape
        assert num_patches == num_patches_h * num_patches_w, "Mismatch in number of patches"
        
        # Compute number of windows, padding if necessary
        num_windows_h = (num_patches_h + self.window_size - 1) // self.window_size  # Ceiling division
        num_windows_w = (num_patches_w + self.window_size - 1) // self.window_size
        padded_h = num_windows_h * self.window_size
        padded_w = num_windows_w * self.window_size
        
        # Pad the patch grid if necessary
        if padded_h != num_patches_h or padded_w != num_patches_w:
            padding_h = padded_h - num_patches_h
            padding_w = padded_w - num_patches_w
            x = x.view(batch, num_patches_h, num_patches_w, embed_dim)
            x = nn.functional.pad(x, (0, 0, 0, padding_w, 0, padding_h))  # Pad height and width
            num_patches_h, num_patches_w = padded_h, padded_w
        
        # Reshape for windowed attention
        x = x.view(batch, num_patches_h, num_patches_w, embed_dim)
        x = x.view(batch, num_windows_h, self.window_size, num_windows_w, self.window_size, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [batch, num_windows_h, num_windows_w, window_size, window_size, embed_dim]
        x = x.view(batch * num_windows_h * num_windows_w, self.window_size * self.window_size, embed_dim)
        
        # Apply attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Reshape back to [batch, num_patches, embed_dim]
        x = x.view(batch, num_windows_h, num_windows_w, self.window_size, self.window_size, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, num_patches_h, num_patches_w, embed_dim)
        x = x.view(batch, num_patches_h * num_patches_w, embed_dim)
        
        # Trim padding if necessary
        x = x[:, :num_patches, :]
        return x

class DiseaseAwareHVT(nn.Module):
    """Hierarchical Vision Transformer with Disease-Focused Cross-Attention."""
    
    def __init__(self, img_size: Tuple[int, int], patch_size: int = PATCH_SIZE, embed_dim: int = EMBED_DIM,
                 depths: List[int] = DEPTHS, num_heads: List[int] = NUM_HEADS, window_size: int = WINDOW_SIZE,
                 num_classes: int = NUM_CLASSES, spectral_channels: int = SPECTRAL_CHANNELS):
        super().__init__()
        self.img_size = img_size
        self.num_stages = len(depths)
        self.embed_dim = embed_dim  # Store initial embed_dim for projection head
        
        # Patch embedding for RGB and spectral
        self.rgb_patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.spectral_patch_embed = PatchEmbed(img_size, patch_size, spectral_channels, embed_dim)
        
        # Positional embeddings
        self.pos_embed_rgb = nn.Parameter(torch.zeros(1, self.rgb_patch_embed.num_patches, embed_dim))
        self.pos_embed_spectral = nn.Parameter(torch.zeros(1, self.spectral_patch_embed.num_patches, embed_dim))
        
        # Transformer stages
        self.stages = nn.ModuleList()
        current_embed_dim = embed_dim
        for i in range(self.num_stages):
            stage = nn.ModuleList([
                SwinTransformerBlock(current_embed_dim, num_heads[i], window_size)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            # Patch merging after each stage (except the last)
            if i < self.num_stages - 1:
                current_embed_dim *= 2
                self.stages.append(nn.Conv2d(current_embed_dim // 2, current_embed_dim, kernel_size=2, stride=2))
        
        # Disease-Focused Cross-Attention for multi-modal fusion
        self.dfca = DiseaseFocusedCrossAttention(current_embed_dim, num_heads[-1])
        
        # Normalization and classification head (for fine-tuning)
        self.norm = nn.LayerNorm(current_embed_dim)
        self.head = nn.Linear(current_embed_dim, num_classes)
        
        # Projection head for pretraining
        self.projection_head = ProjectionHead(current_embed_dim)
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor, pretrain: bool = False) -> torch.Tensor:
        """
        Args:
            rgb (torch.Tensor): RGB input [batch, 3, H, W].
            spectral (torch.Tensor): Spectral input [batch, spectral_channels, H, W].
            pretrain (bool): If True, return features for pretraining; if False, return logits for classification.
        
        Returns:
            torch.Tensor: Features for pretraining or logits for classification.
        """
        # Patch embedding
        rgb = self.rgb_patch_embed(rgb) + self.pos_embed_rgb
        spectral = self.spectral_patch_embed(spectral) + self.pos_embed_spectral
        
        # Compute patch grid dimensions
        num_patches_h = self.rgb_patch_embed.num_patches_h
        num_patches_w = self.rgb_patch_embed.num_patches_w
        
        # Hierarchical stages
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):
                # Transformer blocks
                for block in stage:
                    rgb = block(rgb, num_patches_h, num_patches_w)
                    spectral = block(spectral, num_patches_h, num_patches_w)
            else:
                # Patch merging (Conv2d layer)
                rgb = rgb.transpose(1, 2).view(rgb.shape[0], -1, num_patches_h, num_patches_w)
                rgb = stage(rgb).flatten(2).transpose(1, 2)
                spectral = spectral.transpose(1, 2).view(spectral.shape[0], -1, num_patches_h, num_patches_w)
                spectral = stage(spectral).flatten(2).transpose(1, 2)
                num_patches_h //= 2
                num_patches_w //= 2
        
        # Multi-modal fusion with DFCA
        fused_features = self.dfca(rgb, spectral)
        
        if pretrain:
            # For pretraining: return features after projection head
            features = self.norm(fused_features.mean(dim=1))  # Global average pooling
            return self.projection_head(features)
        else:
            # For fine-tuning: return logits
            fused_features = self.norm(fused_features.mean(dim=1))  # Global average pooling
            logits = self.head(fused_features)
            return logits