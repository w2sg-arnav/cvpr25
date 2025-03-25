import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# DropPath (Stochastic Depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output

# HVT components
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[16, 8, 4], in_channels=3, embed_dims=[768, 384, 192]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList()
        
        base_target_size = img_size // patch_sizes[0]
        
        for patch_size, embed_dim in zip(patch_sizes, embed_dims):
            layer = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
        
        self.target_size = base_target_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = []
        for layer in self.layers:
            feat = layer(x)
            feat = nn.functional.interpolate(feat, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
            features.append(feat)
        return tuple(features)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 24):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, rgb_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        B, N, C = rgb_features.shape
        if spectral_features is None:
            return rgb_features

        q = self.query(rgb_features).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(spectral_features).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.value(spectral_features).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out(out)

class TransformerEncoderLayerWithResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.4, drop_path_rate=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.drop_path(self.dropout1(src2))  # Residual connection
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))  # Residual connection
        src = self.norm2(src)
        return src

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 299, patch_sizes: list = [16, 8, 4], embed_dims: list = [768, 384, 192], num_heads: int = 24, num_layers: int = 16, has_multimodal: bool = False, spectral_dim: int = 299, dropout: float = 0.0):
        super().__init__()
        self.has_multimodal = has_multimodal
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, embed_dims=embed_dims)
        self.embed_dim_total = sum(embed_dims)  # 1344, divisible by 24
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithResidual(
                d_model=self.embed_dim_total, 
                nhead=num_heads, 
                dim_feedforward=self.embed_dim_total*4, 
                dropout=dropout, # Pass the new dropout variable
                drop_path_rate=0.2
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim_total))
        self.num_patches = (self.patch_embed.target_size ** 2)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim_total))
        
        self.norm = nn.LayerNorm(self.embed_dim_total)
        self.head = nn.Linear(self.embed_dim_total, num_classes)
        self.num_layers = num_layers
        
        if has_multimodal:
            self.spectral_patch_embed = nn.Sequential(
                nn.Conv2d(1, self.embed_dim_total, kernel_size=patch_sizes[0], stride=patch_sizes[0]),
                nn.BatchNorm2d(self.embed_dim_total),
                nn.ReLU(inplace=True)
            )
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim_total))
            self.fusion = CrossAttentionFusion(self.embed_dim_total, num_heads)

    def forward(self, rgb: torch.Tensor, spectral: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = rgb.shape[0]
        
        multi_scale_features = self.patch_embed(rgb)
        combined_features = torch.cat([f.flatten(2).transpose(1, 2) for f in multi_scale_features], dim=-1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        combined_features = torch.cat((cls_tokens, combined_features), dim=1)
        combined_features = combined_features + self.pos_embed[:, :combined_features.size(1), :]
        
        combined_features = self.dropout(combined_features)
        for layer in self.transformer_layers:
            combined_features = layer(combined_features)
        
        if self.has_multimodal and spectral is not None:
            if len(spectral.shape) == 3:
                spectral = spectral.unsqueeze(1)
            elif len(spectral.shape) != 4:
                raise ValueError(f"Unexpected spectral shape: {spectral.shape}. Expected [B, 1, H, W] or [B, H, W]")

            spectral_features = self.spectral_patch_embed(spectral)
            spectral_features = spectral_features.flatten(2).transpose(1, 2)
            
            spectral_cls_tokens = torch.zeros(B, 1, self.embed_dim_total, device=spectral.device)
            spectral_features = torch.cat((spectral_cls_tokens, spectral_features), dim=1)
            spectral_features = spectral_features + self.spectral_pos_embed[:, :spectral_features.size(1), :]
            
            combined_features = self.fusion(combined_features, spectral_features)
        
        cls_output = combined_features[:, 0]
        cls_output = self.norm(cls_output)
        return self.head(cls_output)