import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, num_classes=3, img_size=288, patch_sizes=[16, 8, 4], 
                 embed_dims=[768, 384, 192], num_heads=12, num_layers=12,
                 has_multimodal=False, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.has_multimodal = has_multimodal
        
        # Patch embedding
        self.patch_embeds = nn.ModuleList([
            nn.Conv2d(3, dim, kernel_size=size, stride=size)
            for size, dim in zip(patch_sizes, embed_dims)
        ])
        
        # Transformer encoders
        self.encoders = nn.ModuleList()
        for dim in embed_dims:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim*4,
                dropout=dropout,
                batch_first=True
            )
            self.encoders.append(
                nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            )
        
        # Multimodal
        if has_multimodal:
            self.spectral_proj = nn.Sequential(
                nn.Conv2d(1, embed_dims[-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dims[-1]),
                nn.GELU()
            )
            self.fusion = nn.MultiheadAttention(embed_dims[-1], num_heads, dropout=dropout, batch_first=True)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dims[-1]),
            nn.Linear(embed_dims[-1], num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, spectral=None):
        # Process through each level
        features = []
        for i, (embed, encoder) in enumerate(zip(self.patch_embeds, self.encoders)):
            # Patch embedding
            x_patch = embed(x)
            B, C, H, W = x_patch.shape
            x_patch = x_patch.flatten(2).transpose(1, 2)
            
            # Positional encoding
            pos_embed = nn.Parameter(torch.zeros(1, x_patch.shape[1], C)).to(x.device)
            x_patch = x_patch + pos_embed
            
            # Transformer encoder
            x_patch = encoder(x_patch)
            features.append(x_patch)
            
            # Prepare for next level
            if i < len(self.patch_embeds) - 1:
                x_patch = x_patch.transpose(1, 2).reshape(B, C, H, W)
                x_patch = F.interpolate(x_patch, scale_factor=0.5, mode='bilinear')
                x = x_patch
        
        # Multimodal fusion
        if self.has_multimodal and spectral is not None:
            if spectral.dim() == 3:
                spectral = spectral.unsqueeze(1)
            spectral = self.spectral_proj(spectral)
            spectral = spectral.flatten(2).transpose(1, 2)
            x_patch, _ = self.fusion(x_patch, spectral, spectral)
        
        # Classification
        x_patch = x_patch.mean(dim=1)
        logits = self.head(x_patch)
        
        return features, logits

class SSLHierarchicalVisionTransformer(nn.Module):
    def __init__(self, base_model, num_classes=3):
        super().__init__()
        self.base_model = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.head[-1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x, spectral=None):
        features, x = self.base_model(x, spectral)
        logits = self.projection_head(x)
        return features, logits