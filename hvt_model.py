# hvt_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# Updated import to use timm.layers instead of timm.models.layers
from timm.layers import DropPath, trunc_normal_

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        self.layer_scale1 = nn.Parameter(torch.ones(dim) * 1e-5)
        self.layer_scale2 = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x, H, W, mask=None):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Window partition
        window_size = self.window_size
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = shifted_x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)

        # Attention
        if self.shift_size > 0:
            attn_mask = mask
        else:
            attn_mask = None

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = attn_windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.layer_scale1 * x)

        # FFN
        x = x + self.drop_path(self.layer_scale2 * self.mlp(self.norm2(x)))

        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=4, in_channels=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., drop_rate=0.1, attn_drop_rate=0.0,
                 drop_path_rate=0.1, has_multimodal=False):
        super().__init__()
        self.num_classes = num_classes
        self.has_multimodal = has_multimodal
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Swin Transformer blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) + i]
                ) for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(int(embed_dim * 2 ** i_layer)),
                    nn.Linear(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** (i_layer + 1)))
                ))

        # Spectral embedding if multimodal
        if self.has_multimodal:
            self.spectral_patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.fusion = nn.MultiheadAttention(embed_dim=self.num_features, num_heads=num_heads[-1], dropout=drop_rate)

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_rgb, x_spectral=None):
        # Process RGB
        B = x_rgb.shape[0]
        x = self.patch_embed(x_rgb)
        H, W = self.patch_embed.img_size // self.patch_embed.patch_size, self.patch_embed.img_size // self.patch_embed.patch_size
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ModuleList):
                for blk in layer:
                    x = blk(x, H, W)
            else:
                x = layer(x)
                if i < len(self.layers) - 1:
                    H, W = H // 2, W // 2
                    x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    x = x.view(B, -1, H * W).transpose(1, 2)

        x = self.norm(x)
        rgb_features = x.mean(dim=1)  # Global average pooling

        # Process spectral data if available
        spectral_features = None
        if self.has_multimodal and x_spectral is not None:
            spectral = self.spectral_patch_embed(x_spectral)
            spectral = spectral + self.spectral_pos_embed
            spectral = self.pos_drop(spectral)

            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.ModuleList):
                    for blk in layer:
                        spectral = blk(spectral, H, W)
                else:
                    spectral = layer(spectral)
                    if i < len(self.layers) - 1:
                        H, W = H // 2, W // 2
                        spectral = spectral.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                        spectral = spectral.view(B, -1, H * W).transpose(1, 2)

            spectral = self.norm(spectral)
            spectral_features = spectral.mean(dim=1)

        # Fusion of RGB and spectral features
        if spectral_features is not None:
            query = rgb_features.unsqueeze(0)
            key = value = spectral_features.unsqueeze(0)
            fused_features, _ = self.fusion(query, key, value)
            fused_features = fused_features.squeeze(0)
        else:
            fused_features = rgb_features

        # Classification
        logits = self.head(fused_features)
        return rgb_features, logits

class SSLHierarchicalVisionTransformer(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Projection head for SSL
        final_embed_dim = self.base_model.num_features
        self.projection_head = nn.Sequential(
            nn.Linear(final_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

        # Classification head
        self.classification_head = nn.Linear(final_embed_dim, num_classes)

    def forward(self, x_rgb, x_spectral=None):
        features, logits = self.base_model(x_rgb, x_spectral)
        ssl_projection = self.projection_head(features)
        class_logits = self.classification_head(features)
        return ssl_projection, class_logits