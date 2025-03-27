import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=168):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
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

    def get_attention_mask(self, H, W):
        if self.shift_size == 0:
            return None

        device = next(self.parameters()).device
        img_mask = torch.zeros((1, H, W, 1), device=device)

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.get_attention_mask(Hp, Wp))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        device = next(self.parameters()).device
        x = x.to(device)

        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
        elif x.dim() == 3:
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=3, img_size=256, patch_size=16, in_channels=3,
                 embed_dim=168, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, has_multimodal=False, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.has_multimodal = has_multimodal
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.device = device

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if has_multimodal:
            self.spectral_patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.patch_mergings = nn.ModuleList()
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
                    drop_path=dpr[sum(depths[:i_layer]) + i])
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.patch_mergings.append(PatchMerging(int(embed_dim * 2 ** i_layer)))

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)

        if has_multimodal:
            self.fusion = nn.MultiheadAttention(embed_dim=self.num_features, num_heads=num_heads[-1], dropout=drop_rate)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_pos_embed(self, checkpoint_pos_embed, target_num_patches, target_embed_dim):
        """Adapt positional embeddings from checkpoint to current model size."""
        old_num_patches, old_embed_dim = checkpoint_pos_embed.shape[1], checkpoint_pos_embed.shape[2]
        new_grid_size = int(target_num_patches ** 0.5)
        
        # Check if checkpoint includes a class token
        old_grid_size = int((old_num_patches - 1) ** 0.5) if old_num_patches != (int(old_num_patches ** 0.5) ** 2) else int(old_num_patches ** 0.5)
        has_cls_token = (old_num_patches != old_grid_size * old_grid_size)
        
        if has_cls_token:
            cls_token = checkpoint_pos_embed[:, :1, :]  # Extract class token
            spatial_embed = checkpoint_pos_embed[:, 1:, :]  # Spatial embeddings
            spatial_size = old_num_patches - 1
        else:
            cls_token = None
            spatial_embed = checkpoint_pos_embed
            spatial_size = old_num_patches

        # Reshape spatial embeddings to grid
        if spatial_size != old_grid_size * old_grid_size:
            raise ValueError(f"Cannot reshape {spatial_size} patches into a square grid (got {old_grid_size}x{old_grid_size})")
        
        spatial_embed = spatial_embed.view(1, old_grid_size, old_grid_size, old_embed_dim)
        
        # Interpolate to target grid size
        spatial_embed = F.interpolate(
            spatial_embed.permute(0, 3, 1, 2),
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        # Adjust embedding dimension if needed
        if old_embed_dim != target_embed_dim:
            spatial_embed = spatial_embed.view(1, target_num_patches, old_embed_dim)
            proj = nn.Linear(old_embed_dim, target_embed_dim).to(spatial_embed.device)
            spatial_embed = proj(spatial_embed)
        
        # Reattach class token if present
        if has_cls_token:
            if old_embed_dim != target_embed_dim:
                cls_proj = nn.Linear(old_embed_dim, target_embed_dim).to(cls_token.device)
                cls_token = cls_proj(cls_token)
            adapted_pos_embed = torch.cat([cls_token, spatial_embed.view(1, target_num_patches, target_embed_dim)], dim=1)
        else:
            adapted_pos_embed = spatial_embed.view(1, target_num_patches, target_embed_dim)
        
        return adapted_pos_embed

    def forward_features(self, x, is_spectral=False):
        x = self.spectral_patch_embed(x) if is_spectral else self.patch_embed(x)
        pos_embed = self.spectral_pos_embed if is_spectral else self.pos_embed
        x = x + pos_embed.to(x.device)
        x = self.pos_drop(x)

        H, W = self.patch_embed.img_size // self.patch_embed.patch_size, self.patch_embed.img_size // self.patch_embed.patch_size
        for i_layer in range(self.num_layers):
            for blk in self.layers[i_layer]:
                x = blk(x, H, W)
            if i_layer < self.num_layers - 1:
                x = x.view(x.shape[0], H, W, -1)
                x = self.patch_mergings[i_layer](x, H, W)
                H, W = H // 2, W // 2

        x = self.norm(x)
        return x, H, W

    def forward(self, x_rgb, x_spectral=None):
        device = next(self.parameters()).device
        x_rgb = x_rgb.to(device)
        if x_spectral is not None:
            x_spectral = x_spectral.to(device)

        rgb_features, H, W = self.forward_features(x_rgb)
        rgb_features = rgb_features.mean(dim=1)

        spectral_features = None
        if self.has_multimodal and x_spectral is not None:
            spectral_features, _, _ = self.forward_features(x_spectral, is_spectral=True)
            spectral_features = spectral_features.mean(dim=1)

        if spectral_features is not None:
            query = rgb_features.unsqueeze(0)
            key = value = spectral_features.unsqueeze(0)
            fused_features, _ = self.fusion(query, key, value)
            fused_features = fused_features.squeeze(0)
        else:
            fused_features = rgb_features

        logits = self.head(fused_features)
        return rgb_features, logits

class SSLHierarchicalVisionTransformer(nn.Module):
    def __init__(self, base_model, num_classes, device='cuda'):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.device = device

        final_embed_dim = self.base_model.num_features
        self.projection_head = nn.Sequential(
            nn.Linear(final_embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )
        self.classification_head = nn.Linear(final_embed_dim, num_classes)

        self.to(device)

    def forward(self, x_rgb, x_spectral=None):
        x_rgb = x_rgb.to(self.device)
        if x_spectral is not None:
            x_spectral = x_spectral.to(self.device)

        features, logits = self.base_model(x_rgb, x_spectral)
        ssl_projection = self.projection_head(features)
        class_logits = self.classification_head(features)
        return ssl_projection, class_logits