import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
import logging
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
from dataset_utils import CottonLeafDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom transform to denormalize tensor and convert to PIL Image
class DenormalizeToPIL:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        tensor = tensor * self.std + self.mean
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte()
        tensor = tensor.permute(1, 2, 0)
        return Image.fromarray(tensor.numpy())

# SimCLR-style augmentations with tuned pipeline
def get_simclr_transform():
    return transforms.Compose([
        DenormalizeToPIL(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomResizedCrop(size=299, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),  # Reduced intensity
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandAugment(num_ops=2, magnitude=3),  # Reduced magnitude
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# CutMix and MixUp implementation
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    return data, targets, targets[indices], lam

def mixup(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    lam = np.random.beta(alpha, alpha)
    data = lam * data + (1 - lam) * shuffled_data
    return data, targets, targets[indices], lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# Hierarchical Vision Transformer (HVT) Components
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
    def __init__(self, dim: int, num_heads: int = 12):
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

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 299, patch_sizes: list = [16, 8, 4], embed_dims: list = [768, 384, 192], num_heads: int = 12, num_layers: int = 12, has_multimodal: bool = False, spectral_dim: int = 299, stochastic_depth_prob: float = 0.1):
        super().__init__()
        self.has_multimodal = has_multimodal
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, embed_dims=embed_dims)
        self.embed_dim_total = sum(embed_dims)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim_total, nhead=num_heads, dim_feedforward=self.embed_dim_total*4, dropout=0.2, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_scales = nn.ParameterList([nn.Parameter(torch.ones(self.embed_dim_total) * 0.1) for _ in range(num_layers)])
        self.stochastic_depth_prob = stochastic_depth_prob
        
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
        
        for i, layer in enumerate(self.transformer_layers):
            if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
                continue
            combined_features = layer(combined_features)
            combined_features = combined_features + self.layer_scales[i] * combined_features
        
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
        
        return combined_features

class FeaturePyramidAttention(nn.Module):
    def __init__(self, in_dim: int, reduction: int = 4):
        super().__init__()
        self.reduction = reduction
        intermediate_dim = max(1, in_dim // reduction)
        self.conv1 = nn.Conv2d(2, intermediate_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(intermediate_dim, in_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.local_conv1 = nn.Conv2d(in_dim, intermediate_dim, kernel_size=1)
        self.local_conv2 = nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, padding=1)
        self.local_conv3 = nn.Conv2d(intermediate_dim, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat((avg_pool, max_pool), dim=1)
        
        global_path = self.conv1(pooled)
        global_path = self.conv2(global_path)
        global_path = self.conv3(global_path)
        global_path = self.sigmoid(global_path)
        
        local_path = self.local_conv1(x)
        local_path = self.local_conv2(local_path)
        local_path = self.local_conv3(local_path)
        
        out = x * global_path + local_path
        return out

class MIMDecoder(nn.Module):
    def __init__(self, embed_dim: int, img_size: int = 299, patch_size: int = 16, out_channels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = int(self.num_patches ** 0.5)
        self.output_size = self.grid_size * self.patch_size
        
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, patch_size * patch_size * 128)
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(patch_size * patch_size * 128, patch_size * patch_size * 64),
            nn.GELU(),
            nn.Dropout(p=0.5)
        )
        self.skip_projection = nn.Linear(patch_size * patch_size * 128, patch_size * patch_size * 64)
        self.decoder_layer2 = nn.Linear(patch_size * patch_size * 64, patch_size * patch_size * out_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.fpa = FeaturePyramidAttention(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:, :]
        x = self.norm(x)
        x = x + self.pos_embed
        x = self.decoder_embed(x)
        skip = x
        skip = self.skip_projection(skip)
        x = self.decoder_layer1(x)
        x = x + skip
        x = self.decoder_layer2(x)
        x = x.view(x.size(0), self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(x.size(0), -1, self.output_size, self.output_size)
        x = self.fpa(x)
        x = torch.tanh(x)
        return x

class SSLHVT(nn.Module):
    def __init__(self, encoder: HierarchicalVisionTransformer, initial_mask_ratio: float = 0.6):
        super().__init__()
        self.encoder = encoder
        self.decoder = MIMDecoder(embed_dim=encoder.embed_dim_total)
        self.initial_mask_ratio = initial_mask_ratio
        self.current_mask_ratio = initial_mask_ratio
        self.patch_size = 16
        self.num_patches = (299 // self.patch_size) ** 2
        self.grid_size = int(self.num_patches ** 0.5)
        self.h = self.grid_size * self.patch_size
        self.w = self.grid_size * self.patch_size
        self.projection = nn.Sequential(
            nn.Linear(encoder.embed_dim_total, 1024),  # Reduced from 2048
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1024)  # Reduced from 2048
        )
        self.temperature = 0.7  # Increased from 0.5

    def update_mask_ratio(self, epoch, total_epochs):
        progress = epoch / total_epochs
        self.current_mask_ratio = self.initial_mask_ratio + (0.8 - self.initial_mask_ratio) * progress

    def random_rectangular_masking(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mask = torch.ones(B, H, W, device=x.device)
        
        for b in range(B):
            for _ in range(int(self.current_mask_ratio * H * W / (self.patch_size * self.patch_size))):
                h = np.random.randint(self.patch_size // 2, self.patch_size * 2)
                w = np.random.randint(self.patch_size // 2, self.patch_size * 2)
                y = np.random.randint(0, H - h)
                x = np.random.randint(0, W - w)
                mask[b, y:y+h, x:x+w] = 0
        
        return mask

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x1.shape
        
        x1 = nn.functional.interpolate(x1, size=(self.h, self.w), mode='bilinear', align_corners=False)
        x2 = nn.functional.interpolate(x2, size=(self.h, self.w), mode='bilinear', align_corners=False)
        
        mask = self.random_rectangular_masking(x1)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        x1_masked = x1 * mask
        
        features1 = self.encoder(x1_masked)
        features2 = self.encoder(x2)
        recon = self.decoder(features1)
        
        proj1 = self.projection(features1[:, 0, :])
        proj2 = self.projection(features2[:, 0, :])
        proj1 = nn.functional.normalize(proj1, dim=1)
        proj2 = nn.functional.normalize(proj2, dim=1)
        
        return recon, proj1, proj2

# InfoNCE Loss with adjusted temperature
def info_nce_loss(proj1: torch.Tensor, proj2: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    B = proj1.shape[0]
    labels = torch.arange(B, device=proj1.device)
    
    logits = torch.matmul(proj1, proj2.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    
    mask = torch.eye(B, device=proj1.device, dtype=torch.bool)
    log_prob_pos = (log_prob * mask).sum(dim=1) / mask.sum(dim=1)
    loss = -log_prob_pos.mean()
    return loss

# Cyclical Learning Rate Schedule with Warmup
def get_cyclical_lr_schedule(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 0.0005, max_lr: float = 0.002):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (max_lr - min_lr) * (epoch / warmup_epochs) + min_lr
        else:
            cycle = np.floor(1 + (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
            x = np.abs((epoch - warmup_epochs) / (total_epochs - warmup_epochs) - 2 * cycle + 1)
            return min_lr + (max_lr - min_lr) * (1 - x)
    return LambdaLR(optimizer, lr_lambda)

# Dynamic Loss Weighting with Cosine Annealing
def get_loss_weights(epoch, total_epochs):
    progress = epoch / total_epochs
    mim_weight = 0.7 - 0.4 * (0.5 * (1 + np.cos(np.pi * progress)))
    contrastive_weight = 0.3 + 0.4 * (0.5 * (1 + np.cos(np.pi * progress)))
    return mim_weight, contrastive_weight

# Gradient Centralization
def centralize_gradients(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data -= param.grad.data.mean(dim=tuple(range(1, len(param.grad.shape))), keepdim=True)

# Step 1: Load Preprocessed Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    logger.info(f"Using {num_gpus} H100 GPUs with DataParallel.")
else:
    logger.info("Using 1 H100 GPU.")

output_dir = 'phase4_ssl_results'
os.makedirs(output_dir, exist_ok=True)

try:
    checkpoint = torch.load(os.path.join('./phase1_checkpoints', 'phase1_preprocessed_data.pth'))
except FileNotFoundError as e:
    logger.error(f"Preprocessed data not found: {e}")
    raise FileNotFoundError("Run Phase 1 first to generate preprocessed data.")

train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
class_names = checkpoint['class_names']
has_multimodal = checkpoint['has_multimodal']

if len(class_names) > 7:
    class_names = class_names[:7]
num_classes = len(class_names)

simclr_transform = get_simclr_transform()

class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rgb, spectral, label = self.dataset[idx]
        if self.transform and rgb is not None:
            rgb1 = self.transform(rgb)
            rgb2 = self.transform(rgb)
        else:
            rgb1, rgb2 = rgb, rgb
        return rgb1, rgb2, spectral, label

augmented_train_dataset = AugmentedDataset(train_dataset, transform=simclr_transform)
val_dataset = AugmentedDataset(val_dataset, transform=simclr_transform)

batch_size = 16
effective_batch_size = 1024
accumulation_steps = effective_batch_size // batch_size

train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

logger.info(f"Training set size: {len(augmented_train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")

# Step 2: Initialize SSL Model
encoder = HierarchicalVisionTransformer(
    num_classes=num_classes,
    img_size=299,
    patch_sizes=[16, 8, 4],
    embed_dims=[768, 384, 192],
    num_heads=12,
    num_layers=12,
    has_multimodal=has_multimodal,
    spectral_dim=299,
    stochastic_depth_prob=0.1
)
model = SSLHVT(encoder, initial_mask_ratio=0.6)
if num_gpus > 1:
    model = DataParallel(model)
model = model.to(device)
logger.info(f"SSL model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

# Step 3: Define Loss, Optimizer, and Scheduler
criterion_mim = nn.MSELoss()

base_lr = 0.001
param_groups = [
    {'params': [p for n, p in model.named_parameters() if 'decoder' not in n], 'lr': base_lr, 'weight_decay': 0.1},
    {'params': [p for n, p in model.named_parameters() if 'decoder' in n], 'lr': base_lr, 'weight_decay': 0.05}
]
optimizer = optim.AdamW(param_groups)
scheduler = get_cyclical_lr_schedule(optimizer, warmup_epochs=10, total_epochs=50, min_lr=0.0005, max_lr=0.002)
scaler = GradScaler()

# Skip loading checkpoint due to architecture mismatch
start_epoch = 0
best_val_loss = float('inf')
logger.info("Starting training from scratch due to architecture changes (projection head dimensionality increased).")

# Step 4: Pretraining Loop
num_epochs = 50
patience = 15  # Increased from 10
patience_counter = 0

start_time = time.time()

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_mim_loss, train_contrastive_loss, train_total = 0.0, 0.0, 0
    
    model.module.update_mask_ratio(epoch, num_epochs) if num_gpus > 1 else model.update_mask_ratio(epoch, num_epochs)
    
    mim_weight, contrastive_weight = get_loss_weights(epoch, num_epochs)
    
    optimizer.zero_grad(set_to_none=True)
    for i, (rgb1, rgb2, spectral, labels) in enumerate(train_loader):
        rgb1, rgb2, labels = rgb1.to(device), rgb2.to(device), labels.to(device)
        if spectral is not None and has_multimodal:
            spectral = spectral.to(device)
        else:
            spectral = None

        if np.random.rand() < 0.3:
            if np.random.rand() < 0.5:
                rgb1, labels_a, labels_b, lam = cutmix(rgb1, labels)
                rgb2, _, _, _ = cutmix(rgb2, labels)
            else:
                rgb1, labels_a, labels_b, lam = mixup(rgb1, labels)
                rgb2, _, _, _ = mixup(rgb2, labels)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        with autocast():
            recon, proj1, proj2 = model(rgb1, rgb2)
            rgb1_resized = nn.functional.interpolate(rgb1, size=(288, 288), mode='bilinear', align_corners=False)
            mim_loss = criterion_mim(recon, rgb1_resized)
            mim_loss = mim_loss / 1000.0
            contrastive_loss_val = info_nce_loss(proj1, proj2)
            loss = mim_weight * mim_loss + contrastive_weight * contrastive_loss_val
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if 'decoder' in n], max_norm=0.1)
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if 'decoder' not in n], max_norm=0.5)
            centralize_gradients(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_mim_loss += mim_loss.item() * 1000.0
        train_contrastive_loss += contrastive_loss_val.item()
        train_total += 1

    train_mim_loss /= train_total
    train_contrastive_loss /= train_total
    train_loss = mim_weight * train_mim_loss + contrastive_weight * train_contrastive_loss

    model.eval()
    val_mim_loss, val_contrastive_loss, val_total = 0.0, 0.0, 0

    with torch.no_grad():
        for rgb1, rgb2, spectral, labels in val_loader:
            rgb1, rgb2, labels = rgb1.to(device), rgb2.to(device), labels.to(device)
            if spectral is not None and has_multimodal:
                spectral = spectral.to(device)
            else:
                spectral = None

            with autocast():
                recon, proj1, proj2 = model(rgb1, rgb2)
                rgb1_resized = nn.functional.interpolate(rgb1, size=(288, 288), mode='bilinear', align_corners=False)
                mim_loss = criterion_mim(recon, rgb1_resized)
                mim_loss = mim_loss / 1000.0
                contrastive_loss_val = info_nce_loss(proj1, proj2)
                loss = mim_weight * mim_loss + contrastive_weight * contrastive_loss_val

            val_mim_loss += mim_loss.item() * 1000.0
            val_contrastive_loss += contrastive_loss_val.item()
            val_total += 1

    val_mim_loss /= val_total
    val_contrastive_loss /= val_total
    val_loss = 0.4 * val_mim_loss + 0.6 * val_contrastive_loss

    scheduler.step()

    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train MIM Loss: {train_mim_loss:.4f}, Train Contrastive Loss: {train_contrastive_loss:.4f}, Train Total Loss: {train_loss:.4f}, "
                f"Val MIM Loss: {val_mim_loss:.4f}, Val Contrastive Loss: {val_contrastive_loss:.4f}, Val Total Loss: {val_loss:.4f} (Original Weighting)")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, os.path.join(output_dir, 'best_ssl_hvt.pth'))
        logger.info(f"Best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        logger.info(f"Patience counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break

training_time = time.time() - start_time
logger.info(f"Pretraining completed in {training_time/60:.2f} minutes")

# Save the pretrained encoder for fine-tuning in the next phase
torch.save({
    'encoder_state_dict': model.module.encoder.state_dict() if num_gpus > 1 else model.encoder.state_dict(),
}, os.path.join(output_dir, 'pretrained_hvt.pth'))

logger.info(f"Phase 4 complete! Pretrained HVT saved to {output_dir}/pretrained_hvt.pth")