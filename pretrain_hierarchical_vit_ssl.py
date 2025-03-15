import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DataParallel
from torch.utils.checkpoint import checkpoint_sequential
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
import logging
from torchvision import transforms
from typing import Tuple, Optional, List
from dataset_utils import CottonLeafDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hierarchical Vision Transformer (HVT) Components
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[16, 8, 4], in_channels=3, embed_dims=[1024, 512, 256]):
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
    def __init__(self, dim: int, num_heads: int = 16):
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
    def __init__(self, num_classes: int, img_size: int = 299, patch_sizes: list = [16, 8, 4], embed_dims: list = [1024, 512, 256], num_heads: int = 16, num_layers: int = 16, has_multimodal: bool = False, spectral_dim: int = 299):
        super().__init__()
        self.has_multimodal = has_multimodal
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, embed_dims=embed_dims)
        self.embed_dim_total = sum(embed_dims)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim_total, nhead=num_heads, dim_feedforward=self.embed_dim_total*4, dropout=0.2, batch_first=True)
            for _ in range(num_layers)
        ])
        
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
        
        segments = 4
        combined_features = checkpoint_sequential(self.transformer_layers, segments, combined_features, use_reentrant=False)
        
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

# Decoder for Masked Image Modeling
class MIMDecoder(nn.Module):
    def __init__(self, embed_dim: int, img_size: int = 299, patch_size: int = 16, out_channels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = int(self.num_patches ** 0.5)  # 18
        self.output_size = self.grid_size * self.patch_size  # 18 * 16 = 288
        self.decoder_embed = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:, :]  # Remove CLS token
        x = x + self.pos_embed
        x = self.decoder_embed(x)
        x = x.view(x.size(0), self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(x.size(0), -1, self.output_size, self.output_size)
        return x

# Masked Image Modeling and Contrastive Learning Model
class SSLHVT(nn.Module):
    def __init__(self, encoder: HierarchicalVisionTransformer, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = MIMDecoder(embed_dim=encoder.embed_dim_total)
        self.mask_ratio = mask_ratio
        self.patch_size = 16
        self.num_patches = (299 // self.patch_size) ** 2
        self.grid_size = int(self.num_patches ** 0.5)
        self.h = self.grid_size * self.patch_size
        self.w = self.grid_size * self.patch_size
        self.projection = nn.Linear(encoder.embed_dim_total, 1024)  # Increased to 1024
        self.temperature = 0.5  # Increased to 0.5

    def random_masking(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        num_patches = self.num_patches
        mask = torch.ones(B, num_patches, device=x.device)
        num_masked = int(self.mask_ratio * num_patches)
        
        indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask.scatter_(1, indices[:, :num_masked], 0)
        
        return mask

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        x = nn.functional.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)
        
        mask = self.random_masking(x)
        
        mask_img = mask.view(B, self.grid_size, self.grid_size)
        mask_img = mask_img.unsqueeze(3).repeat(1, 1, self.patch_size, 1).unsqueeze(4).repeat(1, 1, 1, 1, self.patch_size)
        mask_img = mask_img.view(B, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        mask_img = mask_img.unsqueeze(1).repeat(1, C, 1, 1)
        
        x_masked = x * mask_img
        
        features = self.encoder(x_masked)
        recon = self.decoder(features)
        
        proj = self.projection(features[:, 0, :])
        proj = nn.functional.normalize(proj, dim=1)
        
        return recon, proj

# Contrastive Loss
def contrastive_loss(proj: torch.Tensor, labels: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    B = proj.shape[0]
    sim_matrix = torch.matmul(proj, proj.T) / temperature
    sim_matrix = sim_matrix - torch.eye(B, device=proj.device) * 1e9
    labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    exp_sim = torch.exp(sim_matrix)
    pos_sim = exp_sim * labels_matrix
    neg_sim = exp_sim * (1 - labels_matrix)
    
    loss = -torch.log((pos_sim.sum(dim=1) + 1e-9) / (exp_sim.sum(dim=1) + 1e-9))
    return loss.mean()

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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if not checkpoint.get('normalized', False) else transforms.Lambda(lambda x: x),
])

class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rgb, spectral, label = self.dataset[idx]
        if self.transform and rgb is not None:
            rgb = self.transform(rgb)
        return rgb, spectral, label

augmented_train_dataset = AugmentedDataset(train_dataset, transform=train_transform)
val_dataset = AugmentedDataset(val_dataset, transform=train_transform)

batch_size = 8  # Reduced to 8
train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

logger.info(f"Training set size: {len(augmented_train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")

# Step 2: Initialize SSL Model
encoder = HierarchicalVisionTransformer(
    num_classes=num_classes,
    img_size=299,
    patch_sizes=[16, 8, 4],
    embed_dims=[1024, 512, 256],
    num_heads=16,
    num_layers=16,
    has_multimodal=has_multimodal,
    spectral_dim=299
)
model = SSLHVT(encoder, mask_ratio=0.75)
if num_gpus > 1:
    model = DataParallel(model)
model = model.to(device)
logger.info(f"SSL model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

# Step 3: Define Loss, Optimizer, and Scheduler
criterion_mim = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)  # Increased lr and weight decay
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
scaler = GradScaler()

# Step 4: Pretraining Loop
num_epochs = 50
best_val_loss = float('inf')
best_model_path = os.path.join(output_dir, 'best_ssl_hvt.pth')
patience = 5
patience_counter = 0

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_mim_loss, train_contrastive_loss, train_total = 0.0, 0.0, 0

    for rgb, spectral, labels in train_loader:
        rgb, labels = rgb.to(device), labels.to(device)
        if spectral is not None and has_multimodal:
            spectral = spectral.to(device)
        else:
            spectral = None

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            recon, proj = model(rgb)
            rgb_resized = nn.functional.interpolate(rgb, size=(288, 288), mode='bilinear', align_corners=False)
            mim_loss = criterion_mim(recon, rgb_resized)
            contrastive_loss_val = contrastive_loss(proj, labels)
            loss = 0.7 * mim_loss + 0.3 * contrastive_loss_val  # Adjusted weighting

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_mim_loss += mim_loss.item()
        train_contrastive_loss += contrastive_loss_val.item()
        train_total += 1

    train_mim_loss /= train_total
    train_contrastive_loss /= train_total
    train_loss = 0.7 * train_mim_loss + 0.3 * train_contrastive_loss

    model.eval()
    val_mim_loss, val_contrastive_loss, val_total = 0.0, 0.0, 0

    with torch.no_grad():
        for rgb, spectral, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            if spectral is not None and has_multimodal:
                spectral = spectral.to(device)
            else:
                spectral = None

            with autocast():
                recon, proj = model(rgb)
                rgb_resized = nn.functional.interpolate(rgb, size=(288, 288), mode='bilinear', align_corners=False)
                mim_loss = criterion_mim(recon, rgb_resized)
                contrastive_loss_val = contrastive_loss(proj, labels)
                loss = 0.7 * mim_loss + 0.3 * contrastive_loss_val

            val_mim_loss += mim_loss.item()
            val_contrastive_loss += contrastive_loss_val.item()
            val_total += 1

    val_mim_loss /= val_total
    val_contrastive_loss /= val_total
    val_loss = 0.7 * val_mim_loss + 0.3 * val_contrastive_loss

    scheduler.step(val_loss)

    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train MIM Loss: {train_mim_loss:.4f}, Train Contrastive Loss: {train_contrastive_loss:.4f}, Train Total Loss: {train_loss:.4f}, "
                f"Val MIM Loss: {val_mim_loss:.4f}, Val Contrastive Loss: {val_contrastive_loss:.4f}, Val Total Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, best_model_path)
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