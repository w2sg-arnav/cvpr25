import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.backends import cudnn
import torch.utils.checkpoint as checkpoint

# Set environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Custom augmentations
def mixup_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Device and optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load data
SAVE_PATH = "./phase1_checkpoints"
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'phase1_preprocessed_data.pth')
if not os.path.exists(CHECKPOINT_PATH):
    logger.error(f"Phase 1 checkpoint not found at {CHECKPOINT_PATH}")
    raise FileNotFoundError(f"Phase 1 checkpoint not found at {CHECKPOINT_PATH}")

checkpoint_data = torch.load(CHECKPOINT_PATH)
train_dataset = checkpoint_data['train_dataset']
val_dataset = checkpoint_data['val_dataset']
test_dataset = checkpoint_data['test_dataset']
class_names = checkpoint_data['class_names']
num_classes = len(class_names)
has_multimodal = checkpoint_data['has_multimodal']
class_distribution = checkpoint_data['original_stats']['class_distribution']

# Class weights and sampler
class_counts = np.array([class_distribution[name][0] for name in class_names])
class_weights = torch.FloatTensor(1.0 / (class_counts / class_counts.sum())).to(device)
class_weights = torch.clamp(class_weights, min=1.0, max=10.0)
class_weights[6] = 2.0
class_weights[0:6] *= 1.5

labels = [int(sample[2].item() if isinstance(sample[2], torch.Tensor) else sample[2]) for sample in train_dataset]
sample_weights = [class_weights[label].item() for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoaders
batch_size = 32  # Further reduced
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=16, pin_memory=True)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        target_one_hot = F.one_hot(target, num_classes=input.size(-1)).float()
        ce_loss = -torch.sum(target_one_hot * log_prob, dim=-1)
        pt = torch.sum(target_one_hot * prob, dim=-1)
        focal_weight = (1 - pt).pow(self.gamma)
        if self.weight is not None:
            focal_weight *= self.weight.gather(0, target)
        loss = focal_weight * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

# HVT components
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[8, 16, 32], in_chans=3, embed_dim=256):  # Reduced embed_dim
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_patches = [(img_size // p) ** 2 for p in patch_sizes]
        self.projections = nn.ModuleList([
            nn.Conv2d(in_chans, embed_dim, kernel_size=p, stride=p) for p in patch_sizes
        ])
        self.embed_dim = embed_dim

    def forward(self, x):
        return [proj(x).flatten(2).transpose(1, 2) for proj in self.projections]

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, rgb_features, spectral_features):
        rgb_features = self.norm1(rgb_features)
        if spectral_features is None:
            return rgb_features
        spectral_features = self.norm2(spectral_features)
        fused, _ = self.cross_attention(rgb_features, spectral_features, spectral_features)
        return self.norm3(fused + rgb_features)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.15, stochastic_depth_prob=0.2):
        super().__init__()
        self.norm0 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.LayerNorm(int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = nn.Dropout(p=stochastic_depth_prob)

    def forward(self, x):
        x = self.norm0(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.stochastic_depth(x)
        mlp_output = self.mlp(x)
        return self.norm2(x + self.dropout(mlp_output))

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[8, 16, 32], in_chans=3, embed_dim=256,  # Reduced embed_dim
                 num_heads=8, depth=8, num_classes=7, dropout=0.15, stochastic_depth_prob=0.2):  # Reduced depth
        super().__init__()
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.cls_tokens = nn.Parameter(torch.zeros(len(patch_sizes), 1, embed_dim))
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n + 1, embed_dim)) for n in self.num_patches
        ])
        self.dropout = nn.Dropout(dropout)
        self.spectral_patch_embeds = nn.ModuleList([
            nn.Conv2d(1, embed_dim, kernel_size=p, stride=p) if has_multimodal else None
            for p in patch_sizes
        ])
        self.cross_attention = CrossAttentionFusion(embed_dim, num_heads)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim * len(patch_sizes))
        self.fusion_head = nn.Linear(embed_dim * len(patch_sizes), embed_dim)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for param in (self.cls_tokens, *self.pos_embeds):
            nn.init.trunc_normal_(param, std=0.02)

    def forward(self, rgb, spectral=None):
        B = rgb.shape[0]
        multi_scale_features = self.patch_embed(rgb)
        scale_outputs = []
        for i, (features, pos_embed) in enumerate(zip(multi_scale_features, self.pos_embeds)):
            cls_tokens = self.cls_tokens[i].expand(B, -1, -1)
            features = torch.cat((cls_tokens, features), dim=1) + pos_embed.expand(B, -1, -1)
            features = self.dropout(features)
            if spectral is not None and self.spectral_patch_embeds[i] is not None:
                spectral_features = self.spectral_patch_embeds[i](spectral.unsqueeze(1)).flatten(2).transpose(1, 2)
                spectral_cls = torch.zeros(B, 1, self.embed_dim).to(spectral.device)
                spectral_features = torch.cat((spectral_cls, spectral_features), dim=1)
                features = self.cross_attention(features, spectral_features)
            for layer in self.transformer_layers:
                # Use checkpointing for transformer layers
                features = checkpoint.checkpoint(layer, features, use_reentrant=False)
            scale_outputs.append(self.norm(features)[:, 0])
        combined_features = self.fusion_norm(torch.cat(scale_outputs, dim=-1))
        fused_features = self.head_norm(self.fusion_head(combined_features))
        return self.head(fused_features)

# Training with gradient accumulation and memory logging
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name="HVT"):
    scaler = GradScaler(enabled=True)
    best_val_acc = 0.0
    best_model_path = f"./phase2_checkpoints/{model_name}_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    patience = 15
    patience_counter = 0
    mixup_alpha = 0.4
    use_cutmix = True
    accum_steps = 4  # Simulate batch_size=128
    
    # Disable torch.compile for now
    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            spectral = batch[1].to(device, non_blocking=True) if has_multimodal else None
            
            if inputs.min() < -2.5 or inputs.max() > 2.5:
                inputs = (inputs - rgb_mean) / rgb_std
            if spectral is not None and (spectral.min() < 0 or spectral.max() > 1):
                spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)
            
            if epoch > 0 and np.random.random() > 0.3:
                if np.random.random() > 0.5 and use_cutmix:
                    inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=mixup_alpha)
                else:
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            
            with autocast():
                outputs = model(inputs, spectral)
                loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # Log memory usage
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_max = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Batch {i+1}: Memory Used: {mem_used:.2f} GiB, Max: {mem_max:.2f} GiB")
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[2].to(device, non_blocking=True)
                spectral = batch[1].to(device, non_blocking=True) if has_multimodal else None
                if inputs.min() < -2.5 or inputs.max() > 2.5:
                    inputs = (inputs - rgb_mean) / rgb_std
                if spectral is not None and (spectral.min() < 0 or spectral.max() > 1):
                    spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)
                outputs = model(inputs, spectral)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} - Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    return best_model_path, training_time

# Evaluation
def evaluate_model(model, test_loader, criterion, device, class_names):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            spectral = batch[1].to(device, non_blocking=True) if has_multimodal else None
            if inputs.min() < -2.5 or inputs.max() > 2.5:
                inputs = (inputs - rgb_mean) / rgb_std
            if spectral is not None and (spectral.min() < 0 or spectral.max() > 1):
                spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)
            outputs = model(inputs, spectral)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct / total
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_labels)
    eval_time = time.time() - start_time
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    return test_acc, report, cm, eval_time

# Main
if __name__ == "__main__":
    try:
        hvt_model = HierarchicalVisionTransformer(
            img_size=299, patch_sizes=[8, 16, 32], in_chans=3, embed_dim=256,
            num_heads=8, depth=8, num_classes=num_classes, dropout=0.15, stochastic_depth_prob=0.1
        ).to(device)
        
        criterion_hvt = FocalLoss(weight=class_weights, gamma=1.5)
        optimizer_hvt = optim.AdamW(hvt_model.parameters(), lr=5e-5, weight_decay=1e-2, betas=(0.9, 0.95))
        warmup_epochs = 10
        total_epochs = 25
        scheduler_hvt = LambdaLR(optimizer_hvt, lr_lambda=lambda e: min(1.0, (e + 1) / warmup_epochs))
        if total_epochs > warmup_epochs:
            scheduler_hvt = torch.optim.lr_scheduler.SequentialLR(
                optimizer_hvt,
                schedulers=[scheduler_hvt, CosineAnnealingLR(optimizer_hvt, T_max=total_epochs - warmup_epochs, eta_min=1e-6)],
                milestones=[warmup_epochs]
            )
        
        logger.info("Training HVT architecture...")
        best_model_path, training_time = train_model(
            hvt_model, train_loader, val_loader, criterion_hvt, optimizer_hvt, scheduler_hvt, total_epochs, device
        )
        
        hvt_model.load_state_dict(torch.load(best_model_path))
        test_acc, report, cm, eval_time = evaluate_model(hvt_model, test_loader, criterion_hvt, device, class_names)
        
        logger.info(f"Total Training Time: {training_time:.2f} seconds")
        logger.info(f"Total Evaluation Time: {eval_time:.2f} seconds")
        logger.info("Phase 2 completed: HVT trained and evaluated.")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise