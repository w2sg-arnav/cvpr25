import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# Custom augmentations (Mixup and CutMix)
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

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load Phase 1 data
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
logger.info(f"Multimodal data enabled: {has_multimodal}")

# Compute class weights for imbalance and oversampling
class_counts = np.array([class_distribution[name][0] for name in class_names])
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = torch.FloatTensor(class_weights).to(device)
class_weights = torch.clamp(class_weights, min=1.0, max=10.0)
class_weights[6] = 2.0
class_weights[0:6] = class_weights[0:6] * 1.5
logger.info(f"Class weights (adjusted): {class_weights.tolist()}")

# Oversampling for minority classes
labels = []
for sample in train_dataset:
    label = sample[2]
    if isinstance(label, torch.Tensor):
        label = label.item()
    labels.append(int(label))
sample_weights = [class_weights[label].item() for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Create DataLoaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Custom Focal Loss with numerical stability
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # Increased for stability
        
    def forward(self, input, target):
        # Apply log_softmax for numerical stability
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        
        # One-hot encoding target for stability
        target_one_hot = F.one_hot(target, num_classes=input.size(-1)).float()
        
        # Calculate cross entropy loss
        ce_loss = -torch.sum(target_one_hot * log_prob, dim=-1)
        
        # Calculate focal weights
        pt = torch.sum(target_one_hot * prob, dim=-1)
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Apply class weights if provided
        if self.weight is not None:
            class_weights = self.weight.gather(0, target)
            focal_weight = focal_weight * class_weights
        
        # Apply focal weighting to CE loss
        loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# Hierarchical Vision Transformer (HVT) Design
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[8, 16, 32], in_chans=3, embed_dim=384):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_patches = []
        self.projections = nn.ModuleList()
        
        for patch_size in patch_sizes:
            num_patch = (img_size // patch_size) ** 2
            self.num_patches.append(num_patch)
            proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.projections.append(proj)
        
        self.embed_dim = embed_dim

    def forward(self, x):
        multi_scale_features = []
        for proj in self.projections:
            patches = proj(x).flatten(2).transpose(1, 2)
            multi_scale_features.append(patches)
        return multi_scale_features

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
        # Use batch_first=True format
        fused, _ = self.cross_attention(rgb_features, spectral_features, spectral_features)
        return self.norm3(fused + rgb_features)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.15, stochastic_depth_prob=0.2):
        super().__init__()
        self.norm0 = nn.LayerNorm(embed_dim)
        # Set batch_first=True for consistent format
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Add layernorms between MLP layers for stability
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
        # Use batch_first=True format
        attn_output, attn_weight = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.stochastic_depth(x)
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))
        return x, attn_weight

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[8, 16, 32], in_chans=3, embed_dim=384, 
                 num_heads=8, depth=12, num_classes=7, dropout=0.15, stochastic_depth_prob=0.2):
        super().__init__()
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.embed_dim_total = embed_dim * len(patch_sizes)
        
        # Better initialization for cls tokens
        self.cls_tokens = nn.Parameter(torch.zeros(len(patch_sizes), 1, embed_dim))
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patch + 1, embed_dim))
            for num_patch in self.num_patches
        ])
        self.dropout = nn.Dropout(dropout)
        
        self.spectral_patch_embeds = nn.ModuleList([
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size) if has_multimodal else None
            for patch_size in patch_sizes
        ])
        self.cross_attention = CrossAttentionFusion(embed_dim, num_heads)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Add normalization layers for fusion stability
        self.fusion_norm = nn.LayerNorm(self.embed_dim_total)
        self.fusion_head = nn.Linear(self.embed_dim_total, embed_dim)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize with improved variance
        for cls_token in self.cls_tokens:
            nn.init.trunc_normal_(cls_token, std=0.02)
        for pos_embed in self.pos_embeds:
            nn.init.trunc_normal_(pos_embed, std=0.02)
        self._init_weights()
    
    def _init_weights(self):
        # Better initialization of weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rgb, spectral=None):
        B = rgb.shape[0]
        multi_scale_features = self.patch_embed(rgb)
        
        scale_outputs = []
        for i, (features, pos_embed) in enumerate(zip(multi_scale_features, self.pos_embeds)):
            cls_tokens = self.cls_tokens[i].expand(B, -1, -1)
            features = torch.cat((cls_tokens, features), dim=1)
            pos_embed = pos_embed.to(features.device)
            features = features + pos_embed.expand(B, -1, -1)
            features = self.dropout(features)
            
            if spectral is not None and self.spectral_patch_embeds[i] is not None:
                spectral_features = self.spectral_patch_embeds[i](spectral.unsqueeze(1))
                spectral_features = spectral_features.flatten(2).transpose(1, 2)
                spectral_cls = torch.zeros(B, 1, self.embed_dim).to(spectral.device)
                spectral_features = torch.cat((spectral_cls, spectral_features), dim=1)
                logger.debug(f"Scale {i}: rgb_features shape={features.shape}, spectral_features shape={spectral_features.shape}")
                features = self.cross_attention(features, spectral_features)
            else:
                logger.debug(f"Scale {i}: rgb_features shape={features.shape}, No spectral features")
            
            # Apply all transformer layers with gradient checkpointing for efficiency if needed
            for layer in self.transformer_layers:
                features, _ = layer(features)
            
            features = self.norm(features)
            cls_output = features[:, 0]  # Extract CLS token output
            scale_outputs.append(cls_output)
        
        # Combine outputs from different scales with normalization for stability
        combined_features = torch.cat(scale_outputs, dim=-1)
        combined_features = self.fusion_norm(combined_features)  # Add normalization before fusion
        fused_features = self.fusion_head(combined_features)
        fused_features = self.head_norm(fused_features)  # Add normalization before final projection
        logits = self.head(fused_features)
        return logits

    def get_attention_weights(self, rgb, spectral=None):
        self.eval()
        attention_weights_all_scales = []
        with torch.no_grad():
            B = rgb.shape[0]
            multi_scale_features = self.patch_embed(rgb)
            
            for i, (features, pos_embed) in enumerate(zip(multi_scale_features, self.pos_embeds)):
                cls_tokens = self.cls_tokens[i].expand(B, -1, -1)
                features = torch.cat((cls_tokens, features), dim=1)
                pos_embed = pos_embed.to(features.device)
                features = features + pos_embed.expand(B, -1, -1)
                features = self.dropout(features)
                
                if spectral is not None and self.spectral_patch_embeds[i] is not None:
                    spectral_features = self.spectral_patch_embeds[i](spectral.unsqueeze(1))
                    spectral_features = spectral_features.flatten(2).transpose(1, 2)
                    spectral_cls = torch.zeros(B, 1, self.embed_dim).to(spectral.device)
                    spectral_features = torch.cat((spectral_cls, spectral_features), dim=1)
                    features = self.cross_attention(features, spectral_features)
                
                attention_weights = []
                for layer in self.transformer_layers:
                    features, attn_weight = layer(features)
                    attention_weights.append(attn_weight)
                
                attention_weights_all_scales.append(attention_weights)
            
            return attention_weights_all_scales

# Training Function with Enhanced Stability
# Training Function to Show Only Epoch Results
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name="HVT"):
    scaler = GradScaler(enabled=True)
    best_val_acc = 0.0
    best_model_path = f"./phase2_checkpoints/{model_name}_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience = 15
    patience_counter = 0
    mixup_alpha = 0.4
    use_cutmix = True
    bad_batch_count = 0
    max_bad_batches = 5
    
    # Normalization stats - use standard ImageNet stats
    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0].to(device)
            labels = batch[2].to(device)
            spectral = batch[1].to(device) if has_multimodal else None
            
            # Apply normalization only if data is not pre-normalized (range outside [-2.5, 2.5])
            if inputs.min() < -2.5 or inputs.max() > 2.5:
                inputs = (inputs - rgb_mean) / rgb_std
            
            if spectral is not None:
                spectral_min = spectral.min()
                spectral_max = spectral.max()
                if spectral_min < 0 or spectral_max > 1:
                    spectral = (spectral - spectral_min) / (spectral_max - spectral_min + 1e-8)
            
            # Apply data augmentation with probability
            if epoch > 0 and np.random.random() > 0.3:
                if np.random.random() > 0.5 and use_cutmix:
                    inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=mixup_alpha)
                else:
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
            else:
                labels_a, labels_b = labels, labels
                lam = 1.0
            
            optimizer.zero_grad()
            
            if epoch == 0:
                outputs = model(inputs, spectral)
                loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
                loss.backward()
            else:
                with autocast():
                    outputs = model(inputs, spectral)
                    loss = criterion(outputs, labels_a) * lam + criterion(outputs, labels_b) * (1 - lam)
                scaler.scale(loss).backward()
            
            if torch.isnan(loss) or torch.isinf(loss):
                bad_batch_count += 1
                if bad_batch_count >= max_bad_batches:
                    raise ValueError(f"Too many bad batches ({bad_batch_count}) with nan/inf loss")
                continue
            
            if epoch == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    bad_batch_count += 1
                    if bad_batch_count >= max_bad_batches:
                        raise ValueError(f"Too many bad batches ({bad_batch_count}) with nan/inf gradients")
                    continue
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if total == 0:
            continue
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total

        bad_batch_count = 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = batch[2].to(device)
                spectral = batch[1].to(device) if has_multimodal else None
                
                if inputs.min() < -2.5 or inputs.max() > 2.5:
                    inputs = (inputs - rgb_mean) / rgb_std
                if spectral is not None:
                    spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)
                
                outputs = model(inputs, spectral)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset) if total > 0 else float('inf')
        val_acc = 100 * correct / total if total > 0 else 0.0
        
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - {model_name}")
        logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        scheduler.step()

    return best_model_path

def evaluate_model(model, test_loader, criterion, device, class_names):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            labels = batch[2].to(device)
            spectral = batch[1].to(device) if has_multimodal else None

            if inputs.min() < -2.5 or inputs.max() > 2.5:
                inputs = (inputs - rgb_mean) / rgb_std
            if spectral is not None:
                spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)

            outputs = model(inputs, spectral)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    logger.info("Classification Report:\n" + report)
    logger.info("Confusion Matrix:\n" + np.array_str(cm))

    return test_accuracy, report, cm

def visualize_attention(model, test_loader, class_names, device, num_samples=5, save_path="./phase2_checkpoints"):
    """Placeholder for attention visualization"""
    logger.info("Visualizing attention...")
    # Replace with your actual attention visualization code
    pass

def efficiency_analysis(model, device, model_name="HVT"):
    logger.info(f"Performing Efficiency Analysis for {model_name}...")
    # Replace with your actual efficiency analysis code
    return 0.0, 0.0

# Main Execution (HVT Only)
if __name__ == "__main__":
    try:
        # Step 1: Initialize and Train HVT
        hvt_model = HierarchicalVisionTransformer(
            img_size=299, patch_sizes=[8, 16, 32], in_chans=3, embed_dim=384,
            num_heads=8, depth=12, num_classes=num_classes, dropout=0.15, stochastic_depth_prob=0.1  # Reduced stochastic depth
        ).to(device)
        
        criterion_hvt = FocalLoss(weight=class_weights, gamma=1.5)
        optimizer_hvt = optim.AdamW(hvt_model.parameters(), lr=5e-5, weight_decay=1e-2)  # Reduced learning rate for stability
        warmup_epochs = 10
        total_epochs = 25
        scheduler_hvt = LambdaLR(optimizer_hvt, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
        if total_epochs > warmup_epochs:
            scheduler_hvt = torch.optim.lr_scheduler.SequentialLR(
                optimizer_hvt,
                schedulers=[scheduler_hvt, CosineAnnealingLR(optimizer_hvt, T_max=total_epochs - warmup_epochs, eta_min=1e-6)],
                milestones=[warmup_epochs]
            )
        
        logger.info("Training HVT architecture...")
        best_model_path = train_model(
            hvt_model, train_loader, val_loader, criterion_hvt, optimizer_hvt, scheduler_hvt, num_epochs=total_epochs, device=device, model_name="HVT"
        )
        
        # Evaluate HVT on test set
        hvt_model.load_state_dict(torch.load(best_model_path))
        hvt_test_acc, hvt_report, hvt_cm = evaluate_model(hvt_model, test_loader, criterion_hvt, device, class_names)
        hvt_model_size, hvt_inference_time = efficiency_analysis(hvt_model, device=device, model_name="HVT")

        # Step 2: Visualize HVT Attention
        visualize_attention(hvt_model, test_loader, class_names, device, num_samples=5, save_path="./phase2_checkpoints")

        # Step 3: Save Results
        phase2_results = {
            'hvt': {
                'train_losses': hvt_train_losses,
                'val_losses': hvt_val_losses,
                'train_accs': hvt_train_accs,
                'val_accs': hvt_val_accs,
                'best_val_acc': hvt_best_val_acc,
                'test_acc': hvt_test_acc,
                'report': hvt_report,
                'confusion_matrix': hvt_cm,
                'model_size': hvt_model_size,
                'inference_time': hvt_inference_time
            }
        }
        torch.save(phase2_results, "./phase2_checkpoints/phase2_results.pth")
        
        logger.info("Phase 2 completed: HVT architecture trained and evaluated.")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise