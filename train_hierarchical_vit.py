# train_hierarchical_vit.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DataParallel
from torch.utils.checkpoint import checkpoint_sequential
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import logging
from torchvision import transforms
from typing import Tuple, Optional, Any
from dataset_utils import CottonLeafDataset

# Set environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# Hierarchical Vision Transformer (HVT) Components
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[16, 8, 4], in_channels=3, embed_dims=[1024, 512, 256]):  # Increased embed_dims
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
    def __init__(self, num_classes: int, img_size: int = 299, patch_sizes: list = [16, 8, 4], embed_dims: list = [1024, 512, 256], num_heads: int = 16, num_layers: int = 16, has_multimodal: bool = False, spectral_dim: int = 299):  # Increased num_layers
        super().__init__()
        self.has_multimodal = has_multimodal
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, embed_dims=embed_dims)
        self.embed_dim_total = sum(embed_dims)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim_total, nhead=num_heads, dim_feedforward=self.embed_dim_total*4, dropout=0.1, batch_first=True)
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
        
        cls_output = combined_features[:, 0]
        cls_output = self.norm(cls_output)
        return self.head(cls_output)

# Step 1: Load Preprocessed Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    logger.info(f"Using {num_gpus} H100 GPUs with DataParallel.")
else:
    logger.info("Using 1 H100 GPU.")

output_dir = 'phase3_hvt_results_improved'
os.makedirs(output_dir, exist_ok=True)

try:
    checkpoint = torch.load(os.path.join('./phase1_checkpoints', 'phase1_preprocessed_data.pth'))
except FileNotFoundError as e:
    logger.error(f"Preprocessed data not found: {e}")
    raise FileNotFoundError("Run Phase 1 first to generate preprocessed data.")

# Use existing dataset objects from checkpoint
train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']
class_names = checkpoint['class_names']
has_multimodal = checkpoint['has_multimodal']

if len(class_names) > 7:
    class_names = class_names[:7]
num_classes = len(class_names)

# Define data augmentation transform with more aggressive augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Reintroduced
    transforms.RandomRotation(10),    # Increased to 10 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
])

# Wrap training dataset with augmentation
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
val_dataset = AugmentedDataset(val_dataset, transform=None)
test_dataset = AugmentedDataset(test_dataset, transform=None)

batch_size = 16
effective_batch_size = 32
accumulation_steps = effective_batch_size // batch_size
train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

logger.info(f"Training set size: {len(augmented_train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")
logger.info(f"Test set size: {len(test_dataset)}")

# Collect labels by iterating over the dataset
labels = []
for _, _, label in augmented_train_dataset:
    labels.append(label)
labels = torch.tensor(labels, device=device)
class_counts = [torch.sum(labels == i).item() for i in range(num_classes)]
class_weights = torch.tensor([1.0 / count if count > 0 else 1.0 for count in class_counts], device=device)
class_weights = class_weights / class_weights.sum() * num_classes
logger.info(f"Class weights: {class_weights}")

# Step 2: Initialize HVT Model
model = HierarchicalVisionTransformer(
    num_classes=num_classes,
    img_size=299,
    patch_sizes=[16, 8, 4],
    embed_dims=[1024, 512, 256],
    num_heads=16,
    num_layers=16,
    has_multimodal=has_multimodal,
    spectral_dim=299
)
if num_gpus > 1:
    model = DataParallel(model)
model = model.to(device)
logger.info(f"HVT model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

# Step 3: Define Loss, Optimizer, and Scheduler
criterion = FocalLoss(gamma=2.0, alpha=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)  # Lowered learning rate, increased weight decay
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # Added ReduceLROnPlateau
scaler = GradScaler()

# Step 4: Training Loop with Gradient Accumulation, Checkpointing, and Mixed Precision
num_epochs = 80
best_val_acc = 0.0
best_model_path = os.path.join(output_dir, 'best_hvt.pth')
patience = 10
trigger_times = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for i, (rgb, spectral, labels) in enumerate(train_loader):
        rgb, labels = rgb.to(device), labels.to(device)
        
        if spectral is not None and has_multimodal:
            spectral = spectral.to(device)
        else:
            spectral = None
            
        if i % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(rgb, spectral)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_accuracy)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for rgb, spectral, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            
            if spectral is not None and has_multimodal:
                spectral = spectral.to(device)
            else:
                spectral = None
                
            with autocast():
                outputs = model(rgb, spectral)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_accuracy)

    scheduler.step(val_loss)
    
    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        model_state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }, best_model_path)
        logger.info(f"Best model saved at epoch {epoch+1} with val_acc: {val_accuracy:.2f}%")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

training_time = time.time() - start_time
logger.info(f"Training completed in {training_time/60:.2f} minutes")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

logger.info("Loading best model for evaluation on test set...")
checkpoint = torch.load(best_model_path)
if num_gpus > 1:
    model.module.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint['model_state_dict'])
logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation accuracy: {checkpoint['val_accuracy']:.2f}%")

model.eval()
test_correct, test_total, test_loss = 0, 0, 0.0
all_preds, all_labels = [], []

with torch.no_grad():
    for rgb, spectral, labels in test_loader:
        rgb, labels = rgb.to(device), labels.to(device)
        
        if spectral is not None and has_multimodal:
            spectral = spectral.to(device)
        else:
            spectral = None
            
        with autocast():
            outputs = model(rgb, spectral)
            loss = criterion(outputs, labels)
            
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100 * test_correct / test_total
logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

logger.info("\nPer-class accuracy:")
class_correct = [0] * num_classes
class_total = [0] * num_classes

for pred, label in zip(all_preds, all_labels):
    class_total[label] += 1
    if pred == label:
        class_correct[label] += 1

for i in range(num_classes):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        logger.info(f"{class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

logger.info("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"HVT Results\n\n")
    f.write(f"Training time: {training_time/60:.2f} minutes\n")
    f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Test accuracy: {test_accuracy:.2f}%\n")
    f.write("\nPer-class accuracy:\n")
    for i in range(num_classes):
        if class_total[i] > 0:
            f.write(f"{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})\n")
    f.write("\nClassification Report:\n")
    for cls in class_names:
        cls_idx = class_names.index(cls)
        cls_metrics = report[cls]
        f.write(f"{cls}:\n")
        f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {cls_metrics['recall']:.4f}\n")
        f.write(f"  F1-score: {cls_metrics['f1-score']:.4f}\n")
        f.write(f"  Support: {cls_metrics['support']}\n\n")

torch.save({
    'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
    'test_accuracy': test_accuracy,
    'class_report': report,
    'confusion_matrix': cm,
    'training_history': history
}, os.path.join(output_dir, 'hvt_model.pth'))

logger.info(f"Phase 3 complete! HVT model trained and evaluated with test accuracy: {test_accuracy:.2f}%")
logger.info(f"Results saved to {output_dir}/")