# train_hvt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from torchvision.models import ViT_B_16_Weights
from torch.cuda.amp import GradScaler, autocast
import logging
from dataset_utils import CottonLeafDataset
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hierarchical Vision Transformer (HVT) Components
class MultiScalePatchEmbed(nn.Module):
    """Multi-scale patch embedding layer for hierarchical processing."""
    def __init__(self, img_size=299, patch_sizes=[16, 8], in_channels=3, embed_dims=[768, 384]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList()
        
        for patch_size, embed_dim in zip(patch_sizes, embed_dims):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ))
        self.pos_embed = nn.Parameter(torch.zeros(1, sum(embed_dims), img_size // min(patch_sizes), img_size // min(patch_sizes)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = []
        for layer in self.layers:
            features.append(layer(x))
        return tuple(features)

class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for multimodal fusion (RGB + spectral)."""
    def __init__(self, dim: int, num_heads: int = 8):
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
    """Hierarchical Vision Transformer with multi-scale and multimodal fusion."""
    def __init__(self, num_classes: int, img_size: int = 299, patch_sizes: list = [16, 8], embed_dims: list = [768, 384], num_heads: int = 12, num_layers: int = 12, has_multimodal: bool = False):
        super().__init__()
        self.has_multimodal = has_multimodal
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, embed_dims=embed_dims)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=sum(embed_dims), nhead=num_heads, dim_feedforward=sum(embed_dims)*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fusion = CrossAttentionFusion(sum(embed_dims)) if has_multimodal else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, sum(embed_dims)))
        self.pos_embed = nn.Parameter(torch.zeros(1, sum(embed_dims), img_size // min(patch_sizes), img_size // min(patch_sizes)))
        self.norm = nn.LayerNorm(sum(embed_dims))
        self.head = nn.Linear(sum(embed_dims), num_classes)

    def forward(self, rgb: torch.Tensor, spectral: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-scale patch embedding
        multi_scale_features = self.patch_embed(rgb)  # List of tensors [B, C, H, W]
        combined_features = torch.cat([f.flatten(2).transpose(1, 2) for f in multi_scale_features], dim=-1)

        # Add cls token and positional embedding
        B = rgb.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        combined_features = torch.cat((cls_tokens, combined_features), dim=1)
        combined_features += self.pos_embed[:, :, :combined_features.size(1), :].expand(B, -1, -1, -1).flatten(2).transpose(1, 2)

        # Transformer encoding
        for layer in self.transformer_layers:
            combined_features = layer(combined_features)

        # Fusion with spectral data if available
        if self.has_multimodal and spectral is not None:
            spectral_flat = spectral.unsqueeze(1).expand(-1, combined_features.size(1), -1)  # Match sequence length
            combined_features = self.fusion(combined_features, spectral_flat)

        # Classification head
        cls_output = combined_features[:, 0]
        cls_output = self.norm(cls_output)
        return self.head(cls_output)

# Step 1: Load Preprocessed Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

output_dir = 'phase3_hvt_results'
os.makedirs(output_dir, exist_ok=True)

try:
    checkpoint = torch.load(os.path.join('./phase1_checkpoints', 'phase1_preprocessed_data.pth'))
except FileNotFoundError as e:
    logger.error(f"Preprocessed data not found: {e}")
    raise FileNotFoundError("Run Phase 1 first to generate preprocessed data.")

combined_train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']
class_names = checkpoint['class_names']
has_multimodal = checkpoint['has_multimodal']

# Adjust class names to 7 classes
if len(class_names) > 7:
    class_names = class_names[:-1]
num_classes = len(class_names)

# Recreate DataLoaders
train_loader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Training set size: {len(combined_train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")
logger.info(f"Test set size: {len(test_dataset)}")

# Step 2: Initialize HVT Model
model = HierarchicalVisionTransformer(
    num_classes=num_classes,
    img_size=299,
    patch_sizes=[16, 8],
    embed_dims=[768, 384],
    num_heads=12,
    num_layers=12,
    has_multimodal=has_multimodal
).to(device)
logger.info(f"HVT model initialized with {sum(p.numel() for p in model.parameters() / 1e6):.2f}M parameters.")

# Step 3: Define Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
scaler = GradScaler() if torch.cuda.is_available() else None

# Step 4: Training Loop with Early Stopping and Mixed Precision
num_epochs = 50
best_val_acc = 0.0
best_model_path = os.path.join(output_dir, 'best_hvt.pth')
patience = 10
trigger_times = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for rgb, spectral, labels in train_loader:
        rgb, labels = rgb.to(device), labels.to(device)
        spectral = spectral.to(device) if spectral is not None else None
        optimizer.zero_grad()

        with autocast() if scaler else torch.no_grad():
            outputs = model(rgb, spectral)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_accuracy)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for rgb, spectral, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            spectral = spectral.to(device) if spectral is not None else None
            with autocast() if scaler else torch.no_grad():
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

    scheduler.step()
    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
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

# Plot training history
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

# Step 5: Evaluate on Test Set
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation accuracy: {checkpoint['val_accuracy']:.2f}%")

model.eval()
test_correct, test_total, test_loss = 0, 0, 0.0
all_preds, all_labels = [], []

with torch.no_grad():
    for rgb, spectral, labels in test_loader:
        rgb, labels = rgb.to(device), labels.to(device)
        spectral = spectral.to(device) if spectral is not None else None
        with autocast() if scaler else torch.no_grad():
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

# Detailed Evaluation
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

# Save results
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
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    for cls in class_names:
        f.write(f"{cls}:\n{report[str(class_names.index(cls))]}\n")

torch.save({
    'model_state_dict': model.state_dict(),
    'test_accuracy': test_accuracy,
    'class_report': report,
    'confusion_matrix': cm,
    'training_history': history
}, os.path.join(output_dir, 'hvt_model.pth'))

logger.info(f"Phase 3 complete! HVT model trained and evaluated with test accuracy: {test_accuracy:.2f}%")
logger.info(f"Results saved to {output_dir}/")