# phase2_hvt_design.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Set Up the Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

output_dir = 'phase2_results'
os.makedirs(output_dir, exist_ok=True)

# Step 2: Load Preprocessed Data from Phase 1
try:
    checkpoint = torch.load(os.path.join('./phase1_checkpoints', 'phase1_preprocessed_data.pth'))
except FileNotFoundError as e:
    logger.error(f"Preprocessed data not found: {e}")
    raise FileNotFoundError("Run Phase 1 first to generate preprocessed data.")

train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']
class_names = checkpoint['class_names']
has_multimodal = checkpoint['has_multimodal']

if len(class_names) > 7:
    class_names = class_names[:7]
num_classes = len(class_names)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Training set size: {len(train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")
logger.info(f"Test set size: {len(test_dataset)}")

# Step 3: Define HVT Components
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

    def forward(self, x: torch.Tensor):
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
        self.attn_weights = None  # For visualization

    def forward(self, rgb_features: torch.Tensor, spectral_features: torch.Tensor):
        B, N, C = rgb_features.shape
        if spectral_features is None:
            return rgb_features

        q = self.query(rgb_features).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(spectral_features).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.value(spectral_features).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()  # Save for visualization
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out(out)

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 299, patch_sizes: list = [16, 8, 4], embed_dims: list = [768, 384, 192], num_heads: int = 12, num_layers: int = 12, has_multimodal: bool = False):
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
        
        if has_multimodal:
            self.spectral_patch_embed = nn.Sequential(
                nn.Conv2d(1, self.embed_dim_total, kernel_size=patch_sizes[0], stride=patch_sizes[0]),
                nn.BatchNorm2d(self.embed_dim_total),
                nn.ReLU(inplace=True)
            )
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim_total))
            self.fusion = CrossAttentionFusion(self.embed_dim_total, num_heads)

    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor = None):
        B = rgb.shape[0]
        
        multi_scale_features = self.patch_embed(rgb)
        combined_features = torch.cat([f.flatten(2).transpose(1, 2) for f in multi_scale_features], dim=-1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        combined_features = torch.cat((cls_tokens, combined_features), dim=1)
        combined_features = combined_features + self.pos_embed[:, :combined_features.size(1), :]
        
        for layer in self.transformer_layers:
            combined_features = layer(combined_features)
        
        if self.has_multimodal and spectral is not None:
            if len(spectral.shape) == 3:
                spectral = spectral.unsqueeze(1)
            spectral_features = self.spectral_patch_embed(spectral)
            spectral_features = spectral_features.flatten(2).transpose(1, 2)
            
            spectral_cls_tokens = torch.zeros(B, 1, self.embed_dim_total, device=spectral.device)
            spectral_features = torch.cat((spectral_cls_tokens, spectral_features), dim=1)
            spectral_features = spectral_features + self.spectral_pos_embed[:, :spectral_features.size(1), :]
            
            combined_features = self.fusion(combined_features, spectral_features)
        
        cls_output = combined_features[:, 0]
        cls_output = self.norm(cls_output)
        return self.head(cls_output)

    def get_attention_weights(self):
        if self.has_multimodal:
            return self.fusion.attn_weights
        return None

# Step 4: Define Inception V3 Baseline
def get_inception_v3(num_classes):
    model = models.inception_v3(weights='IMAGENET1K_V1')
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

# Step 5: Preliminary Efficiency Analysis
def compute_model_size(model):
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_size / 1e6  # Size in MB (approximate)

def measure_inference_time(model, input_rgb, input_spectral=None, device='cuda', num_trials=100):
    model.eval()
    input_rgb = input_rgb.to(device)
    if input_spectral is not None:
        input_spectral = input_spectral.to(device)
    model = model.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_rgb, input_spectral)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(input_rgb, input_spectral)
    avg_time = (time.time() - start_time) / num_trials
    return avg_time * 1000  # Convert to milliseconds

def efficiency_analysis(model, dataloader, has_multimodal, device='cuda'):
    # Model size
    model_size = compute_model_size(model)
    logger.info(f"Model size: {model_size:.2f}M parameters")

    # Prepare a sample batch
    batch = next(iter(dataloader))
    rgb = batch[0] if has_multimodal else batch[0]
    spectral = batch[1] if has_multimodal else None

    # Measure inference time on GPU
    if device.startswith('cuda'):
        gpu_time = measure_inference_time(model, rgb, spectral, device='cuda')
        logger.info(f"Average inference time on GPU: {gpu_time:.2f} ms")

    # Measure inference time on CPU
    cpu_time = measure_inference_time(model, rgb, spectral, device='cpu')
    logger.info(f"Average inference time on CPU: {cpu_time:.2f} ms")

    # Simulate low-power device (e.g., Raspberry Pi) by scaling CPU time
    # Assuming Raspberry Pi 4 is ~10x slower than a modern CPU (empirical approximation)
    rpi_time = cpu_time * 10
    logger.info(f"Estimated inference time on Raspberry Pi (simulated): {rpi_time:.2f} ms")

    return {
        'model_size': model_size,
        'gpu_inference_time': gpu_time if device.startswith('cuda') else None,
        'cpu_inference_time': cpu_time,
        'rpi_inference_time': rpi_time
    }

# Step 6: Train and Evaluate Inception V3 Baseline
inception_model = get_inception_v3(num_classes).to(device)
class_counts = [checkpoint['original_stats']['class_distribution'][cls][0] for cls in class_names]
class_weights = (1.0 / torch.tensor(class_counts, dtype=torch.float)) / torch.sum(1.0 / torch.tensor(class_counts))
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(inception_model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
scaler = GradScaler() if torch.cuda.is_available() else None

num_epochs = 30
best_val_acc = 0.0
best_model_path = os.path.join(output_dir, 'best_inception_v3.pth')
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(num_epochs):
    inception_model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, spectral, labels in train_loader if has_multimodal else zip(train_loader, [None] * len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast() if scaler else torch.no_grad():
            outputs = inception_model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
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

    inception_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, spectral, labels in val_loader if has_multimodal else zip(val_loader, [None] * len(val_loader)):
            images, labels = images.to(device), labels.to(device)
            with autocast() if scaler else torch.no_grad():
                outputs = inception_model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
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
            'model_state_dict': inception_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }, best_model_path)

# Step 7: Evaluate Inception V3 on Test Set
inception_model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
inception_model.eval()
test_correct, test_total, test_loss = 0, 0, 0.0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, spectral, labels in test_loader if has_multimodal else zip(test_loader, [None] * len(test_loader)):
        images, labels = images.to(device), labels.to(device)
        with autocast() if scaler else torch.no_grad():
            outputs = inception_model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
logger.info(f"Inception V3 Test Accuracy: {test_accuracy:.2f}%")

# Step 8: Efficiency Analysis for Inception V3
inception_efficiency = efficiency_analysis(inception_model, test_loader, has_multimodal)

# Step 9: Initialize and Analyze HVT
hvt_model = HierarchicalVisionTransformer(
    num_classes=num_classes,
    img_size=299,
    patch_sizes=[16, 8, 4],
    embed_dims=[768, 384, 192],
    num_heads=12,
    num_layers=12,
    has_multimodal=has_multimodal
).to(device)

hvt_efficiency = efficiency_analysis(hvt_model, test_loader, has_multimodal)

# Step 10: Visualize Attention (Sample)
def visualize_attention(model, dataloader, has_multimodal, save_path="attention_map.png"):
    model.eval()
    batch = next(iter(dataloader))
    rgb = batch[0].to(device)
    spectral = batch[1].to(device) if has_multimodal else None

    with torch.no_grad():
        _ = model(rgb, spectral)
        attn_weights = model.get_attention_weights()
        if attn_weights is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn_weights[0, 0].cpu().numpy(), cmap='hot')
            plt.title("Attention Map (Sample)")
            plt.savefig(os.path.join(output_dir, save_path))
            plt.close()
            logger.info(f"Attention map saved to {save_path}")

visualize_attention(hvt_model, test_loader, has_multimodal)

# Step 11: Save HVT Architecture and Efficiency Report
torch.save({
    'hvt_state_dict': hvt_model.state_dict(),
    'inception_test_accuracy': test_accuracy,
    'hvt_efficiency': hvt_efficiency,
    'inception_efficiency': inception_efficiency
}, os.path.join(output_dir, 'phase2_hvt_design.pth'))

# Save efficiency report
with open(os.path.join(output_dir, 'efficiency_report.txt'), 'w') as f:
    f.write("Phase 2 Efficiency Analysis\n\n")
    f.write("Inception V3:\n")
    f.write(f"Model size: {inception_efficiency['model_size']:.2f}M parameters\n")
    f.write(f"GPU inference time: {inception_efficiency['gpu_inference_time']:.2f} ms\n")
    f.write(f"CPU inference time: {inception_efficiency['cpu_inference_time']:.2f} ms\n")
    f.write(f"Raspberry Pi (simulated) inference time: {inception_efficiency['rpi_inference_time']:.2f} ms\n\n")
    f.write("Hierarchical Vision Transformer (HVT):\n")
    f.write(f"Model size: {hvt_efficiency['model_size']:.2f}M parameters\n")
    f.write(f"GPU inference time: {hvt_efficiency['gpu_inference_time']:.2f} ms\n")
    f.write(f"CPU inference time: {hvt_efficiency['cpu_inference_time']:.2f} ms\n")
    f.write(f"Raspberry Pi (simulated) inference time: {hvt_efficiency['rpi_inference_time']:.2f} ms\n")

logger.info("Phase 2 completed: HVT designed and preliminary efficiency analysis conducted.")