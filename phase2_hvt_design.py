import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

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

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Compute class weights for imbalance
class_counts = np.array([class_distribution[name][0] for name in class_names])
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = torch.FloatTensor(class_weights).to(device)
logger.info(f"Class weights: {class_weights.tolist()}")

# Baseline Inception V3 Model
def get_inception_v3_model(num_classes, pretrained=True):
    model = models.inception_v3(pretrained=pretrained)
    model.aux_logits = False  # Disable auxiliary logits for simplicity
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Freeze all layers except the last 20
    for name, param in model.named_parameters():
        if "Mixed_7" not in name and "fc" not in name:
            param.requires_grad = False
    
    return model.to(device)

# Hierarchical Vision Transformer (HVT) Design
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[16, 32], in_chans=3, embed_dim=768):
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
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, rgb_features, spectral_features):
        if spectral_features is None:
            return rgb_features
        fused, _ = self.cross_attention(rgb_features, spectral_features, spectral_features)
        return self.norm(fused + rgb_features)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))
        return x

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, img_size=299, patch_sizes=[16, 32], in_chans=3, embed_dim=768, 
                 num_heads=12, depth=16, num_classes=7, dropout=0.1):
        super().__init__()
        self.patch_embed = MultiScalePatchEmbed(img_size, patch_sizes, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.embed_dim_total = embed_dim * len(patch_sizes)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim_total))
        self.pos_embed = nn.Parameter(torch.zeros(1, max(self.num_patches) + 1, self.embed_dim_total))
        self.dropout = nn.Dropout(dropout)
        
        self.spectral_embed = nn.Linear(1, embed_dim) if has_multimodal else None
        self.cross_attention = CrossAttentionFusion(self.embed_dim_total, num_heads)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim_total, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim_total)
        self.head = nn.Linear(self.embed_dim_total, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, rgb, spectral=None):
        B = rgb.shape[0]
        multi_scale_features = self.patch_embed(rgb)
        
        combined_features = []
        for i, features in enumerate(multi_scale_features):
            pos = self.pos_embed[:, :self.num_patches[i] + 1, i * self.embed_dim:(i + 1) * self.embed_dim]
            cls_tokens = self.cls_token[:, :, i * self.embed_dim:(i + 1) * self.embed_dim].expand(B, -1, -1)
            features = torch.cat((cls_tokens, features), dim=1)
            features = features + pos
            combined_features.append(features)
        
        combined_features = torch.cat(combined_features, dim=-1)
        combined_features = self.dropout(combined_features)
        
        if spectral is not None and self.spectral_embed is not None:
            spectral = spectral.unsqueeze(-1)
            spectral_features = self.spectral_embed(spectral)
            spectral_features = spectral_features.expand(-1, combined_features.shape[1], -1)
            combined_features = self.cross_attention(combined_features, spectral_features)
        
        for layer in self.transformer_layers:
            combined_features = layer(combined_features)
        
        combined_features = self.norm(combined_features)
        cls_features = combined_features[:, 0]
        logits = self.head(cls_features)
        return logits

    def get_attention_weights(self, rgb, spectral=None):
        self.eval()
        with torch.no_grad():
            B = rgb.shape[0]
            multi_scale_features = self.patch_embed(rgb)
            
            combined_features = []
            for i, features in enumerate(multi_scale_features):
                pos = self.pos_embed[:, :self.num_patches[i] + 1, i * self.embed_dim:(i + 1) * self.embed_dim]
                cls_tokens = self.cls_token[:, :, i * self.embed_dim:(i + 1) * self.embed_dim].expand(B, -1, -1)
                features = torch.cat((cls_tokens, features), dim=1)
                features = features + pos
                combined_features.append(features)
            
            combined_features = torch.cat(combined_features, dim=-1)
            combined_features = self.dropout(combined_features)
            
            if spectral is not None and self.spectral_embed is not None:
                spectral = spectral.unsqueeze(-1)
                spectral_features = self.spectral_embed(spectral)
                spectral_features = spectral_features.expand(-1, combined_features.shape[1], -1)
                combined_features = self.cross_attention(combined_features, spectral_features)
            
            attention_weights = []
            for layer in self.transformer_layers:
                attn_output, attn_weights = layer.attention(combined_features, combined_features, combined_features)
                attention_weights.append(attn_weights)
                combined_features = layer.norm1(combined_features + layer.dropout(attn_output))
                mlp_output = layer.mlp(combined_features)
                combined_features = layer.norm2(combined_features + layer.dropout(mlp_output))
            
            return attention_weights

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="InceptionV3"):
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_path = f"./phase2_checkpoints/{model_name}_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[2].to(device)
            spectral = batch[1].to(device) if has_multimodal else None
            
            optimizer.zero_grad()
            with autocast():
                if isinstance(model, HierarchicalVisionTransformer):
                    outputs = model(inputs, spectral)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
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
                
                with autocast():
                    if isinstance(model, HierarchicalVisionTransformer):
                        outputs = model(inputs, spectral)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - {model_name}")
        logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Acc: {best_val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs, best_val_acc

# Evaluation Function
def evaluate_model(model, test_loader, criterion, device, class_names, model_name="InceptionV3"):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            labels = batch[2].to(device)
            spectral = batch[1].to(device) if has_multimodal else None
            
            with autocast():
                if isinstance(model, HierarchicalVisionTransformer):
                    outputs = model(inputs, spectral)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100 * correct / total
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    logger.info(f"\n{model_name} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logger.info("\nClassification Report:")
    for class_name, metrics in report.items():
        if class_name in class_names:
            logger.info(f"{class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./phase2_checkpoints/{model_name}_confusion_matrix.png")
    plt.close()
    
    return test_acc, report, cm

# Efficiency Analysis
def efficiency_analysis(model, input_size=(3, 299, 299), device='cuda', model_name="InceptionV3"):
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)
    spectral_input = torch.randn(1, 299, 299).to(device) if has_multimodal else None
    
    # Model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)  # MB
    
    # Inference time
    num_trials = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            if isinstance(model, HierarchicalVisionTransformer):
                _ = model(dummy_input, spectral_input)
            else:
                _ = model(dummy_input)
    avg_time = (time.time() - start_time) / num_trials * 1000  # ms
    
    logger.info(f"\n{model_name} Efficiency Analysis:")
    logger.info(f"Model Size: {model_size:.2f} MB")
    logger.info(f"Average Inference Time ({device}): {avg_time:.2f} ms/sample")
    
    return model_size, avg_time

# Visualize Attention for HVT
def visualize_attention(model, dataloader, class_names, device, num_samples=5, save_path="./phase2_checkpoints"):
    model.eval()
    batch = next(iter(dataloader))
    images = batch[0][:num_samples].to(device)
    spectral = batch[1][:num_samples].to(device) if has_multimodal else None
    labels = batch[2][:num_samples].numpy()
    
    attention_weights = model.get_attention_weights(images, spectral)
    
    # Aggregate attention across layers and heads
    avg_attention = torch.stack(attention_weights).mean(dim=0)  # Average across layers
    avg_attention = avg_attention.mean(dim=1)  # Average across heads
    
    # Visualize for each sample
    os.makedirs(save_path, exist_ok=True)
    for i in range(num_samples):
        attn_map = avg_attention[i, 1:]  # Exclude cls token
        attn_map = attn_map.view(int(np.sqrt(attn_map.shape[0])), -1)
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0), 
            size=(299, 299), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        plt.figure(figsize=(10, 5))
        
        # Original Image
        plt.subplot(1, 2, 1)
        img = images[i].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"Class: {class_names[labels[i]]}")
        plt.axis('off')
        
        # Attention Heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.imshow(attn_map.cpu().numpy(), cmap='jet', alpha=0.5)
        plt.title("Attention Heatmap")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"hvt_attention_sample_{i}.png"))
        plt.close()

# Main Execution
if __name__ == "__main__":
    # Step 1: Train and Evaluate Inception V3 Baseline
    inception_model = get_inception_v3_model(num_classes, pretrained=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, inception_model.parameters()), lr=0.001)
    
    logger.info("Training Inception V3...")
    train_losses, val_losses, train_accs, val_accs, best_val_acc = train_model(
        inception_model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device, model_name="InceptionV3"
    )
    
    # Load best model
    inception_model.load_state_dict(torch.load("./phase2_checkpoints/InceptionV3_best.pth"))
    test_acc, report, cm = evaluate_model(inception_model, test_loader, criterion, device, class_names, model_name="InceptionV3")
    inception_model_size, inception_inference_time = efficiency_analysis(inception_model, device=device, model_name="InceptionV3")
    
    # Step 2: Initialize and Validate HVT
    hvt_model = HierarchicalVisionTransformer(
        img_size=299, patch_sizes=[16, 32], in_chans=3, embed_dim=768, 
        num_heads=12, depth=16, num_classes=num_classes, dropout=0.1
    ).to(device)
    
    # Preliminary training to validate architecture
    criterion_hvt = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_hvt = optim.Adam(hvt_model.parameters(), lr=0.0001)
    
    logger.info("Validating HVT architecture with preliminary training...")
    hvt_train_losses, hvt_val_losses, hvt_train_accs, hvt_val_accs, hvt_best_val_acc = train_model(
        hvt_model, train_loader, val_loader, criterion_hvt, optimizer_hvt, num_epochs=5, device=device, model_name="HVT"
    )
    
    # Evaluate HVT on test set
    hvt_model.load_state_dict(torch.load("./phase2_checkpoints/HVT_best.pth"))
    hvt_test_acc, hvt_report, hvt_cm = evaluate_model(hvt_model, test_loader, criterion_hvt, device, class_names, model_name="HVT")
    hvt_model_size, hvt_inference_time = efficiency_analysis(hvt_model, device=device, model_name="HVT")
    
    # Step 3: Visualize HVT Attention
    visualize_attention(hvt_model, test_loader, class_names, device, num_samples=5, save_path="./phase2_checkpoints")
    
    # Step 4: Save Results
    phase2_results = {
        'inception': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'report': report,
            'confusion_matrix': cm,
            'model_size': inception_model_size,
            'inference_time': inception_inference_time
        },
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
    
    logger.info("Phase 2 completed: Inception V3 baseline replicated and HVT architecture validated.")