import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# Assuming Step 1 (data loading) is already defined and working
# Variables like train_loader, val_loader, test_loader, device, has_multimodal, etc., are set up

# Step 2: Improved Hierarchical Vision Transformer with Multimodal Support
class MultiModalFusion(nn.Module):
    def __init__(self, rgb_dim, spectral_dim, fusion_dim, num_heads=8):
        super(MultiModalFusion, self).__init__()
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        self.spectral_proj = nn.Linear(spectral_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
    def forward(self, rgb_features, spectral_features):
        rgb_proj = self.rgb_proj(rgb_features)
        spectral_proj = self.spectral_proj(spectral_features)
        attended_features, _ = self.cross_attention(query=rgb_proj, key=spectral_proj, value=spectral_proj)
        if torch.any(torch.isnan(attended_features)) or torch.any(torch.isinf(attended_features)):
            print("NaN/Inf in cross_attention output")
            attended_features = torch.nan_to_num(attended_features, nan=0.0)
        x = self.norm1(rgb_proj + attended_features)
        ff_output = self.feed_forward(x)
        if torch.any(torch.isnan(ff_output)) or torch.any(torch.isinf(ff_output)):
            print("NaN/Inf in feed_forward output")
            ff_output = torch.nan_to_num(ff_output, nan=0.0)
        fused_features = self.norm2(x + ff_output)
        return fused_features

class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, 
                 image_size=299, 
                 patch_size=13, 
                 num_classes=7, 
                 dim=768, 
                 depths=[2, 2, 6], 
                 heads=[4, 8, 16], 
                 mlp_ratio=4.0,
                 multimodal_support=False,
                 spectral_channels=0,
                 pretrained=False):
        super(HierarchicalVisionTransformer, self).__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.dim = dim
        self.multimodal_support = multimodal_support
        self.use_multimodal = multimodal_support and spectral_channels > 0

        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        scales = [1, 2, 4]
        self.stage_dims = [dim // 3, dim // 2, dim]

        for i, scale in enumerate(scales):
            stage_dim = self.stage_dims[i]
            self.patch_embeds.append(nn.Conv2d(3, stage_dim, kernel_size=patch_size, stride=patch_size))
            downsampled_size = image_size // scale
            num_patches = (downsampled_size // patch_size) ** 2
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, stage_dim))
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embeds.append(pos_embed)

        if self.multimodal_support:
            self.spectral_embed = nn.Linear(spectral_channels, self.stage_dims[-1])
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, 1, self.stage_dims[-1]))
            nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)
            self.modal_fusion = MultiModalFusion(
                rgb_dim=self.stage_dims[-1],
                spectral_dim=self.stage_dims[-1],
                fusion_dim=self.stage_dims[-1],
                num_heads=heads[-1]
            )

        self.layers = nn.ModuleList()
        for i, (depth, head) in enumerate(zip(depths, heads)):
            stage_dim = self.stage_dims[i]
            encoder_layers = []
            for _ in range(depth):
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=stage_dim,
                    nhead=head,
                    dim_feedforward=int(stage_dim * mlp_ratio),
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True
                )
                encoder_layers.append(encoder_layer)
            self.layers.append(nn.ModuleList(encoder_layers))

        self.proj_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(len(scales) - 1):
            current_dim = self.stage_dims[i]
            next_dim = self.stage_dims[i + 1]
            current_scale = scales[i]
            next_scale = scales[i + 1]
            current_img_size = image_size // current_scale
            next_img_size = image_size // next_scale
            current_patches = (current_img_size // patch_size) ** 2
            next_patches = (next_img_size // patch_size) ** 2
            self.proj_layers.append(nn.Linear(current_dim, next_dim))
            self.pool_layers.append(nn.AdaptiveAvgPool1d(next_patches))

        self.cross_attentions = nn.ModuleList()
        for i in range(len(scales) - 1):
            next_dim = self.stage_dims[i + 1]
            self.cross_attentions.append(
                nn.MultiheadAttention(embed_dim=next_dim, num_heads=heads[i+1], batch_first=True)
            )

        self.norm = nn.LayerNorm(self.stage_dims[-1])
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(self.stage_dims[-1], num_classes)
        self.aux_classifiers = nn.ModuleList([nn.Linear(dim, num_classes) for dim in self.stage_dims])
        
        if not pretrained:
            self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, spectral_data=None):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print("NaN/Inf detected in input")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        features = []
        for i, (patch_embed, pos_embed) in enumerate(zip(self.patch_embeds, self.pos_embeds)):
            scale_factor = 2 ** i
            if scale_factor > 1:
                x_scaled = F.interpolate(x, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            x_patched = patch_embed(x_scaled).flatten(2).transpose(1, 2)
            x_patched = x_patched + pos_embed
            features.append(x_patched)

        stage_outputs = []
        aux_outputs = []
        for i, stage_layers in enumerate(self.layers):
            x_stage = features[i]
            for layer in stage_layers:
                x_stage = layer(x_stage)
                if torch.any(torch.isnan(x_stage)) or torch.any(torch.isinf(x_stage)):
                    print(f"NaN/Inf in transformer layer {i}")
                    x_stage = torch.nan_to_num(x_stage, nan=0.0)
            stage_outputs.append(x_stage)
            aux_out = self.aux_classifiers[i](x_stage.mean(dim=1))
            aux_outputs.append(aux_out)

        x = stage_outputs[0]
        for i in range(len(stage_outputs) - 1):
            current_features = x
            fine_features = stage_outputs[i + 1]
            x_proj = self.proj_layers[i](current_features)
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.pool_layers[i](x_proj).transpose(1, 2)
            x_att, _ = self.cross_attentions[i](query=x_proj, key=fine_features, value=fine_features)
            if torch.any(torch.isnan(x_att)) or torch.any(torch.isinf(x_att)):
                print(f"NaN/Inf in cross_attention layer {i}")
                x_att = torch.nan_to_num(x_att, nan=0.0)
            x = x_att + x_proj

        if self.use_multimodal and spectral_data is not None:
            spectral_features = self.spectral_embed(spectral_data)
            spectral_features = spectral_features + self.spectral_pos_embed
            x = self.modal_fusion(x, spectral_features)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.head(x)
        
        if self.training:
            return x, aux_outputs
        return x

# Initialize model
spectral_channels = 0
if has_multimodal:
    try:
        _, spectral_sample, _ = combined_train_dataset.data[0]
        spectral_channels = spectral_sample.shape[0]
        print(f"Detected {spectral_channels} spectral channels")
    except:
        print("Could not determine spectral channels. Defaulting to RGB-only model.")

model = HierarchicalVisionTransformer(
    image_size=299, 
    patch_size=13, 
    num_classes=7,
    dim=768,
    depths=[2, 2, 6], 
    heads=[4, 8, 16],
    mlp_ratio=4.0,
    multimodal_support=True,
    spectral_channels=spectral_channels,
    pretrained=True
)

pretrained_path = '/teamspace/studios/this_studio/best_simclr_pretrained.pth'
if os.path.exists(pretrained_path):
    pretrained_state_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict and 'projection_head' not in k}
    model_dict.update(pretrained_state_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded pretrained weights from {pretrained_path} with strict=False")
else:
    print(f"Warning: Pretrained model {pretrained_path} not found. Training from scratch.")
    model.apply(model._init_weights)

for param in model.patch_embeds.parameters():
    param.requires_grad = False
for param in model.layers[0].parameters():
    param.requires_grad = False

model = model.to(device)

train_class_weights = calculate_class_weights(combined_train_dataset)
print("Class weights:", train_class_weights)

criterion = nn.CrossEntropyLoss(weight=train_class_weights)

# Load checkpoint from epoch 25 and reinitialize optimizer
checkpoint_path = 'checkpoint_epoch_25.pth'
model.load_state_dict(torch.load(checkpoint_path))
print("Loaded checkpoint from epoch 25")
optimizer = Adam(model.parameters(), lr=0.000005, weight_decay=1e-3)  # Lower LR
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
scaler = GradScaler()  # For mixed precision

num_epochs = 50
best_val_loss = 0.6762  # Best from epoch 23, but resuming from 25
best_model_path = 'best_hierarchical_vit_finetuned.pth'
patience = 5
trigger_times = 0

for epoch in range(26, num_epochs):  # Resume from epoch 26
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        if has_multimodal:
            images, spectral, labels = [item.to(device) for item in batch]
        else:
            images, labels = [item.to(device) for item in batch]
        
        if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)):
            print(f"Skipping batch {batch_idx} due to NaN/Inf in input")
            continue
        
        optimizer.zero_grad()
        with autocast():  # Mixed precision
            outputs, aux_outputs = model(images, spectral if has_multimodal else None)
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                print(f"NaN/Inf in outputs at batch {batch_idx}")
                continue
            
            main_loss = criterion(outputs, labels)
            aux_loss = 0
            for aux_out in aux_outputs:
                aux_loss += criterion(aux_out, labels) * 0.1
            loss = main_loss + aux_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf in loss at batch {batch_idx}")
            continue
        
        loss = torch.clamp(loss, max=100.0)
        scaler.scale(loss).backward()  # Scale gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)  # Stricter clipping
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            if has_multimodal:
                images, spectral, labels = [item.to(device) for item in batch]
            else:
                images, labels = [item.to(device) for item in batch]
            with autocast():
                outputs = model(images, spectral if has_multimodal else None)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                if has_multimodal:
                    images, spectral, labels = [item.to(device) for item in batch]
                else:
                    images, labels = [item.to(device) for item in batch]
                with autocast():
                    outputs = model(images, spectral if has_multimodal else None)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        print(f"Test Loss at epoch {epoch+1}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    print(f"Epoch [{epoch+1}/{num_epochs}] - LR: {optimizer.param_groups[0]['lr']:.8f}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        if has_multimodal:
            images, spectral, labels = [item.to(device) for item in batch]
        else:
            images, labels = [item.to(device) for item in batch]
        with autocast():
            outputs = model(images, spectral if has_multimodal else None)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        if has_multimodal:
            images, spectral, labels = [item.to(device) for item in batch]
        else:
            images, labels = [item.to(device) for item in batch]
        with autocast():
            outputs = model(images, spectral if has_multimodal else None)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage',
               'Leaf Hopper Jassids', 'Leaf Redding', 'Leaf Variegation']
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

explainability_samples = 10
attention_fig = plt.figure(figsize=(20, 10))
for i in range(min(explainability_samples, len(test_dataset))):
    if has_multimodal:
        image, spectral, label = test_dataset[i]
        image = image.unsqueeze(0).to(device)
        spectral = spectral.unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast():
                output = model(image, spectral)
    else:
        image, label = test_dataset[i]
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast():
                output = model(image)
            
    if isinstance(output, tuple):
        output = output[0]
        
    _, predicted = torch.max(output.data, 1)
    
    plt.subplot(2, 5, i+1)
    img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    plt.imshow(img_np)
    plt.title(f"True: {class_names[label]}\nPred: {class_names[predicted.item()]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('model_predictions_finetuned.png')
print("Saved sample predictions visualization")

torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'test_accuracy': test_accuracy,
    'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
}, 'hierarchical_vit_finetuned_results.pth')
print("Saved complete model results")