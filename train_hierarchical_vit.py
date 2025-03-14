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
import math

# Define CustomDataset class for supervised learning
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data.data  # Access the underlying list of (image, label) tuples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Check for invalid data
        if torch.any(torch.isnan(image)) or torch.any(torch.isinf(image)):
            print(f"Warning: Image at index {idx} contains NaN or Inf values")
            image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return image, label

# Step 1: Set Up the Environment and Load Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = torch.load('dataset_splits.pth')
combined_train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']

train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

print(f"Training set size: {len(combined_train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Step 2: Define the Hierarchical Vision Transformer Architecture
class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, image_size=299, patch_size=13, num_classes=7, dim=768, depths=[2, 2, 6], heads=[4, 8, 16], mlp_ratio=4.0):
        super(HierarchicalVisionTransformer, self).__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.dim = dim

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

        self.layers = nn.ModuleList()
        for i, (depth, head) in enumerate(zip(depths, heads)):
            stage_dim = self.stage_dims[i]
            self.layers.append(nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=stage_dim, nhead=head, dim_feedforward=int(stage_dim * mlp_ratio), dropout=0.1)
                for _ in range(depth)
            ]))

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
            self.cross_attentions.append(nn.MultiheadAttention(embed_dim=next_dim, num_heads=heads[i+1]))

        self.norm = nn.LayerNorm(self.stage_dims[-1])
        self.head = nn.Linear(self.stage_dims[-1], num_classes)

    def forward(self, x):
        features = []
        for i, (patch_embed, pos_embed) in enumerate(zip(self.patch_embeds, self.pos_embeds)):
            scale_factor = 2 ** i
            x_scaled = nn.functional.interpolate(x, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
            x_patched = patch_embed(x_scaled).flatten(2).transpose(1, 2)
            x_patched = x_patched + pos_embed
            features.append(x_patched)

        stage_outputs = []
        for i, stage_layers in enumerate(self.layers):
            x_stage = features[i]
            for layer in stage_layers:
                x_stage = layer(x_stage)
            stage_outputs.append(x_stage)

        x = stage_outputs[0]
        for i in range(len(stage_outputs) - 1):
            current_features = x
            fine_features = stage_outputs[i + 1]
            x_proj = self.proj_layers[i](current_features)
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.pool_layers[i](x_proj).transpose(1, 2)
            x_att, _ = self.cross_attentions[i](query=x_proj, key=fine_features, value=fine_features)
            x = x_att + x_proj

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

# Step 3: Load Pretrained Weights and Initialize Model
model = HierarchicalVisionTransformer(image_size=299, patch_size=13, num_classes=7)
pretrained_path = '/teamspace/studios/this_studio/best_hierarchical_vit.pth'
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path)
    # Remove unexpected keys and handle mismatches
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded partial pretrained weights from {pretrained_path} with strict=False")
else:
    print(f"Warning: Pretrained model {pretrained_path} not found. Training from scratch.")
model = model.to(device)

# Step 4: Define Loss Function, Optimizer, and Scheduler
def calculate_class_weights(dataset):
    class_counts = np.zeros(7)
    for _, label in dataset.data:
        class_counts[label] += 1
    total_samples = np.sum(class_counts)
    class_counts[class_counts == 0] = 1  # Avoid division by zero
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights).to(device)

train_class_weights = calculate_class_weights(combined_train_dataset)
print("Class weights:", train_class_weights)

criterion = nn.CrossEntropyLoss(weight=train_class_weights)
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Step 5: Fine-Tuning Loop with Early Stopping
num_epochs = 30
best_val_loss = float('inf')
best_model_path = 'best_finetuned_hierarchical_vit.pth'
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

# Step 6: Evaluate the Model on the Test Set
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Step 7: Per-Class Performance Analysis
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage',
               'Leaf Hopper Jassids', 'Leaf Redding', 'Leaf Variegation']
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))