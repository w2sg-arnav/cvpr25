import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms
import numpy as np
import math
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

# Define CustomDataset class for self-supervised learning
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = [item[0] for item in data.data]  # Access the data attribute of the original dataset
        self.transform = transform
        print(f"Dataset initialized with transform: {self.transform is not None}")  # Debug print
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            img1 = self.transform(image)
            img2 = self.transform(image)
        else:
            img1 = image
            img2 = image
        return img1, img2

# Step 1: Set Up the Environment and Load Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = torch.load('dataset_splits.pth')
combined_train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']  # Not used in pretraining, but loaded for consistency
test_dataset = checkpoint['test_dataset']  # Not used in pretraining, but loaded for consistency

# Define SimCLR augmentations
simclr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=299, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset for self-supervised learning (only training set)
train_dataset_ssl = CustomDataset(combined_train_dataset, transform=simclr_transform)
train_loader_ssl = DataLoader(
    train_dataset_ssl,
    batch_size=64,  # Increased batch size for better GPU utilization
    shuffle=True,
    num_workers=8,  # Use multiple workers for data loading
    pin_memory=True  # Speed up data transfer to GPU
)

print(f"Training set size for SSL: {len(train_dataset_ssl)}")

# Step 2: Define the Hierarchical Vision Transformer Architecture (Modified for SimCLR)
class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, image_size=299, patch_size=13, dim=768, depths=[2, 2, 6], heads=[4, 8, 16], mlp_ratio=4.0):
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
        self.projection_head = nn.Sequential(
            nn.Linear(self.stage_dims[-1], 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

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
        x = self.projection_head(x)
        return x

# Step 3: Define NT-Xent Loss for SimCLR
def nt_xent_loss(embeddings, temperature=0.5):
    batch_size = embeddings.size(0) // 2
    embeddings = F.normalize(embeddings, dim=1)
    z_i = embeddings[:batch_size]
    z_j = embeddings[batch_size:]
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.mm(representations, representations.t()) / temperature
    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positive_samples = torch.cat([sim_ij, sim_ji], dim=0)
    mask = (~torch.eye(2 * batch_size, device=similarity_matrix.device).bool()).float()
    numerator = torch.exp(positive_samples)
    denominator = mask * torch.exp(similarity_matrix)
    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
    loss = torch.mean(all_losses)
    return loss

# Step 4: Initialize Model, Optimizer, and Training Loop
model = HierarchicalVisionTransformer(image_size=299, patch_size=13)
model = model.to(device)
print("Hierarchical Vision Transformer initialized for SimCLR pretraining.")

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scaler = GradScaler()  # For mixed precision training

# Step 5: Pretraining Loop
num_epochs = 50
best_loss = float('inf')
patience = 5
trigger_times = 0
best_model_path = 'best_simclr_pretrained.pth'

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (img1, img2) in enumerate(train_loader_ssl):
        img1, img2 = img1.to(device), img2.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision training
        with autocast():
            z1 = model(img1)
            z2 = model(img2)
            embeddings = torch.cat([z1, z2], dim=0)
            loss = nt_xent_loss(embeddings, temperature=0.5)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader_ssl)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader_ssl)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with loss: {best_loss:.4f}")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Save the final pretrained model
final_model_path = 'final_simclr_pretrained.pth'
torch.save(model.state_dict(), final_model_path)
print(f"Final pretrained model saved at: {final_model_path}")

# Step 6: Define a linear evaluation protocol (optional - can be used later)
print("Pretraining complete! You can now use this model for downstream tasks.")