# phase3_ssl_pretraining.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import logging
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from data_utils import CottonLeafDataset, create_dataloaders
from phase2_hvt_design import HierarchicalVisionTransformer  # Assuming HVT is defined here
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Output directory for Phase 3
output_dir = 'phase3_results'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load Preprocessed Data from Phase 1
checkpoint_path = os.path.join("./phase1_checkpoints", "phase1_preprocessed_data.pth")
try:
    checkpoint = torch.load(checkpoint_path)
except FileNotFoundError as e:
    logger.error(f"Preprocessed data not found: {e}")
    raise FileNotFoundError("Run Phase 1 first to generate preprocessed data.")

train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
class_names = checkpoint['class_names']
has_multimodal = checkpoint['has_multimodal']
num_classes = len(class_names)

logger.info(f"Training set size: {len(train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")
logger.info(f"Class names: {class_names}")

# Step 2: Define SimCLR Augmentations
# These augmentations create two views of the same image for contrastive learning
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(299, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Custom Dataset for SimCLR (Two Augmented Views)
class SimCLRDataset(CottonLeafDataset):
    def __init__(self, samples, transform, spectral_path=None, simulate_spectral="grayscale"):
        super().__init__(samples, transform=None, spectral_path=spectral_path, simulate_spectral=simulate_spectral)
        self.transform = transform

    def __getitem__(self, idx):
        img, spectral, label = super().__getitem__(idx)
        # Create two augmented views for SimCLR
        img1 = self.transform(img)
        img2 = self.transform(img)
        spectral1 = self.transform(spectral) if spectral is not None and self.has_multimodal else spectral
        spectral2 = self.transform(spectral) if spectral is not None and self.has_multimodal else spectral
        return (img1, img2, spectral1, spectral2), label

# Create SimCLR Datasets and DataLoaders
train_simclr_dataset = SimCLRDataset(
    train_dataset.samples,
    transform=simclr_transform,
    spectral_path=None,
    simulate_spectral="grayscale"
)
val_simclr_dataset = SimCLRDataset(
    val_dataset.samples,
    transform=simclr_transform,
    spectral_path=None,
    simulate_spectral="grayscale"
)

train_loader = DataLoader(
    train_simclr_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_simclr_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Step 4: Define NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device=device):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, features1, features2):
        """
        Compute NT-Xent loss for SimCLR.
        Theoretical grounding: The loss maximizes the mutual information between two augmented views
        of the same image, as derived in the InfoNCE framework (Oord et al., 2018).
        """
        batch_size = features1.shape[0]
        features = torch.cat([features1, features2], dim=0)
        similarity_matrix = self.cos(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(self.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

# Step 5: Initialize HVT Model
hvt_model = HierarchicalVisionTransformer(
    num_classes=num_classes,
    img_size=299,
    patch_sizes=[16, 8, 4],
    embed_dims=[768, 384, 192],
    num_heads=12,
    num_layers=12,
    has_multimodal=has_multimodal
).to(device)

# Projection Head for SimCLR
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Add projection head to HVT for SimCLR
projection_head = ProjectionHead(in_dim=sum([768, 384, 192]), hidden_dim=512, out_dim=128).to(device)

# Step 6: Training Setup
criterion = NTXentLoss(temperature=0.5, device=device)
optimizer = torch.optim.Adam(list(hvt_model.parameters()) + list(projection_head.parameters()), lr=0.0003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
scaler = GradScaler() if torch.cuda.is_available() else None

num_epochs = 50
best_val_loss = float('inf')
best_model_path = os.path.join(output_dir, 'best_ssl_hvt.pth')
history = {'train_loss': [], 'val_loss': [], 'val_f1_rare': []}

# Step 7: Training Loop
for epoch in range(num_epochs):
    hvt_model.train()
    projection_head.train()
    train_loss = 0.0
    start_time = time.time()

    for (img1, img2, spectral1, spectral2), _ in train_loader:
        img1, img2 = img1.to(device), img2.to(device)
        spectral1 = spectral1.to(device) if spectral1 is not None else None
        spectral2 = spectral2.to(device) if spectral2 is not None else None
        optimizer.zero_grad()

        with autocast() if scaler else torch.no_grad():
            # Forward pass for both views
            feat1 = hvt_model(img1, spectral1)
            feat2 = hvt_model(img2, spectral2)
            proj1 = projection_head(feat1)
            proj2 = projection_head(feat2)
            loss = criterion(proj1, proj2)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    epoch_time = time.time() - start_time
    history['train_loss'].append(train_loss)

    # Validation loop to monitor rare disease representation
    hvt_model.eval()
    projection_head.eval()
    val_loss = 0.0
    all_labels, all_features = [], []

    with torch.no_grad():
        for (img1, img2, spectral1, spectral2), labels in val_loader:
            img1, img2 = img1.to(device), img2.to(device)
            spectral1 = spectral1.to(device) if spectral1 is not None else None
            spectral2 = spectral2.to(device) if spectral2 is not None else None
            labels = labels.to(device)

            with autocast() if scaler else torch.no_grad():
                feat1 = hvt_model(img1, spectral1)
                feat2 = hvt_model(img2, spectral2)
                proj1 = projection_head(feat1)
                proj2 = projection_head(feat2)
                loss = criterion(proj1, proj2)
            val_loss += loss.item()

            # Collect features for k-NN evaluation
            features = feat1.cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_features.extend(features)
            all_labels.extend(labels_np)

    val_loss /= len(val_loader)
    history['val_loss'].append(val_loss)

    # k-NN evaluation to monitor rare class performance
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    if len(all_features) > 0 and len(all_labels) > 0:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(all_features, all_labels)
        pred_labels = knn.predict(all_features)
        # Compute F1-score for rare classes (e.g., Leaf Variegation)
        rare_class_idx = class_names.index("Leaf Variegation")
        f1_rare = f1_score(all_labels, pred_labels, labels=[rare_class_idx], average='macro', zero_division=0)
    else:
        f1_rare = 0.0  # Default if no valid data
    history['val_f1_rare'].append(f1_rare)

    scheduler.step()
    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val F1 (Leaf Variegation): {f1_rare:.4f}, Epoch Time: {epoch_time:.2f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'hvt_state_dict': hvt_model.state_dict(),
            'projection_head_state_dict': projection_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_f1_rare': f1_rare
        }, best_model_path)
        logger.info(f"New best model saved at {best_model_path}")

# Step 8: Plot Training History
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('SimCLR Pretraining Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_f1_rare'], label='Val F1 (Leaf Variegation)')
plt.title('Validation F1-Score for Rare Class')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pretraining_history.png'))
plt.close()

# Step 9: Save Pretrained Encoder for Phase 4
checkpoint_data = {
    'pretrained_hvt_state_dict': hvt_model.state_dict(),
    'class_names': class_names,
    'has_multimodal': has_multimodal,
    'training_history': history
}
torch.save(checkpoint_data, os.path.join(output_dir, 'phase3_pretrained_hvt.pth'))

logger.info(f"Phase 3 completed: Self-supervised pretraining completed. Pretrained encoder saved to {output_dir}/phase3_pretrained_hvt.pth.")