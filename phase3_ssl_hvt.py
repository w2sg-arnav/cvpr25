import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import logging
import time
from PIL import Image
from data_utils import CottonLeafDataset, get_transforms, create_dataloaders

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

# Load Phase 2 data
SAVE_PATH = "./phase2_checkpoints"
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'phase2_results.pth')
if not os.path.exists(CHECKPOINT_PATH):
    logger.error(f"Phase 2 checkpoint not found at {CHECKPOINT_PATH}")
    raise FileNotFoundError(f"Phase 2 checkpoint not found at {CHECKPOINT_PATH}")

phase2_results = torch.load(CHECKPOINT_PATH)
class_names = phase2_results['inception']['report'].keys()
num_classes = len(class_names)
has_multimodal = True  # Assuming multimodal support from Phase 1

# Load Phase 1 datasets
PHASE1_CHECKPOINT_PATH = "./phase1_checkpoints/phase1_preprocessed_data.pth"
if not os.path.exists(PHASE1_CHECKPOINT_PATH):
    logger.error(f"Phase 1 checkpoint not found at {PHASE1_CHECKPOINT_PATH}")
    raise FileNotFoundError(f"Phase 1 checkpoint not found at {PHASE1_CHECKPOINT_PATH}")

checkpoint_data = torch.load(PHASE1_CHECKPOINT_PATH)
train_dataset = checkpoint_data['train_dataset']
val_dataset = checkpoint_data['val_dataset']
rare_classes = checkpoint_data['rare_classes']

# Custom SSL Dataset for Pretraining
class SSLCottonLeafDataset(Dataset):
    def __init__(self, samples, transform=None, rare_transform=None, rare_classes=None, has_multimodal=False):
        self.samples = samples
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes or []
        self.has_multimodal = has_multimodal
        
        # Simulate disease progression stages (early, mid, advanced)
        self.progression_stages = {
            'early': transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1)]),
            'mid': transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.3)]),
            'advanced': transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5)])
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            spectral = None
            
            if self.has_multimodal:
                spectral = self.simulate_spectral_from_rgb(img)
                spectral = transforms.Resize((299, 299))(spectral)
                spectral = transforms.ToTensor()(spectral).squeeze(0)
            
            # Apply random progression stage augmentation
            stage = np.random.choice(['early', 'mid', 'advanced'])
            img = self.progression_stages[stage](img)
            
            if label in self.rare_classes and self.rare_transform:
                img = self.rare_transform(img)
            elif self.transform:
                img = self.transform(img)
            
            return img, spectral, stage
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 299, 299)), torch.zeros((299, 299)) if self.has_multimodal else None, 'mid'

    def simulate_spectral_from_rgb(self, img):
        img = np.array(img) / 255.0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        NDVI = (G - R) / (G + R + 1e-5)
        NDVI = (NDVI + 1) / 2
        return Image.fromarray((NDVI * 255).astype(np.uint8))

# Dynamic Masking Function
def random_rectangular_masking(images, mask_ratio=0.3):
    """
    Apply random rectangular masking with adaptive ratio based on disease characteristics.
    """
    B, C, H, W = images.shape
    mask = torch.ones(B, 1, H, W).to(images.device)
    for i in range(B):
        h_mask = int(H * mask_ratio)
        w_mask = int(W * mask_ratio)
        top = np.random.randint(0, H - h_mask)
        left = np.random.randint(0, W - w_mask)
        mask[i, :, top:top+h_mask, left:left+w_mask] = 0
    return mask * images, mask

# SSL HVT Model with Pretext Tasks
class SSLHierarchicalVisionTransformer(nn.Module):
    def __init__(self, base_model, embed_dim=768, num_classes=7, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim * len(base_model.patch_embed.patch_sizes), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.progression_head = nn.Linear(256, 3)  # 3 stages: early, mid, advanced
        
    def forward(self, rgb, spectral=None, mask=None):
        if mask is not None:
            rgb = rgb * mask
        features = self.base_model(rgb, spectral)
        cls_features = features[:, 0]
        projection = self.projection_head(cls_features)
        progression_logits = self.progression_head(projection)
        return projection, progression_logits

    def get_representation(self, rgb, spectral=None, mask=None):
        with torch.no_grad():
            if mask is not None:
                rgb = rgb * mask
            features = self.base_model(rgb, spectral)
            cls_features = features[:, 0]
            return self.projection_head(cls_features)

# Contrastive Loss
class InfoNCECrossModalLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, rgb_features, spectral_features, labels=None):
        batch_size = rgb_features.size(0)
        if spectral_features is None:
            return torch.tensor(0.0).to(rgb_features.device)
        
        # Normalize features
        rgb_features = F.normalize(rgb_features, dim=1)
        spectral_features = F.normalize(spectral_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(rgb_features, spectral_features.T) / self.temperature
        labels = torch.arange(batch_size).to(rgb_features.device)
        loss = self.criterion(similarity_matrix, labels)
        return loss

# Training Function for SSL
def train_ssl_model(model, train_loader, val_loader, criterion_progression, criterion_contrastive, 
                   optimizer, num_epochs, device, mask_ratio_start=0.3, mask_ratio_end=0.6):
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_model_path = "./phase3_checkpoints/ssl_hvt_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        mask_ratio = mask_ratio_start + (mask_ratio_end - mask_ratio_start) * (epoch / (num_epochs - 1))
        
        for batch in train_loader:
            images, spectral, stage = batch[0].to(device), batch[1].to(device) if has_multimodal else None, batch[2]
            stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)
            
            optimizer.zero_grad()
            with autocast():
                masked_images, mask = random_rectangular_masking(images, mask_ratio)
                proj_features, prog_logits = model(images, spectral)
                masked_proj_features, _ = model(masked_images, spectral, mask)
                
                # Progression prediction loss
                prog_loss = criterion_progression(prog_logits, stage)
                
                # Contrastive loss (if spectral data available)
                contrastive_loss = criterion_contrastive(proj_features, spectral_features=None) if spectral is None else \
                                  criterion_contrastive(proj_features, model.base_model.spectral_embed(spectral))
                
                total_loss = prog_loss + 0.7 * contrastive_loss  # Dynamic weighting (70% contrastive, 30% progression)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - SSL Training Loss: {epoch_loss:.4f}, Mask Ratio: {mask_ratio:.2f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, spectral, stage = batch[0].to(device), batch[1].to(device) if has_multimodal else None, batch[2]
                stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)
                
                masked_images, _ = random_rectangular_masking(images, mask_ratio)
                proj_features, prog_logits = model(images, spectral)
                masked_proj_features, _ = model(masked_images, spectral)
                
                prog_loss = criterion_progression(prog_logits, stage)
                contrastive_loss = criterion_contrastive(proj_features, spectral_features=None) if spectral is None else \
                                  criterion_contrastive(proj_features, model.base_model.spectral_embed(spectral))
                
                total_loss = prog_loss + 0.7 * contrastive_loss
                val_loss += total_loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")
    
    return best_model_path

# Main Execution
if __name__ == "__main__":
    # Step 1: Load and Prepare SSL Dataset
    train_transforms, rare_transforms, _ = get_transforms()
    ssl_train_dataset = SSLCottonLeafDataset(
        train_dataset.samples, 
        transform=train_transforms, 
        rare_transform=rare_transforms, 
        rare_classes=rare_classes, 
        has_multimodal=has_multimodal
    )
    ssl_val_dataset = SSLCottonLeafDataset(
        val_dataset.samples, 
        transform=train_transforms, 
        rare_transform=rare_transforms, 
        rare_classes=rare_classes, 
        has_multimodal=has_multimodal
    )
    
    ssl_train_loader = DataLoader(ssl_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    ssl_val_loader = DataLoader(ssl_val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Step 2: Load HVT from Phase 2
    hvt_model = HierarchicalVisionTransformer(
        img_size=299, patch_sizes=[16, 32], in_chans=3, embed_dim=768, 
        num_heads=12, depth=16, num_classes=num_classes, dropout=0.1
    ).to(device)
    hvt_model.load_state_dict(torch.load("./phase2_checkpoints/HVT_best.pth"))
    ssl_model = SSLHierarchicalVisionTransformer(hvt_model, embed_dim=768, num_classes=7).to(device)

    # Step 3: Define Loss Functions and Optimizer
    criterion_progression = nn.CrossEntropyLoss()
    criterion_contrastive = InfoNCECrossModalLoss(temperature=0.07)
    optimizer = optim.AdamW(ssl_model.parameters(), lr=0.0001, weight_decay=0.01)

    # Step 4: Train SSL Model
    logger.info("Starting Self-Supervised Pretraining...")
    best_model_path = train_ssl_model(
        ssl_model, ssl_train_loader, ssl_val_loader, criterion_progression, criterion_contrastive,
        optimizer, num_epochs=100, device=device, mask_ratio_start=0.3, mask_ratio_end=0.6
    )

    # Step 5: Save Pretrained Model
    torch.save({
        'model_state_dict': ssl_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, "./phase3_checkpoints/ssl_hvt_checkpoint.pth")

    logger.info("Phase 3 completed: Self-Supervised Pretraining with disease progression prediction and cross-modal contrastive learning.")