# phase3_ssl_hvt.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF  # For tensor-based transforms
import numpy as np
import logging
import time
from PIL import Image
from data_utils import CottonLeafDataset, get_transforms
from hvt_model import HierarchicalVisionTransformer

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

class SSLCottonLeafDataset(Dataset):
    def __init__(self, samples, transform=None, rare_transform=None, rare_classes=None, has_multimodal=False):
        self.samples = samples  # Expected to be a CottonLeafDataset instance
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes or []
        self.has_multimodal = has_multimodal

        # Define progression stages using tensor-compatible transforms
        self.progression_stages = {
            'early': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.1), 1.1),
            'mid': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.3), 1.3),
            'advanced': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.5), 1.5)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            # Directly access the sample from the dataset
            img, spectral, label = self.samples[idx]

            # Ensure img is a tensor and has the correct shape
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            
            # Normalize img shape to [3, 299, 299]
            if img.dim() == 2:  # [H, W] -> [1, H, W]
                img = img.unsqueeze(0)
            if img.shape[0] == 1:  # [1, H, W] -> [3, H, W] (repeat channels for grayscale)
                img = img.repeat(3, 1, 1)
            if img.shape[0] != 3 or img.shape[1] != 299 or img.shape[2] != 299:
                img = transforms.Resize((299, 299))(img)
                if img.shape[0] != 3:
                    img = img[:3, :, :]  # Ensure 3 channels

            # Apply transforms if not already applied
            if self.transform:
                # Since img is already a tensor, apply tensor-compatible transforms
                # We assume self.transform includes ToTensor(), so we skip that
                for t in self.transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        img = t(img)

            # Apply disease progression simulation
            stage = np.random.choice(['early', 'mid', 'advanced'])
            img = self.progression_stages[stage](img)

            # Apply rare class transforms if applicable
            if label in self.rare_classes and self.rare_transform:
                for t in self.rare_transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        img = t(img)

            # Ensure spectral is a tensor and has the correct shape
            if self.has_multimodal and spectral is not None:
                if not isinstance(spectral, torch.Tensor):
                    spectral = transforms.ToTensor()(spectral)
                if spectral.dim() == 2:  # [H, W] -> [1, H, W]
                    spectral = spectral.unsqueeze(0)
                if spectral.shape[0] != 1 or spectral.shape[1] != 299 or spectral.shape[2] != 299:
                    spectral = transforms.Resize((299, 299))(spectral)
                    spectral = spectral[:1, :, :]  # Ensure 1 channel

            return img, spectral, stage

        except Exception as e:
            logger.warning(f"Error loading sample at index {idx}: {e}. Returning placeholder.")
            # Return placeholder with consistent shapes
            return torch.zeros((3, 299, 299)), torch.zeros((1, 299, 299)) if self.has_multimodal else None, 'mid'

def random_rectangular_masking(images, mask_ratio=0.3):
    B, C, H, W = images.shape
    mask = torch.ones(B, 1, H, W).to(images.device)
    for i in range(B):
        h_mask = int(H * mask_ratio)
        w_mask = int(W * mask_ratio)
        top = np.random.randint(0, H - h_mask)
        left = np.random.randint(0, W - w_mask)
        mask[i, :, top:top+h_mask, left:left+w_mask] = 0
    return mask * images, mask

class SSLHierarchicalVisionTransformer(nn.Module):
    def __init__(self, base_model, num_classes=3):
        super().__init__()
        self.base_model = base_model
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, 299, 299).to(device)
            test_spectral = torch.randn(1, 1, 299, 299).to(device) if hasattr(base_model, 'has_multimodal') and base_model.has_multimodal else None
            test_output = self.base_model(test_input, test_spectral)
            if test_output.dim() == 3:
                test_output = test_output[:, 0, :]
            self.feature_dim = test_output.size(-1)
            logger.info(f"Detected feature dimension: {self.feature_dim}")
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        
        if hasattr(base_model, 'has_multimodal') and base_model.has_multimodal:
            self.spectral_projection = nn.Sequential(
                nn.Linear(299*299, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )
        
        self.progression_head = nn.Linear(256, num_classes)

    def forward(self, rgb, spectral=None, mask=None):
        if mask is not None:
            rgb = rgb * mask
        
        with torch.no_grad():
            features = self.base_model(rgb, spectral)
        
        if features.dim() == 3:
            features = features[:, 0, :]
        elif features.dim() != 2:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
            
        projection = self.projection_head(features)
        progression_logits = self.progression_head(projection)
        return projection, progression_logits

    def get_spectral_embedding(self, spectral):
        if hasattr(self, 'spectral_projection') and spectral is not None:
            spectral = spectral.flatten(start_dim=1)
            return self.spectral_projection(spectral)
        return None

class InfoNCECrossModalLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, rgb_features, spectral_features, labels=None):
        batch_size = rgb_features.size(0)
        if spectral_features is None:
            return torch.tensor(0.0).to(rgb_features.device)

        rgb_features = F.normalize(rgb_features, dim=1)
        spectral_features = F.normalize(spectral_features, dim=1)

        similarity_matrix = torch.matmul(rgb_features, spectral_features.T) / self.temperature
        labels = torch.arange(batch_size).to(rgb_features.device)
        loss = self.criterion(similarity_matrix, labels)
        return loss

def train_ssl_model(model, train_loader, val_loader, criterion_progression, criterion_contrastive,
                    optimizer, num_epochs, device, train_mean, train_std, mask_ratio_start=0.3, mask_ratio_end=0.6):
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_model_path = "./phase3_checkpoints/ssl_hvt_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        mask_ratio = mask_ratio_start + (mask_ratio_end - mask_ratio_start) * (epoch / (num_epochs - 1))

        for batch_idx, (images, spectral, stage) in enumerate(train_loader):
            images = images.to(device)
            spectral = spectral.to(device) if spectral is not None else None
            stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)
            
            if images.min() >= 0 and images.max() <= 1:
                images = (images - train_mean) / train_std

            optimizer.zero_grad()
            
            with autocast():
                masked_images, mask = random_rectangular_masking(images, mask_ratio)
                proj_features, prog_logits = model(images, spectral)
                masked_proj_features, _ = model(masked_images, spectral, mask)

                prog_loss = criterion_progression(prog_logits, stage)
                spectral_embed = model.get_spectral_embedding(spectral)
                contrastive_loss = criterion_contrastive(proj_features, spectral_embed)

                total_loss = prog_loss + 0.7 * contrastive_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item() * images.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - SSL Training Loss: {epoch_loss:.4f}, Mask Ratio: {mask_ratio:.2f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, spectral, stage in val_loader:
                images = images.to(device)
                spectral = spectral.to(device) if spectral is not None else None
                stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)
                
                if images.min() >= 0 and images.max() <= 1:
                    images = (images - train_mean) / train_std

                masked_images, _ = random_rectangular_masking(images, mask_ratio)
                proj_features, prog_logits = model(images, spectral)
                masked_proj_features, _ = model(masked_images, spectral)

                prog_loss = criterion_progression(prog_logits, stage)
                spectral_embed = model.get_spectral_embedding(spectral)
                contrastive_loss = criterion_contrastive(proj_features, spectral_embed)

                total_loss = prog_loss + 0.7 * contrastive_loss
                val_loss += total_loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        logger.info(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    return best_model_path

if __name__ == "__main__":
    try:
        PHASE1_CHECKPOINT_PATH = "./phase1_checkpoints/phase1_preprocessed_data.pth"
        HVT_CHECKPOINT_PATH = "./phase2_checkpoints/HVT_best.pth"
        PHASE3_SAVE_PATH = "./phase3_checkpoints"
        os.makedirs(PHASE3_SAVE_PATH, exist_ok=True)

        # Step 1: Load Phase 1 Data
        try:
            checkpoint_data = torch.load(PHASE1_CHECKPOINT_PATH)
            train_dataset = checkpoint_data['train_dataset']
            val_dataset = checkpoint_data['val_dataset']
            rare_classes = checkpoint_data['rare_classes']
            has_multimodal = checkpoint_data['has_multimodal']
            train_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            train_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            class_names = checkpoint_data['class_names']
            num_classes = len(class_names)
            logger.info("Successfully loaded Phase 1 data")
        except Exception as e:
            logger.error(f"Error loading Phase 1 checkpoints: {e}")
            raise

        # Step 2: Initialize and Load HVT Model
        try:
            hvt_model = HierarchicalVisionTransformer(
                num_classes=num_classes,
                img_size=299,
                patch_sizes=[16, 8, 4],
                embed_dims=[768, 384, 192],  # Match Phase 2 architecture
                num_heads=8,
                num_layers=16,  # Match Phase 2 architecture
                has_multimodal=has_multimodal,
                dropout=0.3
            ).to(device)
            
            if os.path.exists(HVT_CHECKPOINT_PATH):
                try:
                    state_dict = torch.load(HVT_CHECKPOINT_PATH, map_location=device)
                    hvt_model.load_state_dict(state_dict, strict=True)
                    logger.info("Loaded pretrained HVT weights")
                except RuntimeError as e:
                    logger.warning(f"Failed to load HVT checkpoint due to architecture mismatch: {e}")
                    logger.info("Proceeding with a randomly initialized HVT model")
            else:
                logger.warning("HVT checkpoint not found, initializing new model")
        except Exception as e:
            logger.error(f"Error initializing HVT model: {e}")
            raise

        # Step 3: Prepare SSL Dataset
        train_transforms, rare_transforms, _ = get_transforms()
        ssl_train_dataset = SSLCottonLeafDataset(
            train_dataset,
            transform=train_transforms,
            rare_transform=rare_transforms,
            rare_classes=rare_classes,
            has_multimodal=has_multimodal
        )
        ssl_val_dataset = SSLCottonLeafDataset(
            val_dataset,
            transform=train_transforms,
            rare_transform=rare_transforms,
            rare_classes=rare_classes,
            has_multimodal=has_multimodal
        )
        logger.info("SSL datasets prepared")

        # Step 4: Create DataLoaders
        ssl_train_loader = DataLoader(ssl_train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        ssl_val_loader = DataLoader(ssl_val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        logger.info("DataLoaders created")

        # Step 5: Initialize SSL Model
        ssl_model = SSLHierarchicalVisionTransformer(hvt_model, num_classes=3).to(device)
        logger.info("SSL model initialized")

        # Step 6: Define Loss Functions and Optimizer
        criterion_progression = nn.CrossEntropyLoss()
        criterion_contrastive = InfoNCECrossModalLoss(temperature=0.07)
        optimizer = optim.AdamW(ssl_model.parameters(), lr=3e-4, weight_decay=0.01)
        logger.info("Optimizer and loss functions initialized")

        # Step 7: Train SSL Model
        logger.info("Starting Self-Supervised Pretraining...")
        best_model_path = train_ssl_model(
            ssl_model, ssl_train_loader, ssl_val_loader, 
            criterion_progression, criterion_contrastive,
            optimizer, num_epochs=50, device=device,  # Increased to 50 epochs
            train_mean=train_mean, train_std=train_std
        )

        # Step 8: Save Final Model
        final_model_path = os.path.join(PHASE3_SAVE_PATH, "ssl_hvt_final.pth")
        torch.save({
            'model_state_dict': ssl_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_mean': train_mean,
            'train_std': train_std,
            'feature_dim': ssl_model.feature_dim,
            'has_multimodal': has_multimodal
        }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        logger.info("Phase 3 completed successfully!")

    except Exception as e:
        logger.error(f"Error in Phase 3: {e}")
        raise