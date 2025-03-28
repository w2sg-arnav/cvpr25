import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import logging
import sys
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import os
import gc
import numpy as np
from PIL import Image
import torchvision.transforms as T
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset

# Assuming config.py exists with these constants
from config import PRETRAIN_LR, ACCUM_STEPS, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, PROGRESSIVE_RESOLUTIONS, NUM_CLASSES, TEMPERATURE, PROJECTION_DIM

logger = logging.getLogger(__name__)

# Suppress verbose EXIF logging from Pillow
logging.getLogger("PIL").setLevel(logging.INFO)

sys.path.append("/teamspace/studios/this_studio/cvpr25/")
#from phase4_finetuning.dataset import SARCLD2024Dataset
#from utils.augmentations import SimCLRAugmentation
#from utils.losses import InfoNCELoss
from utils.logging_setup import setup_logging

# Set up logging to file only, default to INFO level
setup_logging(log_file="pretrain.log", level=logging.INFO)

# HVT Block Definition
class HVTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

# DiseaseAwareHVT Model with Reduced Memory Footprint
class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size: tuple, embed_dim: int = 256, num_heads: int = 4, num_blocks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // 8) * (img_size[1] // 8)  # Reduced from 4 to 8: 1024 patches
        
        # RGB patch embedding with larger stride
        self.rgb_patch_embed = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Spectral patch embedding (unused in pretraining)
        self.spectral_patch_embed = nn.Conv2d(10, embed_dim, kernel_size=8, stride=8)
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks with reduced complexity
        self.blocks = nn.ModuleList([
            HVTBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])
        
        # Projection head for SimCLR
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, PROJECTION_DIM),
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)
    
    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor = None, pretrain: bool = True):
        rgb = self.rgb_patch_embed(rgb)
        rgb = rgb.flatten(2).transpose(1, 2)
        x = rgb + self.rgb_pos_embed
        
        # Spectral path (only if provided)
        # Spectral path (only if provided)
        if spectral is not None:
            spectral = self.spectral_patch_embed(spectral)
            spectral = spectral.flatten(2).transpose(1, 2) + self.spectral_pos_embed
            x = torch.cat((x, spectral), dim=1)
        
        # Use gradient checkpointing for all blocks
        for i, block in enumerate(self.blocks):
            x = checkpoint(block, x, use_reentrant=False)
            logger.debug(f"After block {i}, x requires grad: {x.requires_grad}")
        
        if pretrain:
            x = x.mean(dim=1)
            x = self.projection_head(x)
            logger.debug(f"Projection output requires grad: {x.requires_grad}")
        return x

# Pretrainer with Mixed Precision
class Pretrainer:
    def __init__(self, model: nn.Module, augmentations, loss_fn, device: str):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        
        self.optimizer = Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=PRETRAIN_LR,
            weight_decay=1e-4
        )
        self.scaler = GradScaler()  # For mixed precision
        self.accum_steps = ACCUM_STEPS
        self.step_count = 0
        
        # Log parameter status
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}, Trainable: {trainable_params}")
        logger.info(f"Total parameters: {total_params}, Trainable: {trainable_params}")
        for name, param in self.model.named_parameters():
            logger.info(f"{name}: requires_grad={param.requires_grad}")
    
    def train_step(self, rgb: torch.Tensor, spectral: torch.Tensor = None):
        self.model.train()
        rgb = rgb.to(self.device)
        if spectral is not None:
            spectral = spectral.to(self.device)
        
        rgb_view1 = self.augmentations(rgb)
        rgb_view2 = self.augmentations(rgb)
        logger.debug(f"RGB View 1 requires grad: {rgb_view1.requires_grad}")
        
        with autocast():
            features1 = self.model(rgb_view1, spectral, pretrain=True)
            features2 = self.model(rgb_view2, spectral, pretrain=True)
            loss = self.loss_fn(features1, features2) / self.accum_steps
        
        logger.debug(f"Features 1 requires grad: {features1.requires_grad}")
        logger.debug(f"Loss requires grad: {loss.requires_grad}")

        self.scaler.scale(loss).backward()

        # Check gradients (ignore spectral params in pretraining)
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None and 'spectral' not in name:
                logger.error(f"Trainable parameter {name} has no gradient!")
            elif param.requires_grad and 'spectral' not in name:
                logger.debug(f"{name} gradient norm: {param.grad.norm().item()}")
        
        self.step_count += 1
        if self.step_count % self.accum_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accum_steps
    
    def pretrain(self, train_loader, total_epochs: int):
        for epoch in range(1, total_epochs + 1):
            total_loss = 0.0
            num_batches = len(train_loader)
            with tqdm(total=num_batches, desc=f"Epoch {epoch}/{total_epochs}", file=sys.stdout) as pbar:
                for rgb, _ in train_loader:
                    batch_loss = self.train_step(rgb, None)
                    total_loss += batch_loss
                    pbar.update(1)
                    pbar.set_postfix({"Batch Loss": f"{batch_loss:.4f}"})
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{total_epochs} completed, Average Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {epoch}/{total_epochs}, Pretrain Loss: {avg_loss:.4f}")
            yield epoch
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

# Linear Probe Evaluation (unchanged except for batch size alignment)
def evaluate_linear_probe(model, train_loader, val_loader, device, num_classes=NUM_CLASSES, epochs=10):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    linear_probe = nn.Linear(model.embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        linear_probe.train()
        total_loss = 0.0
        for rgb, labels in train_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            with torch.no_grad():
                features = model(rgb, pretrain=False).mean(dim=1)
            logits = linear_probe(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Linear Probe Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    linear_probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for rgb, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            features = model(rgb, pretrain=False).mean(dim=1)
            logits = linear_probe(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logger.info(f"Linear Probe Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# Main Function with Memory Optimizations
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Using device: {device}")
    
    # Enable expandable segments to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    img_size = PROGRESSIVE_RESOLUTIONS[1]  # 256x256
    print(f"Image size: {img_size}")
    logger.info(f"Image size: {img_size}")
    
    dataset_root = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    train_dataset = SARCLD2024Dataset(dataset_root, img_size, split="train", train_split=0.8, normalize=False)
    val_dataset = SARCLD2024Dataset(dataset_root, img_size, split="val", train_split=0.8, normalize=False)
    
    # Reduced batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Reduced from potentially higher value
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        prefetch_factor=2
    )
    
    # Reduced model complexity
    model = DiseaseAwareHVT(img_size=img_size, num_blocks=4, num_heads=4).to(device)
    augmentations = SimCLRAugmentation(img_size, device=device)
    loss_fn = InfoNCELoss()
    
    pretrainer = Pretrainer(model, augmentations, loss_fn, device)
    
    for epoch in pretrainer.pretrain(train_loader, total_epochs=PRETRAIN_EPOCHS):
        if epoch % 5 == 0:
            accuracy = evaluate_linear_probe(model, train_loader, val_loader, device)
            checkpoint_path = f"pretrained_hvt_simclr_epoch_{epoch}.pth"
            pretrainer.save_model(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}, Linear Probe Accuracy: {accuracy:.2f}%")
            logger.info(f"Checkpoint saved to {checkpoint_path}, Linear Probe Accuracy: {accuracy:.2f}%")
    
    pretrainer.save_model("pretrained_hvt_simclr.pth")
    print("Pretrained model saved to pretrained_hvt_simclr.pth")
    logger.info("Pretrained model saved to pretrained_hvt_simclr.pth")

if __name__ == "__main__":
    main()

# Assuming these are in separate files but included here for completeness
class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8, normalize: bool = True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        self.normalize = normalize
        
        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf", 
            "Herbicide Growth Damage", "Leaf Hopper Jassids", 
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        
        logger.info(f"Loading dataset from: {root_dir}")
        
        for dataset_type in ["Original Dataset", "Augmented Dataset"]:
            dataset_path = os.path.join(root_dir, dataset_type)
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset path does not exist, skipping: {dataset_path}")
                continue
            
            logger.info(f"Scanning dataset: {dataset_type}")
            for class_name in self.classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    logger.warning(f"Class path does not exist, skipping: {class_path}")
                    continue
                
                logger.info(f"Scanning class: {class_name}")
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the dataset at {root_dir}")
        
        class_counts = Counter(self.labels)
        logger.info("Class distribution:")
        for idx, count in class_counts.items():
            class_name = self.classes[idx]
            logger.info(f"Class {class_name}: {count} samples ({count/len(self.labels)*100:.2f}%)")
        
        logger.info(f"Total images found: {len(self.image_paths)}")
        
        indices = np.arange(len(self.image_paths))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_split)
        if self.split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        logger.info(f"{self.split.capitalize()} split size: {len(self.indices)} samples")
        
        transforms_list = [
            T.Resize(self.img_size),
            T.ToTensor(),
        ]
        if self.normalize:
            transforms_list.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = T.Compose(transforms_list)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        img = Image.open(img_path).convert("RGB")
        rgb = self.transform(img)
        return rgb, torch.tensor(label, dtype=torch.long)

class InfoNCELoss:
    def __init__(self):
        self.temperature = TEMPERATURE
    
    def __call__(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        batch_size = features1.shape[0]
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        labels = torch.arange(batch_size, device=features1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class SimCLRAugmentation:
    def __init__(self, img_size: tuple, device: str = "cuda"):
        self.device = device
        self.img_size = img_size
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.resize(x, self.img_size)
        if torch.rand(1) < 0.5:
            x = TF.hflip(x)
        
        brightness_factor = torch.rand(1).item() * 1.6 + 0.2
        x = TF.adjust_brightness(x, brightness_factor)
        
        contrast_factor = torch.rand(1).item() * 1.6 + 0.2
        x = TF.adjust_contrast(x, contrast_factor)
        
        saturation_factor = torch.rand(1).item() * 1.6 + 0.2
        x = TF.adjust_saturation(x, saturation_factor)
        
        hue_factor = torch.rand(1).item() * 0.4 - 0.2
        x = TF.adjust_hue(x, hue_factor)
        
        if torch.rand(1) < 0.2:
            x = TF.rgb_to_grayscale(x, num_output_channels=3)
        
        x = TF.gaussian_blur(x, kernel_size=self.img_size[0]//10*2+1, sigma=(0.1, 2.0))
        x = x.to(self.device)
        return x