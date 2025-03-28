import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as TF
from PIL import Image
import numpy as np
import os
import logging
import sys
from tqdm import tqdm
from collections import Counter
import gc
import time

# Config (assumed values, replace with your actual config.py)
PRETRAIN_LR = 3e-4
ACCUM_STEPS = 1
PRETRAIN_EPOCHS = 50
PRETRAIN_BATCH_SIZE = 16  # Reduced from 32 to mitigate memory issues
PROGRESSIVE_RESOLUTIONS = [(128, 128), (256, 256), (512, 512)]
NUM_CLASSES = 7
TEMPERATURE = 0.07
PROJECTION_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = "pretrain.log"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)

# HVT Block
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

# DiseaseAwareHVT Model
class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size: tuple, embed_dim: int = 128, num_heads: int = 4, num_blocks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // 8) * (img_size[1] // 8)

        self.rgb_patch_embed = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)
        nn.init.kaiming_normal_(self.rgb_patch_embed.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.rgb_patch_embed.bias)
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([HVTBlock(embed_dim, num_heads) for _ in range(num_blocks)])

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, PROJECTION_DIM),
        )

        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)

    def forward(self, rgb: torch.Tensor, pretrain: bool = True):
        rgb = self.rgb_patch_embed(rgb)
        rgb = rgb.flatten(2).transpose(1, 2)
        x = rgb + self.rgb_pos_embed

        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)

        if pretrain:
            x = x.mean(dim=1)
            x = self.projection_head(x)
        return x

# SARCLD2024 Dataset
class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8, normalize: bool = False):
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
            for class_name in self.classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    logger.warning(f"Class path does not exist, skipping: {class_path}")
                    continue
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

        if not self.image_paths:
            raise ValueError(f"No images found in dataset at {root_dir}")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        class_counts = Counter(self.labels)
        logger.info("Class distribution:")
        for idx, count in class_counts.items():
            logger.info(f"Class {self.classes[idx]}: {count} samples ({count/len(self.labels)*100:.2f}%)")
        logger.info(f"Total images found: {len(self.image_paths)}")

        indices = np.arange(len(self.image_paths))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split)
        self.indices = indices[:split_idx] if split == "train" else indices[split_idx:]
        logger.info(f"{split.capitalize()} split size: {len(self.indices)} samples")

        transforms_list = [TF.Resize(self.img_size), TF.ToTensor()]
        if normalize:
            transforms_list.append(TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = TF.Compose(transforms_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        img = Image.open(img_path).convert("RGB")
        rgb = self.transform(img)
        return rgb, torch.tensor(label, dtype=torch.long)

# SimCLR Augmentations
class SimCLRAugmentation:
    def __init__(self, img_size: tuple, device: str = DEVICE):
        self.img_size = img_size
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.Resize(self.img_size)(x)
        if torch.rand(1) < 0.5:
            x = TF.RandomHorizontalFlip(p=1.0)(x)
        brightness_factor = torch.rand(1).item() * 1.6 + 0.2
        contrast_factor = torch.rand(1).item() * 1.6 + 0.2
        saturation_factor = torch.rand(1).item() * 1.6 + 0.2
        hue_factor = torch.rand(1).item() * 0.4 - 0.2
        hue_min, hue_max = min(-hue_factor, hue_factor), max(-hue_factor, hue_factor)
        x = TF.ColorJitter(
            brightness=(max(0, 1 - brightness_factor), 1 + brightness_factor),
            contrast=(max(0, 1 - contrast_factor), 1 + contrast_factor),
            saturation=(max(0, 1 - saturation_factor), 1 + saturation_factor),
            hue=(hue_min, hue_max)
        )(x)
        if torch.rand(1) < 0.2:
            x = TF.Grayscale(num_output_channels=3)(x)
        x = TF.GaussianBlur(kernel_size=self.img_size[0]//10*2+1, sigma=(0.1, 2.0))(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x.to(self.device)

# InfoNCE Loss
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
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected in InfoNCELoss")
        return loss

# Pretrainer
class Pretrainer:
    def __init__(self, model: nn.Module, augmentations, loss_fn, device: str = DEVICE):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
        self.scaler = GradScaler()
        self.accum_steps = ACCUM_STEPS
        self.step_count = 0

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params}, Trainable: {trainable_params}")

    def train_step(self, rgb: torch.Tensor):
        self.model.train()
        rgb = rgb.to(self.device)

        logger.debug(f"Step {self.step_count} - Starting batch, RGB min: {rgb.min().item()}, max: {rgb.max().item()}")

        start_time = time.time()
        rgb_view1 = self.augmentations(rgb)
        rgb_view2 = self.augmentations(rgb)
        aug_time = time.time() - start_time
        logger.debug(f"Step {self.step_count} - Augmentation took {aug_time:.2f}s")

        try:
            with autocast():
                start_time = time.time()
                features1 = self.model(rgb_view1, pretrain=True)
                features2 = self.model(rgb_view2, pretrain=True)
                forward_time = time.time() - start_time
                logger.debug(f"Step {self.step_count} - Forward pass took {forward_time:.2f}s")
                logger.debug(f"Step {self.step_count} - Features1 min: {features1.min().item()}, max: {features1.max().item()}")
                logger.debug(f"Step {self.step_count} - Features2 min: {features2.min().item()}, max: {features2.max().item()}")

                loss = self.loss_fn(features1, features2) / self.accum_steps

            if not torch.isfinite(loss):
                logger.error(f"Step {self.step_count} - Loss is non-finite: {loss.item()}")
                return 0.0

            if self.step_count % 10 == 0:
                logger.info(f"Step {self.step_count}, Loss: {loss.item()}")

            start_time = time.time()
            self.scaler.scale(loss).backward()
            backward_time = time.time() - start_time
            logger.debug(f"Step {self.step_count} - Backward pass took {backward_time:.2f}s")

            found_inf = False
            for n, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    found_inf = True
                    logger.warning(f"Step {self.step_count} - Infinite gradient found for {n}")
                    break

            self.scaler.unscale_(self.optimizer)
            if not found_inf:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
            else:
                logger.warning(f"Step {self.step_count} - Skipping step due to infinite gradient")
            self.scaler.update()
            self.optimizer.zero_grad()

            self.step_count += 1
            logger.debug(f"Step {self.step_count-1} - Batch completed")
            return loss.item() * self.accum_steps

        except Exception as e:
            logger.exception(f"Step {self.step_count} - Exception in train_step: {e}")
            return float('nan')

    def pretrain(self, train_loader, total_epochs: int):
        try:
            for epoch in range(1, total_epochs + 1):
                total_loss = 0.0
                num_batches = len(train_loader)
                with tqdm(total=num_batches, desc=f"Epoch {epoch}/{total_epochs}", file=sys.stdout) as pbar:
                    for batch_idx, (rgb, _) in enumerate(train_loader):
                        logger.debug(f"Epoch {epoch}, Batch {batch_idx} - Starting")
                        batch_loss = self.train_step(rgb)
                        if torch.isnan(torch.tensor(batch_loss)):
                            logger.error(f"Epoch {epoch}, Batch {batch_idx} - Batch loss is NaN, skipping batch")
                            continue
                        total_loss += batch_loss
                        pbar.update(1)
                        pbar.set_postfix({"Batch Loss": f"{batch_loss:.4f}"})
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch}/{total_epochs}, Pretrain Loss: {avg_loss:.4f}")
                torch.cuda.empty_cache()  # Clear GPU memory between epochs
                yield epoch
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during pretraining: {e}")
            raise

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

# Linear Probe Evaluation
def evaluate_linear_probe(model, train_loader, val_loader, device, num_classes=NUM_CLASSES, epochs=10):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    linear_probe = nn.Linear(model.embed_dim, num_classes).to(device)
    optimizer = Adam(linear_probe.parameters(), lr=1e-3)
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

# Main Function
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    img_size = PROGRESSIVE_RESOLUTIONS[1]  # 256x256
    logger.info(f"Image size: {img_size}")

    dataset_root = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    train_dataset = SARCLD2024Dataset(dataset_root, img_size, split="train", train_split=0.8, normalize=False)
    val_dataset = SARCLD2024Dataset(dataset_root, img_size, split="val", train_split=0.8, normalize=False)

    train_loader = DataLoader(
        train_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    model = DiseaseAwareHVT(img_size=img_size).to(DEVICE)
    augmentations = SimCLRAugmentation(img_size)
    loss_fn = InfoNCELoss()

    pretrainer = Pretrainer(model, augmentations, loss_fn)

    try:
        for epoch in pretrainer.pretrain(train_loader, total_epochs=PRETRAIN_EPOCHS):
            if epoch % 5 == 0:
                accuracy = evaluate_linear_probe(model, train_loader, val_loader, DEVICE)
                checkpoint_path = f"pretrained_hvt_simclr_epoch_{epoch}.pth"
                pretrainer.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}, Linear Probe Accuracy: {accuracy:.2f}%")
    except Exception as e:
        logger.exception(f"Main loop error: {e}")
        raise

    pretrainer.save_model("pretrained_hvt_simclr.pth")
    logger.info("Pretrained model saved to pretrained_hvt_simclr.pth")

if __name__ == "__main__":
    main()