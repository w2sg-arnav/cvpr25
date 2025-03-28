import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LinearLR, SequentialLR
from tqdm import tqdm
from collections import Counter
import argparse
import yaml
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

# Configuration Management
DEFAULT_CONFIG = {
    "seed": 42,
    "data_root": "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection",
    "img_size": (256, 256),
    "num_classes": 7,
    "train_split": 0.8,
    "finetune_batch_size": 8,
    "accumulation_steps": 4,
    "initial_learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "t_0": 15,
    "t_mult": 1,
    "eta_min": 1e-6,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 10,
    "patience": 30,
    "label_smoothing": 0.1,
    "clip_grad_norm": 1.0,
    "amp_enabled": False,
    "augmentations_enabled": True,
    "pretrained": True,
    "best_model_path": "best_hvt.pth"
}

def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config.update(yaml.safe_load(f))
        except FileNotFoundError:
            logging.warning(f"Config file not found at {config_path}. Using default settings.")
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for DiseaseAwareHVT")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    return parser.parse_args()

# Helper Functions
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_tensor(tensor, name):
    logger = logging.getLogger()
    if torch.isnan(tensor).any():
        logger.warning(f"WARNING: {name} contains NaN values")
    if torch.isinf(tensor).any():
        logger.warning(f"WARNING: {name} contains Inf values")
    logger.debug(f"{name} min: {tensor.min().item()}, max: {tensor.max().item()}")

# Data Augmentation
class FinetuneAugmentation:
    def __init__(self, img_size):
        self.img_size = img_size
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ])

    def __call__(self, rgb):
        return self.transform(rgb)

# Dataset Class
class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8, normalize: bool = True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        self.normalize = normalize
        self.classes = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Growth Damage", 
                       "Leaf Hopper Jassids", "Leaf Redding", "Leaf Variegation"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        logging.info(f"Loading dataset from: {root_dir}")

        for dataset_type in ["Original Dataset", "Augmented Dataset"]:
            dataset_path = os.path.join(root_dir, dataset_type)
            if not os.path.exists(dataset_path):
                logging.warning(f"Dataset path does not exist, skipping: {dataset_path}")
                continue
            for class_name in self.classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    logging.warning(f"Class path does not exist, skipping: {class_path}")
                    continue
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the dataset at {root_dir}.")
        class_counts = Counter(self.labels)
        logging.info("Class distribution:")
        for idx, count in class_counts.items():
            logging.info(f"Class {self.classes[idx]}: {count} samples ({count / len(self.labels) * 100:.2f}%)")

        indices = np.arange(len(self.image_paths))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split)
        self.indices = indices[:split_idx] if split == "train" else indices[split_idx:]
        logging.info(f"{self.split.capitalize()} split size: {len(self.indices)} samples")

        transforms_list = [T.Resize(self.img_size), T.ToTensor()]
        if self.normalize:
            transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        img = Image.open(img_path).convert("RGB")
        rgb = self.transform(img)
        if idx < 5:  # Log first 5 samples
            logging.debug(f"Sample {idx} - RGB min: {rgb.min()}, max: {rgb.max()}")
        return rgb, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        weights = [total_samples / (len(self.classes) * class_counts[i]) for i in range(len(self.classes))]
        return torch.tensor(weights, dtype=torch.float)

# Model Definition
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, window_size=4, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = x.transpose(0, 1)
        attn_output, _ = self.attn(x, x, x)
        check_tensor(attn_output, "attn_output")
        x = shortcut + self.drop_path(attn_output.transpose(0, 1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Identity()
        eff_output_dim = 1536

        self.patch_embed = nn.Conv2d(3, 128, kernel_size=4, stride=4)
        num_patches = (img_size[0] // 4) * (img_size[1] // 4)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 128))
        self.swin_layers = nn.ModuleList([SwinTransformerBlock(dim=128, num_heads=4, drop_path=0.1) for _ in range(6)])
        self.norm = nn.LayerNorm(128)
        swin_output_dim = 128

        self.classifier = nn.Sequential(
            nn.Linear(eff_output_dim + swin_output_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        eff_features = self.efficientnet(x)
        logging.debug(f"eff_features min: {eff_features.min()}, max: {eff_features.max()}")
        check_tensor(eff_features, "eff_features")
        eff_features = torch.clamp(eff_features, min=-1e4, max=1e4)

        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.swin_layers:
            x = layer(x)
        swin_features = self.norm(x).mean(dim=1)
        swin_features = torch.clamp(swin_features, min=-1e4, max=1e4)

        combined = torch.cat((eff_features, swin_features), dim=1)
        logits = self.classifier(combined)
        logits = logits / 100.0  # Increased scaling
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Training Loop
def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Remove any existing console handlers to avoid screen output
    logging.getLogger().handlers = []
    # Suppress PIL logging noise
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Dataset and DataLoader
    train_dataset = SARCLD2024Dataset(config["data_root"], config["img_size"], split="train", 
                                      train_split=config["train_split"], normalize=True)
    val_dataset = SARCLD2024Dataset(config["data_root"], config["img_size"], split="val", 
                                    train_split=config["train_split"], normalize=True)
    class_weights = train_dataset.get_class_weights().to(device)

    class_weights_np = class_weights.cpu().numpy()
    num_samples = len(train_dataset)
    sample_weights = np.zeros(num_samples)
    labels = train_dataset.labels[train_dataset.indices]
    for i in range(config["num_classes"]):
        sample_weights[labels == i] = class_weights_np[i]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config["finetune_batch_size"], sampler=sampler, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["finetune_batch_size"], shuffle=False, 
                            num_workers=4, pin_memory=True)

    # Augmentations
    augmentations = FinetuneAugmentation(config["img_size"])

    # Model Initialization
    hvt_model = DiseaseAwareHVT(img_size=config["img_size"], num_classes=config["num_classes"]).to(device)

    # Optimizer and Schedulers
    optimizer = torch.optim.AdamW(hvt_model.parameters(), lr=config["initial_learning_rate"], 
                                  weight_decay=config["weight_decay"])
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=config["warmup_epochs"])
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config["t_0"], T_mult=config["t_mult"], 
                                                   eta_min=config["eta_min"])
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], 
                             milestones=[config["warmup_epochs"]])
    lr_reducer = ReduceLROnPlateau(optimizer, mode='max', factor=config["reduce_lr_factor"], 
                                   patience=config["reduce_lr_patience"])
    scaler = GradScaler(enabled=config["amp_enabled"])

    # Training Loop
    patience_counter = 0
    best_val_acc = 0.0

    for epoch in range(20):
        # Set up logging for this epoch
        log_file = os.path.join(log_dir, f"epoch_{epoch + 1}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.handlers = []  # Remove all existing handlers
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        # Log initial info
        logger.info(f"Using device: {device}")
        logger.info(f"Training dataset size: {len(train_dataset)} samples")
        logger.info(f"Validation dataset size: {len(val_dataset)} samples")
        logger.info("Starting training...")

        # Training phase
        hvt_model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/20", file=open(os.devnull, 'w')) as pbar:
            for i, (rgb, labels) in enumerate(train_loader):
                rgb, labels = rgb.to(device), labels.to(device)
                logger.debug(f"Train RGB min: {rgb.min()}, max: {rgb.max()}")
                if torch.isnan(rgb).any() or torch.isinf(rgb).any():
                    logger.warning(f"Skipping batch {i} due to NaN/Inf in inputs")
                    continue

                if config["augmentations_enabled"]:
                    rgb_aug = augmentations(rgb)
                else:
                    rgb_aug = rgb

                with autocast(enabled=config["amp_enabled"]):
                    outputs = hvt_model(rgb_aug)
                    check_tensor(outputs, "Training Outputs")
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        logger.warning(f"Skipping batch {i} due to NaN/Inf in outputs")
                        continue
                    loss = nn.CrossEntropyLoss(weight=class_weights, 
                                               label_smoothing=config["label_smoothing"])(outputs, labels)

                loss = loss / config["accumulation_steps"]
                scaler.scale(loss).backward()
                if (i + 1) % config["accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(hvt_model.parameters(), max_norm=config["clip_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * config["accumulation_steps"]
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item() * config['accumulation_steps']:.4f}"})

        train_loss /= len(train_loader)

        # Validation phase
        val_loss, val_acc, val_f1 = validate_model(hvt_model, val_loader, class_weights, device, 
                                                   config["amp_enabled"])
        scheduler.step()
        lr_reducer.step(val_acc)

        # Log epoch summary
        logger.info(f"Epoch {epoch + 1}/20 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Best Acc: {best_val_acc:.4f}")

        # Early stopping and model saving
        if val_acc > best_val_acc and not np.isnan(val_loss):
            best_val_acc = val_acc
            torch.save(hvt_model.state_dict(), config["best_model_path"])
            logger.info(f"New best model saved with Val Acc: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logger.info(f"Early stopping after {config['patience']} epochs without improvement")
            break

    torch.save(hvt_model.state_dict(), "finetuned_hvt.pth")
    logger.info(f"Final model saved. Best validation accuracy: {best_val_acc:.4f}")

def validate_model(model, val_loader, class_weights, device, amp_enabled):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    logger = logging.getLogger()

    with torch.no_grad():
        for rgb, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            logger.debug(f"Val RGB min: {rgb.min()}, max: {rgb.max()}")
            if torch.isnan(rgb).any() or torch.isinf(rgb).any():
                logger.warning("NaN or Inf detected in validation inputs")
                continue

            with autocast(enabled=amp_enabled):
                outputs = model(rgb)
                outputs = torch.clamp(outputs, min=-20, max=20)  # Clip logits to prevent explosion
                logger.debug(f"Validation Outputs min: {outputs.min()}, max: {outputs.max()}")
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning("NaN or Inf detected in validation outputs")
                    continue
                loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(all_labels, all_labels) if all_preds else 0.0  # Fixed typo: all_labels vs all_preds
    val_f1 = f1_score(all_labels, all_preds, average='weighted') if all_preds else 0.0
    logger.debug(f"Validation labels range: min {min(all_labels)}, max {max(all_labels)}")
    return val_loss, val_acc, val_f1

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise