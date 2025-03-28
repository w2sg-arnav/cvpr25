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
    "best_model_path": "best_hvt.pth",
    "embed_dim": 128,
    "num_heads": 4,
    "log_interval": 50  # Log training info every 50 batches
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
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import logging
from collections import Counter

class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8, normalize: bool = True):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            img_size (tuple): Desired image size (height, width), e.g., (384, 384).
            split (str): 'train' or 'val' to specify the dataset split.
            train_split (float): Fraction of data to use for training (e.g., 0.8 for 80% train, 20% val).
            normalize (bool): Whether to apply ImageNet normalization (True for fine-tuning, False for pretraining).
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        self.normalize = normalize
        
        # Define class names and labels
        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf", 
            "Herbicide Growth Damage", "Leaf Hopper Jassids", 
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Check if root_dir exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        
        logging.info(f"Loading dataset from: {root_dir}")
        
        # Traverse both Original and Augmented datasets
        for dataset_type in ["Original Dataset", "Augmented Dataset"]:
            dataset_path = os.path.join(root_dir, dataset_type)
            if not os.path.isdir(dataset_path):
                logging.warning(f"Dataset path does not exist, skipping: {dataset_path}")
                continue
            
            logging.info(f"Scanning dataset: {dataset_type}")
            for class_name in self.classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.isdir(class_path):
                    logging.warning(f"Class path does not exist, skipping: {class_path}")
                    continue
                
                logging.info(f"Scanning class: {class_name}")
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        # Convert to numpy arrays for splitting
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        # Check if any images were found
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the dataset at {root_dir}. Please check the directory structure and file extensions.")
        
        # Log class distribution
        class_counts = Counter(self.labels)
        logging.info("Class distribution:")
        for idx, count in class_counts.items():
            class_name = self.classes[idx]
            logging.info(f"Class {class_name}: {count} samples ({count/len(self.labels)*100:.2f}%)")
        
        logging.info(f"Total images found: {len(self.image_paths)}")
        
        # Split into train and validation sets
        indices = np.arange(len(self.image_paths))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        if self.split == "train":
            self.indices = train_indices
        else:  # val
            self.indices = val_indices
        
        logging.info(f"{self.split.capitalize()} split size: {len(self.indices)} samples")
        
        transforms_list = [
            T.Resize(self.img_size),
            T.ToTensor(),
        ]
        if self.normalize:
            transforms_list.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
            )
        self.transform = T.Compose(transforms_list)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from the split indices
        actual_idx = self.indices[idx]
        
        # Load image
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        logging.debug(f"Accessing image: {img_path}, label: {label}, idx: {idx}")
        
        # Load image as RGB
        img = Image.open(img_path).convert("RGB")
        rgb = self.transform(img)
        
        return rgb, torch.tensor(label, dtype=torch.long)

    def get_class_names(self):
        return self.classes
    
    def get_class_weights(self):
        # Compute class weights for imbalanced dataset
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        weights = [total_samples / (len(self.classes) * class_counts[i]) for i in range(len(self.classes))]
        return torch.tensor(weights, dtype=torch.float)

# Model Definition
class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop_path = nn.Dropout(0.1) if 0.1 > 0 else nn.Identity() #Use the correct drop identity here
        self._initialize_weights() #add initilizer

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

    def _initialize_weights(self): #add initilizer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size, num_classes, embed_dim = 128, num_heads = 4):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Identity()
        eff_output_dim = 1536

        self.patch_embed = nn.Conv2d(3, 128, kernel_size=4, stride=4)
        num_patches = (img_size[0] // 4) * (img_size[1] // 4)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 128))
        self.swin_layers = nn.ModuleList([SwinTransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(6)])
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
            # Print parameters
        for name, param in self.named_parameters():
            logging.info(f"Parameter name: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")

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

# Custom Metrics calculation function (moved out of utils due to import issues)
def compute_metrics(preds, labels):
    """Computes accuracy and F1-score using sklearn."""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1_score": f1}

# Training Loop
def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging for this epoch

    logging.getLogger().handlers = []

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"training.log",
                        filemode='w')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

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
    finetune_model = DiseaseAwareHVT(img_size=config["img_size"], num_classes=config["num_classes"], embed_dim = config["embed_dim"], num_heads = config["num_heads"]).to(device) #Changed name from pretrain to finetune

    # Optimizer and Schedulers
    optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=config["initial_learning_rate"], 
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

        # Log initial info
        root_logger.info(f"Using device: {device}")
        root_logger.info(f"Training dataset size: {len(train_dataset)} samples")
        root_logger.info(f"Validation dataset size: {len(val_dataset)} samples")
        root_logger.info("Starting training...")

        # Training phase
        finetune_model.train() #Changed name from pretrain to finetune
        train_loss = 0.0
        optimizer.zero_grad()

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/20") as pbar:
            for i, (rgb, labels) in enumerate(train_loader):
                rgb, labels = rgb.to(device), labels.to(device)
                logging.debug(f"Train RGB min: {rgb.min()}, max: {rgb.max()}")
                if torch.isnan(rgb).any() or torch.isinf(rgb).any():
                    logging.warning(f"Skipping batch {i} due to NaN/Inf in inputs")
                    continue

                if config["augmentations_enabled"]:
                    rgb_aug = augmentations(rgb)
                else:
                    rgb_aug = rgb

                with autocast(enabled=config["amp_enabled"]):
                    outputs = finetune_model(rgb_aug) #Changed name from pretrain to finetune
                    check_tensor(outputs, "Training Outputs")
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        logging.warning(f"Skipping batch {i} due to NaN/Inf in outputs")
                        continue
                    loss = nn.CrossEntropyLoss(weight=class_weights, 
                                               label_smoothing=config["label_smoothing"])(outputs, labels)

                loss = loss / config["accumulation_steps"]
                scaler.scale(loss).backward()
                if (i + 1) % config["accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), max_norm=config["clip_grad_norm"]) #Changed name from pretrain to finetune
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * config["accumulation_steps"]
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item() * config['accumulation_steps']:.4f}"})
                if i % config["log_interval"] == 0:
                    logging.info(f"Batch {i}: Loss = {loss.item() * config['accumulation_steps']:.4f}")

        train_loss /= len(train_loader)

        # Validation phase
        val_loss, val_acc, val_f1 = validate_model(finetune_model, val_loader, class_weights, device, #Changed name from pretrain to finetune
                                                   config["amp_enabled"])
        scheduler.step()
        lr_reducer.step(val_acc)

        # Log epoch summary
        print(f"Epoch {epoch + 1}/20 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Best Acc: {best_val_acc:.4f}")

        # Early stopping and model saving
        if val_acc > best_val_acc and not np.isnan(val_loss):
            best_val_acc = val_acc
            torch.save(finetune_model.state_dict(), config["best_model_path"]) #Changed name from pretrain to finetune
            logging.info(f"New best model saved with Val Acc: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logging.info(f"Early stopping after {config['patience']} epochs without improvement")
            break

    torch.save(finetune_model.state_dict(), "finetuned_hvt.pth") #Changed name from pretrain to finetune
    logging.info(f"Final model saved. Best validation accuracy: {best_val_acc:.4f}")

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
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    val_loss /= len(val_loader)
    if len(all_labels) > 0 and len(all_preds) > 0:
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        val_acc = 0.0
        val_f1 = 0.0

    logger.debug(f"Validation labels range: min {np.min(all_labels)}, max {np.max(all_labels)}")
    return val_loss, val_acc, val_f1

if __name__ == "__main__":
    #Move here
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

    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise