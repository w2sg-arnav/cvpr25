import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2  # For edge detection
import logging
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global paths
DATA_ROOT = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
SAVE_PATH_PHASE1 = "./phase1_checkpoints"
SAVE_PATH_PHASE2 = "./phase2_checkpoints"
os.makedirs(SAVE_PATH_PHASE1, exist_ok=True)
os.makedirs(SAVE_PATH_PHASE2, exist_ok=True)

if not os.path.exists(DATA_ROOT):
    logger.error(f"Dataset root {DATA_ROOT} not found.")
    raise FileNotFoundError(f"Dataset root {DATA_ROOT} not found.")

# Simulate spectral data using NDVI and EVI
def simulate_spectral_from_rgb(img):
    """Simulate NDVI and EVI from RGB channels as a multi-channel spectral proxy."""
    img = np.array(img) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    NDVI = (G - R) / (G + R + 1e-5)  # Normalized Difference Vegetation Index
    EVI = 2.5 * (G - R) / (G + 2.4 * R + 1 + 1e-5)  # Enhanced Vegetation Index
    spectral = np.stack([(NDVI + 1) / 2, (EVI + 1) / 2], axis=-1)  # Normalize to [0, 1]
    spectral = (spectral * 255).astype(np.uint8)
    return Image.fromarray(spectral)

# Custom Dataset Class
class CottonLeafDataset(Dataset):
    """Custom dataset for cotton leaf disease detection with multimodal support."""
    def __init__(self, samples, transform=None, rare_transform=None, rare_classes=None, spectral_path=None):
        self.samples = samples
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes or []
        self.spectral_path = spectral_path
        self.spectral_data = None
        self.has_multimodal = False
        
        if spectral_path and os.path.exists(spectral_path):
            self.spectral_data = datasets.ImageFolder(root=spectral_path).samples
            self.has_multimodal = True
        elif not spectral_path:
            logger.warning("Spectral data not found. Simulating NDVI+EVI as placeholder.")
            self.spectral_data = [(s[0], s[1]) for s in samples]
            self.has_multimodal = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if min(img.size) < 50:
                raise ValueError("Image too small")

            spectral = None
            if self.has_multimodal:
                if self.spectral_path and os.path.exists(self.spectral_path):
                    spectral_path, _ = self.spectral_data[idx]
                    spectral = Image.open(spectral_path).convert('L')
                else:
                    spectral = simulate_spectral_from_rgb(img)
                spectral = transforms.Resize((299, 299))(spectral)
                spectral = transforms.ToTensor()(spectral)  # Shape: (2, 299, 299)

            if label in self.rare_classes and self.rare_transform:
                img = self.rare_transform(img)
            elif self.transform:
                img = self.transform(img)

            return (img, spectral, label) if self.has_multimodal else (img, label)

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            placeholder_img = torch.zeros((3, 299, 299))
            placeholder_spectral = torch.zeros((2, 299, 299)) if self.has_multimodal else None
            return (placeholder_img, placeholder_spectral, label) if self.has_multimodal else (placeholder_img, label)

# Dataset Analysis
def analyze_dataset(data_path, dataset_type="Original", save_path="./analysis"):
    """Analyze dataset for class distribution, image properties, and field condition variability."""
    logger.info(f"Analyzing {dataset_type} dataset at {data_path}...")
    try:
        dataset = datasets.ImageFolder(root=data_path)
    except Exception as e:
        logger.error(f"Failed to load {dataset_type} dataset: {e}")
        raise ValueError(f"Failed to load {dataset_type} dataset: {e}")

    class_names = dataset.classes
    class_counts = np.zeros(len(class_names), dtype=int)
    img_sizes = []
    corrupt_images = []
    lighting_variability = []
    occlusion_variability = []

    for path, class_idx in dataset.samples:
        class_counts[class_idx] += 1
        try:
            with Image.open(path) as img:
                img_array = np.array(img)
                img_sizes.append(img.size)
                mean_intensity = np.mean(img_array)
                lighting_variability.append(mean_intensity)
                edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 100, 200)
                edge_density = np.sum(edges) / (img.size[0] * img.size[1])
                occlusion_variability.append(edge_density)
        except Exception as e:
            corrupt_images.append((path, str(e)))

    total_images = len(dataset)
    class_distribution = {class_names[i]: (count, count / total_images * 100) for i, count in enumerate(class_counts)}
    class_imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')

    logger.info(f"{dataset_type} dataset - Total images: {total_images}")
    for class_name, (count, percentage) in class_distribution.items():
        logger.info(f"{class_name}: {count} images ({percentage:.2f}%)")
    logger.info(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")
    if img_sizes:
        widths, heights = zip(*img_sizes)
        logger.info(f"Width - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
        logger.info(f"Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
    logger.info(f"Lighting - min: {min(lighting_variability):.1f}, max: {max(lighting_variability):.1f}, mean: {np.mean(lighting_variability):.1f}")
    logger.info(f"Occlusion - min: {min(occlusion_variability):.3f}, max: {max(occlusion_variability):.3f}, mean: {np.mean(occlusion_variability):.3f}")
    if corrupt_images:
        logger.warning(f"Corrupted images: {len(corrupt_images)}")
        for path, error in corrupt_images[:5]:
            logger.warning(f"- {path}: {error}")

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts)
    plt.title(f'Class Distribution in {dataset_type} Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_type.lower()}_class_distribution.png'))
    plt.close()

    return {
        'class_names': class_names,
        'class_distribution': class_distribution,
        'corrupt_images': corrupt_images,
        'class_imbalance_ratio': class_imbalance_ratio
    }

# Load and Split Dataset
def load_and_split_dataset(root_path, train_transform, rare_transform, val_test_transform, spectral_path=None, corrupt_images=None, rare_class_threshold=200):
    """Load and split dataset into train, validation, and test sets with stratified sampling."""
    dataset = datasets.ImageFolder(root=root_path)
    class_names = dataset.classes
    all_samples = dataset.samples

    class_counts = np.zeros(len(class_names), dtype=int)
    for _, label in all_samples:
        class_counts[label] += 1
    rare_classes = [i for i, count in enumerate(class_counts) if count < rare_class_threshold]
    logger.info(f"Rare classes (less than {rare_class_threshold} samples): {[class_names[i] for i in rare_classes]}")

    if corrupt_images is None:
        corrupt_images = []
    valid_samples = [(path, label) for path, label in all_samples if path not in {p for p, _ in corrupt_images}]
    logger.info(f"Total samples: {len(all_samples)}, Valid samples: {len(valid_samples)}")

    train_idx, temp_idx = train_test_split(
        range(len(valid_samples)),
        test_size=0.3,
        stratify=[s[1] for s in valid_samples],
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[valid_samples[i][1] for i in temp_idx],
        random_state=42
    )

    train_samples = [valid_samples[i] for i in train_idx]
    val_samples = [valid_samples[i] for i in val_idx]
    test_samples = [valid_samples[i] for i in test_idx]

    train_dataset = CottonLeafDataset(train_samples, transform=train_transform, rare_transform=rare_transform, rare_classes=rare_classes, spectral_path=spectral_path)
    val_dataset = CottonLeafDataset(val_samples, transform=val_test_transform, rare_transform=None, rare_classes=rare_classes, spectral_path=spectral_path)
    test_dataset = CottonLeafDataset(test_samples, transform=val_test_transform, rare_transform=None, rare_classes=rare_classes, spectral_path=spectral_path)

    return train_dataset, val_dataset, test_dataset, class_names, rare_classes

# Load Augmented Dataset
def load_augmented_dataset(root_path, train_transform, rare_transform, rare_classes, spectral_path=None, corrupt_images=None):
    """Load the augmented dataset for training."""
    dataset = datasets.ImageFolder(root=root_path)
    if corrupt_images is None:
        corrupt_images = []
    valid_samples = [(path, label) for path, label in dataset.samples if path not in {p for p, _ in corrupt_images}]
    return CottonLeafDataset(valid_samples, transform=train_transform, rare_transform=rare_transform, rare_classes=rare_classes, spectral_path=spectral_path)

# Define Transformations
def get_transforms():
    """Define transformations for training, validation, and test sets."""
    train_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomCrop((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rare_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomCrop((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    robustness_transforms = {
        'gaussian_blur': transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        'occlusion': transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        'lighting': transforms.ColorJitter(brightness=0.5)
    }

    return train_transforms, rare_transforms, val_test_transforms, robustness_transforms

# Create DataLoaders
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """Create DataLoaders for train, validation, and test datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

# Visualization Function
def visualize_batch(dataloader, n_samples=16, title="Sample Images", has_multimodal=False, save_path="./visualizations"):
    """Visualize a batch of images from the dataloader."""
    try:
        batch = next(iter(dataloader))
        images = batch[0] if has_multimodal else batch[0]
        images_denorm = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

        grid = vutils.make_grid(images_denorm[:n_samples], nrow=4, padding=2, normalize=True)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{title.replace(" ", "_").lower()}.png'))
        plt.close()
        logger.info(f"Visualization saved for {title}")
    except Exception as e:
        logger.error(f"Failed to visualize batch: {e}")

# Phase 1 Execution
def run_phase1():
    """Execute Phase 1: Dataset Preparation and Multimodal Integration."""
    # Analyze datasets
    original_stats = analyze_dataset(os.path.join(DATA_ROOT, "Original Dataset"), "Original", save_path=SAVE_PATH_PHASE1)
    augmented_stats = analyze_dataset(os.path.join(DATA_ROOT, "Augmented Dataset"), "Augmented", save_path=SAVE_PATH_PHASE1)

    # Get transformations
    train_transforms, rare_transforms, val_test_transforms, robustness_transforms = get_transforms()

    # Load and split original dataset
    spectral_path = os.path.join(DATA_ROOT, "Spectral Dataset")
    if not os.path.exists(spectral_path):
        spectral_path = None
    has_multimodal = spectral_path is not None or True
    original_train_dataset, original_val_dataset, original_test_dataset, class_names, rare_classes = load_and_split_dataset(
        os.path.join(DATA_ROOT, "Original Dataset"),
        train_transforms,
        rare_transforms,
        val_test_transforms,
        spectral_path,
        corrupt_images=original_stats['corrupt_images']
    )

    # Load augmented dataset
    augmented_dataset = load_augmented_dataset(
        os.path.join(DATA_ROOT, "Augmented Dataset"),
        train_transforms,
        rare_transforms,
        rare_classes,
        spectral_path,
        corrupt_images=augmented_stats['corrupt_images']
    )

    # Combine original training samples with augmented dataset
    combined_train_samples = original_train_dataset.samples + augmented_dataset.samples
    combined_train_dataset = CottonLeafDataset(
        combined_train_samples,
        transform=train_transforms,
        rare_transform=rare_transforms,
        rare_classes=rare_classes,
        spectral_path=spectral_path
    )

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        combined_train_dataset,
        original_val_dataset,
        original_test_dataset,
        batch_size=32
    )

    # Create robustness test loader
    test_robustness_dataset = CottonLeafDataset(
        original_test_dataset.samples,
        transform=transforms.Compose([val_test_transforms, robustness_transforms['gaussian_blur']]),
        rare_transform=None,
        rare_classes=rare_classes,
        spectral_path=spectral_path
    )
    test_robustness_loader = DataLoader(
        test_robustness_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Visualize
    visualize_batch(train_loader, title="Training Samples", has_multimodal=has_multimodal, save_path=SAVE_PATH_PHASE1)
    visualize_batch(val_loader, title="Validation Samples", has_multimodal=has_multimodal, save_path=SAVE_PATH_PHASE1)
    visualize_batch(test_loader, title="Test Samples", has_multimodal=has_multimodal, save_path=SAVE_PATH_PHASE1)

    # Save preprocessed data
    checkpoint_data = {
        'train_dataset': combined_train_dataset,
        'val_dataset': original_val_dataset,
        'test_dataset': original_test_dataset,
        'test_robustness_dataset': test_robustness_dataset,
        'class_names': class_names,
        'rare_classes': rare_classes,
        'original_stats': original_stats,
        'augmented_stats': augmented_stats,
        'has_multimodal': has_multimodal,
        'metadata': {
            'version': '1.1',
            'transforms': {
                'train_transforms': str(train_transforms),
                'rare_transforms': str(rare_transforms),
                'val_test_transforms': str(val_test_transforms)
            }
        }
    }
    torch.save(checkpoint_data, os.path.join(SAVE_PATH_PHASE1, 'phase1_preprocessed_data.pth'))

    logger.info("Phase 1 completed: Dataset prepared with class-specific augmentations and multimodal integration.")

# Multi-Scale Patch Embedding
class MultiScalePatchEmbed(nn.Module):
    """Multi-scale patch embedding for hierarchical processing."""
    def __init__(self, img_sizes=[128, 256, 299], patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_sizes = img_sizes
        self.patch_size = patch_size
        self.num_patches = [(img_size // patch_size) ** 2 for img_size in img_sizes]
        self.projections = nn.ModuleList([
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in img_sizes
        ])
        self.resizers = nn.ModuleList([
            nn.Sequential(
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            for img_size in img_sizes
        ])
        self.embed_dim = embed_dim

    def forward(self, x):
        multi_scale_features = []
        for resizer, proj in zip(self.resizers, self.projections):
            resized_x = resizer(x)
            patches = proj(resized_x).flatten(2).transpose(1, 2)
            multi_scale_features.append(patches)
        return multi_scale_features

# Cross-Attention Fusion
class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for RGB and spectral features."""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, rgb_features, spectral_features):
        rgb_features = self.norm1(rgb_features)
        if spectral_features is None:
            return rgb_features
        spectral_features = self.norm2(spectral_features)
        fused, _ = self.cross_attention(rgb_features, spectral_features, spectral_features)
        return self.norm3(fused + rgb_features)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with stochastic depth."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.15, stochastic_depth_prob=0.2):
        super().__init__()
        self.norm0 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.LayerNorm(int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = nn.Dropout(p=stochastic_depth_prob)

    def forward(self, x):
        x = self.norm0(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.stochastic_depth(x)
        mlp_output = self.mlp(x)
        return self.norm2(x + self.dropout(mlp_output))

# Hierarchical Vision Transformer
class HierarchicalVisionTransformer(nn.Module):
    """Hierarchical Vision Transformer for multi-scale disease detection."""
    def __init__(self, img_sizes=[128, 256, 299], patch_size=16, in_chans=3, embed_dim=256,
                 num_heads=8, depth=8, num_classes=7, dropout=0.15, stochastic_depth_prob=0.2):
        super().__init__()
        self.patch_embed = MultiScalePatchEmbed(img_sizes, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.cls_tokens = nn.Parameter(torch.zeros(len(img_sizes), 1, embed_dim))
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n + 1, embed_dim)) for n in self.num_patches
        ])
        self.dropout = nn.Dropout(dropout)
        self.spectral_patch_embeds = nn.ModuleList([
            nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size) if has_multimodal else None
            for _ in img_sizes
        ])
        self.cross_attention = CrossAttentionFusion(embed_dim, num_heads)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim * len(img_sizes))
        self.fusion_head = nn.Linear(embed_dim * len(img_sizes), embed_dim)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for param in (self.cls_tokens, *self.pos_embeds):
            nn.init.trunc_normal_(param, std=0.02)

    def forward(self, rgb, spectral=None):
        B = rgb.shape[0]
        multi_scale_features = self.patch_embed(rgb)
        scale_outputs = []
        for i, (features, pos_embed) in enumerate(zip(multi_scale_features, self.pos_embeds)):
            cls_tokens = self.cls_tokens[i].expand(B, -1, -1)
            features = torch.cat((cls_tokens, features), dim=1) + pos_embed.expand(B, -1, -1)
            features = self.dropout(features)
            if spectral is not None and self.spectral_patch_embeds[i] is not None:
                spectral_features = self.spectral_patch_embeds[i](spectral).flatten(2).transpose(1, 2)
                spectral_cls = torch.zeros(B, 1, self.embed_dim).to(spectral.device)
                spectral_features = torch.cat((spectral_cls, spectral_features), dim=1)
                features = self.cross_attention(features, spectral_features)
            for layer in self.transformer_layers:
                features = layer(features)
            scale_outputs.append(self.norm(features)[:, 0])
        combined_features = self.fusion_norm(torch.cat(scale_outputs, dim=-1))
        fused_features = self.head_norm(self.fusion_head(combined_features))
        return self.head(fused_features)

    def get_attention_weights(self, rgb, spectral=None):
        self.eval()
        attention_weights_all_scales = []
        with torch.no_grad():
            B = rgb.shape[0]
            multi_scale_features = self.patch_embed(rgb)
            for i, (features, pos_embed) in enumerate(zip(multi_scale_features, self.pos_embeds)):
                cls_tokens = self.cls_tokens[i].expand(B, -1, -1)
                features = torch.cat((cls_tokens, features), dim=1) + pos_embed.expand(B, -1, -1)
                features = self.dropout(features)
                if spectral is not None and self.spectral_patch_embeds[i] is not None:
                    spectral_features = self.spectral_patch_embeds[i](spectral).flatten(2).transpose(1, 2)
                    spectral_cls = torch.zeros(B, 1, self.embed_dim).to(spectral.device)
                    spectral_features = torch.cat((spectral_cls, spectral_features), dim=1)
                    features = self.cross_attention(features, spectral_features)
                attention_weights = []
                for layer in self.transformer_layers:
                    features, attn_weight = layer(features)
                    attention_weights.append(attn_weight)
                attention_weights_all_scales.append(attention_weights)
        return attention_weights_all_scales

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, model_name="HVT"):
    """Train the model with mixed precision and learning rate scheduling."""
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_path = os.path.join(SAVE_PATH_PHASE2, f"{model_name}_best.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[2].to(device)
            spectral = batch[1].to(device) if has_multimodal else None

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs, spectral)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = batch[2].to(device)
                spectral = batch[1].to(device) if has_multimodal else None
                with autocast():
                    outputs = model(inputs, spectral)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(f"Epoch {epoch+1}/{num_epochs} - {model_name}")
        logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Acc: {best_val_acc:.2f}%")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    return train_losses, val_losses, train_accs, val_accs, best_val_acc

# Evaluation Function
def evaluate_model(model, test_loader, criterion, device, class_names, model_name="HVT", num_samples=10):
    """Evaluate model with uncertainty estimation using Monte Carlo Dropout."""
    model.train()  # Enable dropout for MC Dropout
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            labels = batch[2].to(device)
            spectral = batch[1].to(device) if has_multimodal else None

            outputs_samples = []
            for _ in range(num_samples):
                with autocast():
                    outputs = model(inputs, spectral)
                outputs_samples.append(F.softmax(outputs, dim=1))
            outputs_mean = torch.stack(outputs_samples).mean(dim=0)
            outputs_var = torch.stack(outputs_samples).var(dim=0).mean(dim=1)  # Uncertainty

            test_loss += criterion(outputs_mean, labels).item() * inputs.size(0)
            _, predicted = torch.max(outputs_mean, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs_var.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100 * correct / total
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"\n{model_name} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    for class_name, metrics in report.items():
        if class_name in class_names:
            logger.info(f"{class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH_PHASE2, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return test_acc, report, cm, all_probs

# Efficiency Analysis
def efficiency_analysis(model, input_size=(3, 299, 299), device='cuda', model_name="HVT"):
    """Analyze model efficiency (size and inference time)."""
    model.eval()
    batch_sizes = [1, 32]
    efficiency_metrics = {}

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        spectral_input = torch.randn(batch_size, 2, 299, 299).to(device) if has_multimodal else None

        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
        num_trials = 100
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(dummy_input, spectral_input)
        avg_time = (time.time() - start_time) / num_trials * 1000

        efficiency_metrics[batch_size] = {'model_size': model_size, 'inference_time': avg_time}
        logger.info(f"\n{model_name} Efficiency (Batch Size {batch_size}):")
        logger.info(f"Model Size: {model_size:.2f} MB")
        logger.info(f"Inference Time: {avg_time:.2f} ms/sample")

    return efficiency_metrics

# Attention Visualization
def visualize_attention(model, dataloader, class_names, device, num_samples=5, save_path="./phase2_checkpoints"):
    """Visualize attention maps highlighting disease-relevant regions."""
    model.eval()
    batch = next(iter(dataloader))
    images = batch[0][:num_samples].to(device)
    spectral = batch[1][:num_samples].to(device) if has_multimodal else None
    labels = batch[2][:num_samples].numpy()

    attention_weights_all_scales = model.get_attention_weights(images, spectral)

    for scale_idx, (img_size, attention_weights) in enumerate(zip([128, 256, 299], attention_weights_all_scales)):
        avg_attention = torch.stack(attention_weights).mean(dim=0).mean(dim=1)
        num_patches_per_side = img_size // 16
        num_patches = num_patches_per_side ** 2

        for i in range(num_samples):
            attn_map = avg_attention[i, 1:num_patches+1].view(num_patches_per_side, num_patches_per_side)
            attn_map = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=(299, 299), mode='bilinear').squeeze()
            attn_map = (attn_map > attn_map.quantile(0.9)).float() * attn_map  # Highlight top 10%

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            img = images[i].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title(f"Class: {class_names[labels[i]]}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.imshow(attn_map.cpu().numpy(), cmap='jet', alpha=0.5)
            plt.title(f"Disease-Relevant Attention (Scale {img_size})")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"hvt_attention_sample_{i}_scale_{img_size}.png"))
            plt.close()

# Phase 2 Execution
def run_phase2():
    """Execute Phase 2: Baseline Replication with Inception V3 and HVT Development."""
    # Load Phase 1 data
    CHECKPOINT_PATH = os.path.join(SAVE_PATH_PHASE1, 'phase1_preprocessed_data.pth')
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Phase 1 checkpoint not found at {CHECKPOINT_PATH}")
        raise FileNotFoundError(f"Phase 1 checkpoint not found at {CHECKPOINT_PATH}")

    checkpoint_data = torch.load(CHECKPOINT_PATH)
    train_dataset = checkpoint_data['train_dataset']
    val_dataset = checkpoint_data['val_dataset']
    test_dataset = checkpoint_data['test_dataset']
    test_robustness_dataset = checkpoint_data['test_robustness_dataset']
    class_names = checkpoint_data['class_names']
    num_classes = len(class_names)
    has_multimodal = checkpoint_data['has_multimodal']
    class_distribution = checkpoint_data['original_stats']['class_distribution']

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_robustness_loader = DataLoader(
        test_robustness_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Compute class weights
    class_counts = np.array([class_distribution[name][0] for name in class_names])
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Inception V3 Baseline
    inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception_model.aux_logits = False
    num_ftrs = inception_model.fc.in_features
    inception_model.fc = nn.Linear(num_ftrs, num_classes)
    for name, param in inception_model.named_parameters():
        if "Mixed_7" not in name and "fc" not in name:
            param.requires_grad = False
    inception_model = inception_model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(inception_model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    logger.info("Training Inception V3...")
    train_losses, val_losses, train_accs, val_accs, best_val_acc = train_model(
        inception_model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
        device=device, scheduler=scheduler, model_name="InceptionV3"
    )

    inception_model.load_state_dict(torch.load(os.path.join(SAVE_PATH_PHASE2, "InceptionV3_best.pth")))
    test_acc, report, cm, uncertainties = evaluate_model(inception_model, test_loader, criterion, device, class_names, "InceptionV3")
    robustness_acc, _, _, _ = evaluate_model(inception_model, test_robustness_loader, criterion, device, class_names, "InceptionV3_Robustness")
    inception_metrics = efficiency_analysis(inception_model, model_name="InceptionV3")

    # HVT Model
    hvt_model = HierarchicalVisionTransformer(
        img_sizes=[128, 256, 299],
        patch_size=16,
        in_chans=3,
        embed_dim=256,
        num_heads=8,
        depth=8,
        num_classes=num_classes,
        dropout=0.15,
        stochastic_depth_prob=0.2
    ).to(device)

    criterion_hvt = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_hvt = optim.Adam(hvt_model.parameters(), lr=0.0001)
    scheduler_hvt = LambdaLR(optimizer_hvt, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 5.0))

    logger.info("Training HVT architecture...")
    train_losses_hvt, val_losses_hvt, train_accs_hvt, val_accs_hvt, best_val_acc_hvt = train_model(
        hvt_model, train_loader, val_loader, criterion_hvt, optimizer_hvt, num_epochs=20,
        device=device, scheduler=scheduler_hvt, model_name="HVT"
    )

    hvt_model.load_state_dict(torch.load(os.path.join(SAVE_PATH_PHASE2, "HVT_best.pth")))
    hvt_test_acc, hvt_report, hvt_cm, hvt_uncertainties = evaluate_model(hvt_model, test_loader, criterion_hvt, device, class_names, "HVT")
    hvt_robustness_acc, _, _, _ = evaluate_model(hvt_model, test_robustness_loader, criterion_hvt, device, class_names, "HVT_Robustness")
    hvt_metrics = efficiency_analysis(hvt_model, model_name="HVT")

    # Visualize Attention
    visualize_attention(hvt_model, test_loader, class_names, device, save_path=SAVE_PATH_PHASE2)

    # Save Results
    phase2_results = {
        'inception': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'robustness_acc': robustness_acc,
            'report': report,
            'confusion_matrix': cm,
            'uncertainties': uncertainties,
            'efficiency_metrics': inception_metrics
        },
        'hvt': {
            'train_losses': train_losses_hvt,
            'val_losses': val_losses_hvt,
            'train_accs': train_accs_hvt,
            'val_accs': val_accs_hvt,
            'best_val_acc': best_val_acc_hvt,
            'test_acc': hvt_test_acc,
            'robustness_acc': hvt_robustness_acc,
            'report': hvt_report,
            'confusion_matrix': hvt_cm,
            'uncertainties': hvt_uncertainties,
            'efficiency_metrics': hvt_metrics
        }
    }
    torch.save(phase2_results, os.path.join(SAVE_PATH_PHASE2, 'phase2_results.pth'))

    logger.info("Phase 2 completed: HVT architecture trained and evaluated.")

if __name__ == "__main__":
    run_phase1()
    run_phase2()