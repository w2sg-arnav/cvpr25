import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Set Up the Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define global variables
DATA_ROOT = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
SAVE_PATH = "./phase1_checkpoints"
if not os.path.exists(DATA_ROOT):
    logger.error(f"Dataset root {DATA_ROOT} not found. Please verify the path or download the dataset.")
    raise FileNotFoundError(f"Dataset root {DATA_ROOT} not found.")
os.makedirs(SAVE_PATH, exist_ok=True)

# Step 2: Analyze Dataset
def analyze_dataset(data_path, dataset_type="Original"):
    """Analyze the dataset to understand class distribution and image properties."""
    logger.info(f"Analyzing {dataset_type} dataset at {data_path}...")
    
    try:
        dataset = datasets.ImageFolder(root=data_path)
    except Exception as e:
        logger.error(f"Failed to load {dataset_type} dataset at {data_path}: {e}")
        raise ValueError(f"Failed to load {dataset_type} dataset: {e}")

    class_names = dataset.classes
    class_counts = np.zeros(len(class_names), dtype=int)
    img_sizes = []
    corrupt_images = []

    for path, class_idx in dataset.samples:
        class_counts[class_idx] += 1
        try:
            with Image.open(path) as img:
                img_sizes.append(img.size)
        except Exception as e:
            corrupt_images.append((path, str(e)))

    total_images = len(dataset)
    class_distribution = {class_names[i]: (count, count / total_images * 100) 
                         for i, count in enumerate(class_counts)}

    # Log statistics
    logger.info(f"{dataset_type} dataset - Total images: {total_images}")
    logger.info(f"\n{dataset_type} Class distribution:")
    for class_name, (count, percentage) in class_distribution.items():
        logger.info(f"{class_name}: {count} images ({percentage:.2f}%)")

    if img_sizes:
        widths, heights = zip(*img_sizes)
        logger.info(f"\n{dataset_type} Image statistics:")
        logger.info(f"Width - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
        logger.info(f"Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")

    logger.info(f"\n{dataset_type} Corrupted images found: {len(corrupt_images)}")
    if corrupt_images:
        logger.warning(f"First 5 corrupted images in {dataset_type}:")
        for path, error in corrupt_images[:5]:
            logger.warning(f"- {path}: {error}")

    # Save class distribution plot
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts)
    plt.title(f'Class Distribution in {dataset_type} Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f'{dataset_type.lower()}_class_distribution.png'))
    plt.close()

    return {
        'class_names': class_names,
        'class_distribution': class_distribution,
        'corrupt_images': corrupt_images,
        'img_sizes': img_sizes
    }

# Analyze datasets
original_stats = analyze_dataset(os.path.join(DATA_ROOT, "Original Dataset"), "Original")
augmented_stats = analyze_dataset(os.path.join(DATA_ROOT, "Augmented Dataset"), "Augmented")

# Step 3: Define Data Preprocessing Transformations
train_transforms = transforms.Compose([
    transforms.Resize((320, 320)),  # Slightly larger for cropping
    transforms.RandomCrop((299, 299)),  # Match Inception V3 input
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),  # Reduced to preserve features
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 4: Custom Dataset Class
class CottonLeafDataset(Dataset):
    def __init__(self, samples, transform=None, return_paths=False, spectral_path=None):
        self.samples = samples
        self.transform = transform
        self.return_paths = return_paths
        self.spectral_path = spectral_path
        self.spectral_data = None
        self.has_multimodal = False
        
        # Simulate spectral data if not available
        if spectral_path and os.path.exists(spectral_path):
            self.spectral_data = datasets.ImageFolder(root=spectral_path).samples
            self.has_multimodal = True
        elif not spectral_path:
            logger.warning("Spectral data not found. Simulating grayscale as placeholder.")
            self.spectral_data = [(s[0], s[1]) for s in samples]  # Placeholder
            self.has_multimodal = True  # Enable multimodal support with simulated data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load the image as a PIL Image
            img = Image.open(img_path).convert('RGB')
            if min(img.size) < 50:
                raise ValueError("Image too small")

            # Simulate spectral data using the original PIL image
            spectral = None
            if self.has_multimodal:
                if self.spectral_path and os.path.exists(self.spectral_path):
                    spectral_path, _ = self.spectral_data[idx]
                    spectral = Image.open(spectral_path).convert('L')
                else:
                    # Simulate spectral data as grayscale from the original RGB image
                    spectral = img.convert('L')
                spectral = transforms.Resize((299, 299))(spectral)
                spectral = transforms.ToTensor()(spectral).squeeze(0)

            # Apply transform to the RGB image after spectral simulation
            if self.transform:
                img = self.transform(img)

            if self.return_paths:
                return img, spectral, label, img_path
            return (img, spectral, label) if self.has_multimodal else (img, label)

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            placeholder_img = torch.zeros((3, 299, 299))
            placeholder_spectral = torch.zeros((299, 299)) if self.has_multimodal else None
            if self.return_paths:
                return placeholder_img, placeholder_spectral, label, img_path
            return (placeholder_img, placeholder_spectral, label) if self.has_multimodal else (placeholder_img, label)

# Step 5: Load and Split Dataset
def load_and_split_dataset(root_path, train_transform, val_test_transform, spectral_path=None):
    dataset = datasets.ImageFolder(root=root_path)
    class_names = dataset.classes
    all_samples = dataset.samples

    valid_samples = []
    corrupt_samples = []
    for path, label in all_samples:
        try:
            with Image.open(path) as img:
                if min(img.size) < 50:
                    corrupt_samples.append((path, label))
                    continue
                valid_samples.append((path, label))
        except Exception:
            corrupt_samples.append((path, label))

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Valid samples: {len(valid_samples)}")
    logger.info(f"Corrupt samples: {len(corrupt_samples)}")

    # Stratified split with 70/15/15
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

    train_dataset = CottonLeafDataset(train_samples, transform=train_transform, spectral_path=spectral_path)
    val_dataset = CottonLeafDataset(val_samples, transform=val_test_transform, spectral_path=spectral_path)
    test_dataset = CottonLeafDataset(test_samples, transform=val_test_transform, spectral_path=spectral_path)

    return train_dataset, val_dataset, test_dataset, class_names

# Step 6: Load Augmented Dataset
def load_augmented_dataset(root_path, transform, spectral_path=None):
    dataset = datasets.ImageFolder(root=root_path)
    return CottonLeafDataset(dataset.samples, transform=transform, spectral_path=spectral_path)

# Step 7: Visualization Function
def visualize_batch(dataloader, n_samples=16, title="Sample Images", has_multimodal=False):
    try:
        batch = next(iter(dataloader))
        images = batch[0] if has_multimodal else batch[0]
        images_denorm = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

        grid = vutils.make_grid(images_denorm[:n_samples], nrow=4, padding=2, normalize=True)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(SAVE_PATH, f'{title.replace(" ", "_").lower()}.png'))
        plt.close()
        logger.info(f"Visualization saved for {title}")
    except Exception as e:
        logger.error(f"Failed to visualize batch: {e}")

# Step 8: Load and Prepare Data
spectral_path = os.path.join(DATA_ROOT, "Spectral Dataset")
if not os.path.exists(spectral_path):
    logger.warning("Spectral dataset not found. Simulating multimodal data.")
    spectral_path = None  # Use simulated data in dataset class
has_multimodal = spectral_path is not None or True  # Enable simulation

original_train_dataset, original_val_dataset, original_test_dataset, class_names = load_and_split_dataset(
    os.path.join(DATA_ROOT, "Original Dataset"),
    train_transforms,
    val_test_transforms,
    spectral_path
)

augmented_dataset = load_augmented_dataset(
    os.path.join(DATA_ROOT, "Augmented Dataset"),
    train_transforms,
    spectral_path
)

# Combine only for training (augmented used only in training)
combined_train_dataset = CottonLeafDataset(
    original_train_dataset.samples + augmented_dataset.samples,
    transform=train_transforms,
    spectral_path=spectral_path
)

# Create DataLoaders
train_loader = DataLoader(
    combined_train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    original_val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    original_test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Step 9: Visualize and Save
visualize_batch(train_loader, title="Training Samples", has_multimodal=has_multimodal)
visualize_batch(val_loader, title="Validation Samples", has_multimodal=has_multimodal)
visualize_batch(test_loader, title="Test Samples", has_multimodal=has_multimodal)

# Save preprocessed data
torch.save({
    'train_dataset': combined_train_dataset,
    'val_dataset': original_val_dataset,
    'test_dataset': original_test_dataset,
    'class_names': class_names,
    'original_stats': original_stats,
    'augmented_stats': augmented_stats,
    'has_multimodal': has_multimodal
}, os.path.join(SAVE_PATH, 'phase1_preprocessed_data.pth'))

logger.info(f"Training set size: {len(combined_train_dataset)}")
logger.info(f"Validation set size: {len(original_val_dataset)}")
logger.info(f"Test set size: {len(original_test_dataset)}")
logger.info(f"Number of classes: {len(class_names)}")
logger.info(f"Class names: {class_names}")
logger.info(f"Multimodal support: {has_multimodal}")
logger.info("Phase 1 completed: Dataset prepared and multimodal integration set up.")