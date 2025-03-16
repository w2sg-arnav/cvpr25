# data_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from PIL import Image
from sklearn.model_selection import train_test_split
import logging
import cv2  # For edge detection to analyze occlusions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Dataset Class for Cotton Leaf Disease Detection
class CottonLeafDataset(Dataset):
    def __init__(self, samples, transform=None, return_paths=False, spectral_path=None):
        """
        Custom dataset for cotton leaf disease detection with multimodal support.

        Args:
            samples (list): List of (image_path, label) tuples.
            transform (callable, optional): Transformations to apply to the images.
            return_paths (bool): If True, return image paths along with data.
            spectral_path (str, optional): Path to spectral dataset.
        """
        self.samples = samples
        self.transform = transform
        self.return_paths = return_paths
        self.spectral_path = spectral_path
        self.spectral_data = None
        self.has_multimodal = False
        
        if spectral_path and os.path.exists(spectral_path):
            self.spectral_data = datasets.ImageFolder(root=spectral_path).samples
            self.has_multimodal = True
        elif not spectral_path:
            logger.warning("Spectral data not found. Simulating grayscale as placeholder.")
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
                    spectral = img.convert('L')
                spectral = transforms.Resize((299, 299))(spectral)
                spectral = transforms.ToTensor()(spectral).squeeze(0)

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

# Dataset Analysis with Field Condition Variability
def analyze_dataset(data_path, dataset_type="Original", save_path="./analysis"):
    """
    Analyze the dataset to understand class distribution, image properties, and field condition variability.

    Args:
        data_path (str): Path to the dataset.
        dataset_type (str): Type of dataset (e.g., "Original", "Augmented").
        save_path (str): Directory to save analysis visualizations.

    Returns:
        dict: Statistics including class distribution, corrupt images, and field condition variability.
    """
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
    lighting_variability = []  # To measure pixel intensity for lighting analysis
    occlusion_variability = []  # To measure edge density for occlusion analysis

    for path, class_idx in dataset.samples:
        class_counts[class_idx] += 1
        try:
            with Image.open(path) as img:
                img_array = np.array(img)
                img_sizes.append(img.size)

                # Lighting variability: Compute mean pixel intensity
                mean_intensity = np.mean(img_array)
                lighting_variability.append(mean_intensity)

                # Occlusion variability: Compute edge density using Canny edge detection
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(img_gray, 100, 200)
                edge_density = np.sum(edges) / (img.size[0] * img.size[1])
                occlusion_variability.append(edge_density)
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

    logger.info(f"\n{dataset_type} Field condition variability:")
    logger.info(f"Lighting (mean pixel intensity) - min: {min(lighting_variability):.1f}, max: {max(lighting_variability):.1f}, mean: {np.mean(lighting_variability):.1f}")
    logger.info(f"Occlusion (edge density) - min: {min(occlusion_variability):.3f}, max: {max(occlusion_variability):.3f}, mean: {np.mean(occlusion_variability):.3f}")

    logger.info(f"\n{dataset_type} Corrupted images found: {len(corrupt_images)}")
    if corrupt_images:
        logger.warning(f"First 5 corrupted images in {dataset_type}:")
        for path, error in corrupt_images[:5]:
            logger.warning(f"- {path}: {error}")

    # Save visualizations
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts)
    plt.title(f'Class Distribution in {dataset_type} Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_type.lower()}_class_distribution.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(lighting_variability, bins=50, color='blue', alpha=0.7)
    plt.title(f'Lighting Variability in {dataset_type} Dataset')
    plt.xlabel('Mean Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_path, f'{dataset_type.lower()}_lighting_variability.png'))
    plt.close()

    return {
        'class_names': class_names,
        'class_distribution': class_distribution,
        'corrupt_images': corrupt_images,
        'img_sizes': img_sizes,
        'lighting_variability': lighting_variability,
        'occlusion_variability': occlusion_variability
    }

# Load and Split Dataset with Corrupt Image Handling
def load_and_split_dataset(root_path, train_transform, val_test_transform, spectral_path=None, corrupt_images=None):
    """
    Load and split the dataset into train, validation, and test sets with stratified sampling.

    Args:
        root_path (str): Path to the dataset.
        train_transform (callable): Transformations for training data.
        val_test_transform (callable): Transformations for validation/test data.
        spectral_path (str, optional): Path to spectral dataset.
        corrupt_images (list, optional): List of corrupt image paths to exclude.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names)
    """
    dataset = datasets.ImageFolder(root=root_path)
    class_names = dataset.classes
    all_samples = dataset.samples

    # Exclude corrupt images
    if corrupt_images is None:
        corrupt_images = []
    valid_samples = []
    corrupt_samples = []
    corrupt_paths = {path for path, _ in corrupt_images}
    for path, label in all_samples:
        if path in corrupt_paths:
            corrupt_samples.append((path, label))
            continue
        try:
            with Image.open(path) as img:
                if min(img.size) < 50:
                    corrupt_samples.append((path, label))
                    continue
                valid_samples.append((path, label))
        except Exception:
            corrupt_samples.append((path, label))

    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Valid samples after excluding corrupt images: {len(valid_samples)}")
    logger.info(f"Corrupt samples: {len(corrupt_samples)}")

    # Stratified split
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

# Load Augmented Dataset
def load_augmented_dataset(root_path, transform, spectral_path=None, corrupt_images=None):
    """
    Load the augmented dataset.

    Args:
        root_path (str): Path to the augmented dataset.
        transform (callable): Transformations to apply.
        spectral_path (str, optional): Path to spectral dataset.
        corrupt_images (list, optional): List of corrupt image paths to exclude.

    Returns:
        CottonLeafDataset: Augmented dataset.
    """
    dataset = datasets.ImageFolder(root=root_path)
    if corrupt_images is None:
        corrupt_images = []
    valid_samples = [(path, label) for path, label in dataset.samples if path not in {p for p, _ in corrupt_images}]
    return CottonLeafDataset(valid_samples, transform=transform, spectral_path=spectral_path)

# Visualization Function
def visualize_batch(dataloader, n_samples=16, title="Sample Images", has_multimodal=False, save_path="./visualizations"):
    """
    Visualize a batch of images from the dataloader.

    Args:
        dataloader (DataLoader): DataLoader to visualize.
        n_samples (int): Number of samples to display.
        title (str): Title of the visualization.
        has_multimodal (bool): If True, dataset includes spectral data.
        save_path (str): Directory to save the visualization.
    """
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

# Define Standard Transformations
def get_transforms():
    """
    Define standard transformations for training, validation, and test sets.

    Returns:
        tuple: (train_transforms, val_test_transforms)
    """
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

    val_test_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms

# Utility to Create DataLoaders
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Create DataLoaders for train, validation, and test datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
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

if __name__ == "__main__":
    # Example usage
    DATA_ROOT = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    if not os.path.exists(DATA_ROOT):
        logger.error(f"Dataset root {DATA_ROOT} not found.")
        raise FileNotFoundError(f"Dataset root {DATA_ROOT} not found.")

    # Analyze dataset
    original_stats = analyze_dataset(os.path.join(DATA_ROOT, "Original Dataset"), "Original")
    augmented_stats = analyze_dataset(os.path.join(DATA_ROOT, "Augmented Dataset"), "Augmented")

    # Get transformations
    train_transforms, val_test_transforms = get_transforms()

    # Load and split dataset
    spectral_path = os.path.join(DATA_ROOT, "Spectral Dataset")
    if not os.path.exists(spectral_path):
        spectral_path = None
    original_train_dataset, original_val_dataset, original_test_dataset, class_names = load_and_split_dataset(
        os.path.join(DATA_ROOT, "Original Dataset"),
        train_transforms,
        val_test_transforms,
        spectral_path,
        corrupt_images=original_stats['corrupt_images']
    )

    # Load augmented dataset
    augmented_dataset = load_augmented_dataset(
        os.path.join(DATA_ROOT, "Augmented Dataset"),
        train_transforms,
        spectral_path,
        corrupt_images=augmented_stats['corrupt_images']
    )

    # Combine datasets
    combined_train_dataset = CottonLeafDataset(
        original_train_dataset.samples + augmented_dataset.samples,
        transform=train_transforms,
        spectral_path=spectral_path
    )

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        combined_train_dataset,
        original_val_dataset,
        original_test_dataset,
        batch_size=32
    )

    # Visualize
    has_multimodal = spectral_path is not None or True
    visualize_batch(train_loader, title="Training Samples", has_multimodal=has_multimodal)
    visualize_batch(val_loader, title="Validation Samples", has_multimodal=has_multimodal)
    visualize_batch(test_loader, title="Test Samples", has_multimodal=has_multimodal)