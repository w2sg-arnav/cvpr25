import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import logging
import cv2
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility function to simulate spectral data (NDVI).
def simulate_spectral_from_rgb(img: Image.Image) -> Image.Image:
    """Simulates NDVI from RGB image data.
    Args:
        img (PIL.Image): RGB image.
    Returns:
        PIL.Image: Simulated NDVI image.
    """
    img = np.array(img) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    NDVI = (G - R) / (G + R + 1e-5)  # Avoid division by zero
    NDVI = (NDVI + 1) / 2  # Normalize to [0, 1]
    NDVI = Image.fromarray((NDVI * 255).astype(np.uint8))
    return NDVI

# Custom Dataset Class for Cotton Leaf Disease Detection
class CottonLeafDataset(Dataset):
    """Custom dataset for cotton leaf disease detection. Supports RGB and spectral data.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[transforms.Compose] = None, rare_transform: Optional[transforms.Compose] = None, rare_classes: Optional[List[int]] = None, spectral_dir: Optional[str] = None, simulate_spectral: bool = True):
        """
        Args:
            samples (List[Tuple[str, int]]): List of (image_path, label) tuples.
            transform (transforms.Compose, optional): Transformations for non-rare classes.
            rare_transform (transforms.Compose, optional): Transformations for rare classes.
            rare_classes (List[int], optional): List of indices representing rare classes.
            spectral_dir (str, optional): Path to spectral dataset directory.  If None, spectral data is simulated.
            simulate_spectral (bool): Set to False if your RGB data is also spectral.
        """
        self.samples = samples
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes or []
        self.spectral_dir = spectral_dir
        self.simulate_spectral = simulate_spectral
        self.has_multimodal = (spectral_dir is not None) or simulate_spectral
        self.spectral_samples = self._load_spectral_samples()

    def _load_spectral_samples(self) -> Optional[List[Tuple[str, int]]]:
        """Loads spectral data samples, simulating them if necessary.
        Returns:
            List[Tuple[str, int]]: List of (spectral_image_path, label) tuples, or None if no spectral data.
        """
        if self.spectral_dir and os.path.exists(self.spectral_dir):
            try:
                spectral_dataset = datasets.ImageFolder(root=self.spectral_dir)
                return spectral_dataset.samples
            except Exception as e:
                logger.error(f"Error loading spectral data from {self.spectral_dir}: {e}")
                return None
        elif self.simulate_spectral:
            logger.warning("Spectral data directory not provided. Simulating NDVI.")
            return [(s[0], s[1]) for s in self.samples]  # Use RGB image path for simulation
        else:
            logger.info("No spectral data directory or simulation requested. Using RGB only.")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Returns a tuple of (image, spectral, label).
        If no spectral data is available, spectral is None.
        """
        img_path, label = self.samples[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            img = self._apply_transform(img, label) # Apply transformation

            spectral = None
            if self.has_multimodal:
                if self.spectral_samples:
                    spectral_path, _ = self.spectral_samples[idx]
                    try:
                        if self.spectral_dir:
                            spectral = Image.open(spectral_path).convert('L') # For now, convert to grayscale
                        else:
                            spectral = simulate_spectral_from_rgb(img)  # Simulate if needed
                        spectral = transforms.Resize((299, 299))(spectral)
                        spectral = transforms.ToTensor()(spectral).squeeze(0) # Squeeze for grayscale
                    except Exception as e:
                        logger.warning(f"Problem opening spectral image: {e}. Setting to None")
                        spectral = torch.zeros((299, 299)) #placeholder spectral
                else:
                    logger.warning("No spectral samples loaded, setting spectral to None")

            return img, spectral, label

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            img = torch.zeros((3, 299, 299))
            spectral = torch.zeros((299, 299)) if self.has_multimodal else None # create placeholder spectral

            return img, spectral, label

    def _apply_transform(self, img: Image.Image, label: int) -> torch.Tensor:
        """Applies the correct transformation (rare or standard) to the image.
        """
        if label in self.rare_classes and self.rare_transform:
            img = self.rare_transform(img)
        elif self.transform:
            img = self.transform(img)

        return img

# Dataset Analysis with Field Condition Variability
def analyze_dataset(data_path: str, dataset_type: str = "Original", save_path: str = "./analysis") -> dict:
    """Analyzes a dataset to understand class distribution, image properties, and field condition variability.

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
    lighting_variability = []
    occlusion_variability = []

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

    # Compute class imbalance ratio
    class_imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')

    # Log statistics
    logger.info(f"{dataset_type} dataset - Total images: {total_images}")
    logger.info(f"\n{dataset_type} Class distribution:")
    for class_name, (count, percentage) in class_distribution.items():
        logger.info(f"{class_name}: {count} images ({percentage:.2f}%)")
    logger.info(f"Class imbalance ratio: {class_imbalance_ratio:.2f}")

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
        'occlusion_variability': occlusion_variability,
        'class_imbalance_ratio': class_imbalance_ratio
    }

# Load and Split Dataset with Corrupt Image Handling
def load_and_split_dataset(root_path: str, train_transform: transforms.Compose, rare_transform: transforms.Compose, val_test_transform: transforms.Compose, spectral_dir: Optional[str] = None, simulate_spectral: bool = True, corrupt_images: Optional[List[Tuple[str, str]]] = None, rare_class_threshold: int = 200) -> Tuple[CottonLeafDataset, CottonLeafDataset, CottonLeafDataset, List[str], List[int]]:
    """Loads and splits the dataset into train, validation, and test sets with stratified sampling.

    Args:
        root_path (str): Path to the dataset.
        train_transform (transforms.Compose): Transformations for training data (non-rare classes).
        rare_transform (transforms.Compose): Transformations for rare classes.
        val_test_transform (transforms.Compose): Transformations for validation/test data.
        spectral_dir (str, optional): Path to the spectral dataset directory.
        simulate_spectral (bool): Set to False if RGB data also contains spectral data.
        corrupt_images (List[Tuple[str, str]], optional): List of corrupt image paths to exclude.
        rare_class_threshold (int): Threshold to identify rare classes (based on sample count).

    Returns:
        Tuple[CottonLeafDataset, CottonLeafDataset, CottonLeafDataset, List[str], List[int]]:
            (train_dataset, val_dataset, test_dataset, class_names, rare_classes)
    """
    dataset = datasets.ImageFolder(root=root_path)
    class_names = dataset.classes
    all_samples = dataset.samples

    # Identify rare classes
    class_counts = np.zeros(len(class_names), dtype=int)
    for _, label in all_samples:
        class_counts[label] += 1
    rare_classes = [i for i, count in enumerate(class_counts) if count < rare_class_threshold]
    logger.info(f"Rare classes (less than {rare_class_threshold} samples): {[class_names[i] for i in rare_classes]}")

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

    train_dataset = CottonLeafDataset(
        train_samples, transform=train_transform, rare_transform=rare_transform,
        rare_classes=rare_classes, spectral_dir=spectral_dir, simulate_spectral=simulate_spectral
    )
    val_dataset = CottonLeafDataset(
        val_samples, transform=val_test_transform, rare_transform=None,
        rare_classes=rare_classes, spectral_dir=spectral_dir, simulate_spectral=simulate_spectral
    )
    test_dataset = CottonLeafDataset(
        test_samples, transform=val_test_transform, rare_transform=None,
        rare_classes=rare_classes, spectral_dir=spectral_dir, simulate_spectral=simulate_spectral
    )

    return train_dataset, val_dataset, test_dataset, class_names, rare_classes

# Load Augmented Dataset
def load_augmented_dataset(root_path: str, transform: transforms.Compose, rare_transform: transforms.Compose, rare_classes: List[int], spectral_dir: Optional[str] = None, simulate_spectral: bool = True, corrupt_images: Optional[List[Tuple[str, str]]] = None) -> CottonLeafDataset:
    """Loads the augmented dataset.

    Args:
        root_path (str): Path to the augmented dataset.
        transform (transforms.Compose): Transformations for non-rare classes.
        rare_transform (transforms.Compose): Transformations for rare classes.
        rare_classes (List[int]): List of indices of rare classes.
        spectral_dir (str, optional): Path to the spectral dataset directory.
        simulate_spectral (bool): Set to False if RGB data also contains spectral data.
        corrupt_images (List[Tuple[str, str]], optional): List of corrupt image paths to exclude.

    Returns:
        CottonLeafDataset: Augmented dataset.
    """
    dataset = datasets.ImageFolder(root=root_path)
    if corrupt_images is None:
        corrupt_images = []
    valid_samples = [(path, label) for path, label in dataset.samples if path not in {p for p, _ in corrupt_images}]
    return CottonLeafDataset(
        valid_samples, transform=transform, rare_transform=rare_transform,
        rare_classes=rare_classes, spectral_dir=spectral_dir, simulate_spectral=simulate_spectral
    )

# Visualization Function
def visualize_batch(dataloader: DataLoader, n_samples: int = 16, title: str = "Sample Images", has_multimodal: bool = False, save_path: str = "./visualizations"):
    """Visualizes a batch of images from the dataloader.

    Args:
        dataloader (DataLoader): DataLoader to visualize.
        n_samples (int): Number of samples to display.
        title (str): Title of the visualization.
        has_multimodal (bool): If True, dataset includes spectral data.
        save_path (str): Directory to save the visualization.
    """
    try:
        batch = next(iter(dataloader))
        images = batch[0]
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

# Define Standard Transformations with Class-Specific Augmentations
def get_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Defines transformations for training, validation, and test sets, including rare class-specific augmentations.

    Returns:
        Tuple[transforms.Compose, transforms.Compose, transforms.Compose]: (train_transforms, rare_transforms, val_test_transforms)
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

    rare_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomCrop((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Enhanced for rare classes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, rare_transforms, val_test_transforms

# Utility to Create DataLoaders
def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for train, validation, and test datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
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