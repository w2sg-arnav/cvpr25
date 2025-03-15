# dataset_utils.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CottonLeafDataset(Dataset):
    """
    A custom dataset class for handling cotton leaf images with optional multimodal (e.g., spectral) data.

    Args:
        samples (List[Tuple[str, int]]): List of tuples containing (image_path, label) pairs.
        transform (Optional[transforms.Compose]): Optional transform to apply to RGB images.
        return_paths (bool): If True, return image paths along with data and labels.
        spectral_path (Optional[str]): Path to the spectral dataset directory (if available).

    Attributes:
        samples (List[Tuple[str, int]]): Stored sample paths and labels.
        transform (Optional[transforms.Compose]): Transformation pipeline.
        return_paths (bool): Flag to include paths in output.
        spectral_path (Optional[str]): Path to spectral data.
        spectral_data (Optional[List[Tuple[str, int]]]): Loaded spectral data samples.
        has_multimodal (bool): Indicates if multimodal data is available or simulated.
    """
    
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        return_paths: bool = False,
        spectral_path: Optional[str] = None
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.return_paths = return_paths
        self.spectral_path = spectral_path
        self.spectral_data = None
        self.has_multimodal = False

        # Initialize spectral data
        if spectral_path and os.path.exists(spectral_path):
            try:
                self.spectral_data = datasets.ImageFolder(root=spectral_path).samples
                self.has_multimodal = True
                logger.info(f"Spectral data loaded from {spectral_path} with {len(self.spectral_data)} samples.")
            except Exception as e:
                logger.warning(f"Failed to load spectral data from {spectral_path}: {e}. Falling back to simulation.")
                self.spectral_data = [(s[0], s[1]) for s in samples]  # Fallback to sample list
                self.has_multimodal = True
        elif not spectral_path:
            logger.warning("Spectral data not found. Simulating grayscale as placeholder.")
            self.spectral_data = [(s[0], s[1]) for s in samples]  # Placeholder
            self.has_multimodal = True

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple containing (RGB image, spectral data, label, [path]) based on configuration.
        """
        img_path, label = self.samples[idx]

        try:
            # Load the image as a PIL Image
            img = Image.open(img_path).convert('RGB')
            if min(img.size) < 50:
                raise ValueError(f"Image {img_path} is too small: {img.size}")

            # Simulate or load spectral data using the original PIL image
            spectral = None
            if self.has_multimodal:
                if self.spectral_path and os.path.exists(self.spectral_path) and self.spectral_data:
                    spectral_path, _ = self.spectral_data[idx % len(self.spectral_data)]  # Handle length mismatch
                    spectral = Image.open(spectral_path).convert('L')
                else:
                    # Simulate spectral data as grayscale from the original RGB image
                    spectral = img.convert('L')
                spectral = transforms.Resize((299, 299))(spectral)
                spectral = transforms.ToTensor()(spectral).squeeze(0)  # Shape: [299, 299]

            # Apply transform to the RGB image
            if self.transform:
                img = self.transform(img)

            # Prepare output based on configuration
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

def analyze_dataset(data_path: str, dataset_type: str = "Original", save_path: str = "./phase1_checkpoints") -> Dict[str, Any]:
    """
    Analyze the dataset to understand class distribution and image properties.

    Args:
        data_path (str): Path to the dataset directory.
        dataset_type (str): Type of dataset (e.g., "Original", "Augmented").
        save_path (str): Directory to save analysis plots.

    Returns:
        Dict containing class names, distribution, corrupt images, and image sizes.
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

    return {
        'class_names': class_names,
        'class_distribution': class_distribution,
        'corrupt_images': corrupt_images,
        'img_sizes': img_sizes
    }

def load_and_split_dataset(
    root_path: str,
    train_transform: transforms.Compose,
    val_test_transform: transforms.Compose,
    spectral_path: Optional[str] = None
) -> Tuple[CottonLeafDataset, CottonLeafDataset, CottonLeafDataset, List[str]]:
    """
    Load and split the dataset into train, validation, and test sets.

    Args:
        root_path (str): Path to the dataset directory.
        train_transform (transforms.Compose): Transform for training data.
        val_test_transform (transforms.Compose): Transform for validation and test data.
        spectral_path (Optional[str]): Path to the spectral dataset directory.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, class_names).
    """
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

# Note: The load_augmented_dataset function and visualization function are omitted here for brevity.
# You can add them back if needed, following the same pattern with type hints and documentation.