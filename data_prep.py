import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

# Step 1: Set Up the Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define global variables
DATA_ROOT = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
SAVE_PATH = "./data_checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)

# Step 2: Data Analysis Function
def analyze_dataset(data_path):
    """Analyze the dataset to understand class distribution and image properties."""
    print(f"Analyzing dataset at {data_path}...")
    
    # Load the dataset without transformations
    dataset = datasets.ImageFolder(root=data_path)
    
    # Get class names and distribution
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx
    class_counts = [0] * len(class_names)
    
    # Image stats containers
    img_sizes = []
    img_channels = []
    corrupt_images = []
    
    # Analyze each image
    for path, class_idx in dataset.samples:
        class_counts[class_idx] += 1
        
        try:
            with Image.open(path) as img:
                img_sizes.append(img.size)
                if img.mode == 'RGB':
                    img_channels.append(3)
                elif img.mode == 'L':
                    img_channels.append(1)
                else:
                    img_channels.append(0)  # Unknown/other
        except Exception as e:
            corrupt_images.append((path, str(e)))
    
    # Calculate statistics
    total_images = len(dataset)
    class_distribution = {class_names[i]: (count, count/total_images*100) 
                         for i, count in enumerate(class_counts)}
    
    # Print results
    print(f"Total images: {total_images}")
    print("\nClass distribution:")
    for class_name, (count, percentage) in class_distribution.items():
        print(f"{class_name}: {count} images ({percentage:.2f}%)")
    
    print("\nImage statistics:")
    if img_sizes:
        widths, heights = zip(*img_sizes)
        print(f"Width - min: {min(widths)}, max: {max(widths)}, mean: {sum(widths)/len(widths):.1f}")
        print(f"Height - min: {min(heights)}, max: {max(heights)}, mean: {sum(heights)/len(heights):.1f}")
    
    print(f"\nCorrupted images found: {len(corrupt_images)}")
    if corrupt_images:
        print("First 5 corrupted images:")
        for path, error in corrupt_images[:5]:
            print(f"- {path}: {error}")
    
    # Create and save class distribution plot
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, 'class_distribution.png'))
    plt.close()
    
    return {
        'class_names': class_names,
        'class_distribution': class_distribution,
        'corrupt_images': corrupt_images,
        'img_sizes': img_sizes
    }

# Analyze both original and augmented datasets
original_stats = analyze_dataset(os.path.join(DATA_ROOT, "Original Dataset"))
augmented_stats = analyze_dataset(os.path.join(DATA_ROOT, "Augmented Dataset"))

# Step 3: Define Enhanced Data Preprocessing Transformations
# Define more advanced transformations for training
train_transforms = transforms.Compose([
    transforms.Resize((320, 320)),  # Slightly larger for crop
    transforms.RandomCrop((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),  # Add vertical flips
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add slight translations
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),  # Occasional blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # Add random erasing
])

# Validation and test transforms
val_test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 4: Define a Custom Dataset with Cleaning and Handling of Corrupt Images
class CottonLeafDataset(Dataset):
    def __init__(self, samples, transform=None, return_paths=False):
        self.samples = samples
        self.transform = transform
        self.return_paths = return_paths
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Basic quality check
            if min(img.size) < 50:  # Arbitrarily small image
                raise ValueError("Image too small")
                
            if self.transform:
                img = self.transform(img)
                
            if self.return_paths:
                return img, label, img_path
            else:
                return img, label
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = torch.zeros((3, 299, 299))
            if self.return_paths:
                return placeholder, label, img_path
            else:
                return placeholder, label

# Step 5: Load and Split the Original Dataset
def load_and_split_dataset(root_path, train_transform, val_test_transform):
    # Load the dataset without transformations first
    dataset = datasets.ImageFolder(root=root_path)
    class_names = dataset.classes
    
    # Get all samples
    all_samples = dataset.samples
    
    # Count instances per class
    class_counts = {}
    for path, label in all_samples:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Check for corrupt images
    valid_samples = []
    corrupt_samples = []
    
    for path, label in all_samples:
        try:
            with Image.open(path) as img:
                # Basic quality check
                if min(img.size) < 50:
                    corrupt_samples.append((path, label))
                    continue
                valid_samples.append((path, label))
        except Exception as e:
            corrupt_samples.append((path, label))
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Corrupt samples: {len(corrupt_samples)}")
    
    # Split the data
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
    
    # Create dataset splits
    train_samples = [valid_samples[i] for i in train_idx]
    val_samples = [valid_samples[i] for i in val_idx]
    test_samples = [valid_samples[i] for i in test_idx]
    
    # Create datasets with appropriate transforms
    train_dataset = CottonLeafDataset(train_samples, transform=train_transform)
    val_dataset = CottonLeafDataset(val_samples, transform=val_test_transform)
    test_dataset = CottonLeafDataset(test_samples, transform=val_test_transform)
    
    return train_dataset, val_dataset, test_dataset, class_names

# Step 6: Load and Process the Augmented Dataset
def load_augmented_dataset(root_path, transform):
    dataset = datasets.ImageFolder(root=root_path)
    samples = dataset.samples
    
    # Create dataset with transform
    augmented_dataset = CottonLeafDataset(samples, transform=transform)
    
    return augmented_dataset

# Step 7: Visualization Function
def visualize_batch(dataloader, n_samples=16, title="Sample Images"):
    """Visualize a batch of images from a dataloader."""
    batch = next(iter(dataloader))
    images, labels = batch
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_denorm = images * std + mean
    
    # Convert to grid
    grid = vutils.make_grid(images_denorm[:n_samples], nrow=4, padding=2, normalize=True)
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(SAVE_PATH, f'{title.replace(" ", "_").lower()}.png'))
    plt.close()

# Step 8: Multimodal Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, rgb_dataset, spectral_dataset=None, fusion_mode='concatenate'):
        """
        A dataset that combines RGB images with spectral data if available.
        
        Args:
            rgb_dataset: The RGB image dataset
            spectral_dataset: The spectral data dataset (optional)
            fusion_mode: How to combine modalities ('concatenate', 'attention', etc.)
        """
        self.rgb_dataset = rgb_dataset
        self.spectral_dataset = spectral_dataset
        self.fusion_mode = fusion_mode
        
    def __len__(self):
        return len(self.rgb_dataset)
    
    def __getitem__(self, idx):
        rgb_img, label = self.rgb_dataset[idx]
        
        # If spectral data is available, get it
        if self.spectral_dataset is not None:
            try:
                spectral_data, _ = self.spectral_dataset[idx]
                return rgb_img, spectral_data, label
            except:
                # If spectral data access fails, create a placeholder
                spectral_data = torch.zeros_like(rgb_img)
                return rgb_img, spectral_data, label
        
        # If no spectral data, return only RGB
        return rgb_img, label

# Step 9: Load and Split the Data
original_train_dataset, original_val_dataset, original_test_dataset, class_names = load_and_split_dataset(
    os.path.join(DATA_ROOT, "Original Dataset"),
    train_transforms,
    val_test_transforms
)

augmented_dataset = load_augmented_dataset(
    os.path.join(DATA_ROOT, "Augmented Dataset"),
    train_transforms
)

# Step 10: Create Balanced Sampler for Training
def create_balanced_sampler(dataset):
    targets = []
    for _, label in dataset:
        targets.append(label)
    
    targets = torch.tensor(targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

# Step 11: Combine Original and Augmented Training Data
combined_train_dataset = ConcatDataset([original_train_dataset, augmented_dataset])

# Step 12: Create Multimodal Datasets
train_multimodal_dataset = MultimodalDataset(combined_train_dataset)
val_multimodal_dataset = MultimodalDataset(original_val_dataset)
test_multimodal_dataset = MultimodalDataset(original_test_dataset)

# Create balanced sampler for training
train_sampler = create_balanced_sampler(combined_train_dataset)

# Step 13: Create DataLoaders
train_loader = DataLoader(
    train_multimodal_dataset,
    batch_size=32,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_multimodal_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_multimodal_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Step 14: Visualize a Batch from Each Split
visualize_batch(train_loader, title="Training Samples")
visualize_batch(val_loader, title="Validation Samples")
visualize_batch(test_loader, title="Test Samples")

# Step 15: Save the Preprocessed Data
torch.save({
    'train_dataset': train_multimodal_dataset,
    'val_dataset': val_multimodal_dataset,
    'test_dataset': test_multimodal_dataset,
    'class_names': class_names,
    'original_stats': original_stats,
    'augmented_stats': augmented_stats
}, os.path.join(SAVE_PATH, 'preprocessed_data.pth'))

# Print dataset statistics
print(f"\nTraining set size: {len(train_multimodal_dataset)}")
print(f"Validation set size: {len(val_multimodal_dataset)}")
print(f"Test set size: {len(test_multimodal_dataset)}")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")