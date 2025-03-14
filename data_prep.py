import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch

# Step 1: Set Up the Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Define Data Preprocessing Transformations
# Separate transforms for training (with augmentation) and validation/test (no augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 3: Load the Original Dataset with Different Transforms for Training Portion
# Load the full original dataset without immediate transformation
original_dataset = datasets.ImageFolder(root=os.path.join("/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection", "Original Dataset"),
                                       transform=None)  # No transform yet

# Split the original dataset first, then apply transforms
total_size = len(original_dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Split without transformation
original_train_dataset, original_val_dataset, original_test_dataset = random_split(
    original_dataset, [train_size, val_size, test_size]
)

# Apply transforms after splitting
train_dataset_transformed = [(train_transforms(img), label) for img, label in original_train_dataset]
val_dataset_transformed = [(val_test_transforms(img), label) for img, label in original_val_dataset]
test_dataset_transformed = [(val_test_transforms(img), label) for img, label in original_test_dataset]

# Step 4: Load the Augmented Dataset
augmented_dataset = datasets.ImageFolder(root=os.path.join("/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection", "Augmented Dataset"),
                                        transform=train_transforms)  # Use train transforms for augmentation

# Step 5: Combine and Create DataLoaders
# Convert transformed lists to a custom dataset format if needed, or use ConcatDataset with care
combined_train_dataset = train_dataset_transformed + [(img, label) for img, label in augmented_dataset]  # Concatenate lists

# Create a custom Dataset class to handle the list of (image, label) tuples
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

combined_train_dataset = CustomDataset(combined_train_dataset)
val_dataset = CustomDataset(val_dataset_transformed)
test_dataset = CustomDataset(test_dataset_transformed)

# Create DataLoader objects
train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print sizes
print(f"Training set size: {len(combined_train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Step 6: Visualize a Sample (Optional)
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Get a batch from the training loader
images, labels = next(iter(train_loader))
imshow(images[0])
print(f"Label: {labels[0].item()} ({original_dataset.classes[labels[0].item()]})")

# Step 7: Prepare for Multimodal Integration
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_dataset, spectral_data=None):
        self.rgb_dataset = rgb_dataset
        self.spectral_data = spectral_data

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, idx):
        rgb_img, label = self.rgb_dataset[idx]
        if self.spectral_data is not None:
            spectral_img = torch.zeros_like(rgb_img)  # Placeholder
            return rgb_img, spectral_img, label
        return rgb_img, label

multimodal_train_dataset = MultimodalDataset(combined_train_dataset)
multimodal_val_dataset = MultimodalDataset(val_dataset)
multimodal_test_dataset = MultimodalDataset(test_dataset)

# Update DataLoaders
train_loader = DataLoader(multimodal_train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(multimodal_val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(multimodal_test_dataset, batch_size=32, shuffle=False)

# Step 8: Save and Verify the Setup
torch.save({
    'train_dataset': combined_train_dataset,
    'val_dataset': val_dataset,
    'test_dataset': test_dataset
}, 'dataset_splits.pth')

for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1} - Images shape: {images.shape}, Labels: {labels}")
    if batch_idx == 0: break