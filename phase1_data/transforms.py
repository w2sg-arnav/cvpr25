# transforms.py
import torchvision.transforms as T
from config import IMAGE_SIZE

def get_transforms(train: bool = True) -> T.Compose:
    """Return augmentation pipeline for training or evaluation."""
    if train:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Simulate leaf movement
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Weather effects
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
        ])
    else:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])