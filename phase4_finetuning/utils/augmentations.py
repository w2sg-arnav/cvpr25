# phase4_finetuning/utils/augmentations.py
import torch
import torchvision.transforms as T

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
        # Apply augmentations to RGB data only
        rgb = self.transform(rgb)
        return rgb