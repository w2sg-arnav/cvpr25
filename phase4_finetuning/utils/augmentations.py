# phase4_finetuning/utils/augmentations.py
import torch
import torchvision.transforms as T

class FinetuneAugmentation:
    """Data augmentation pipeline for fine-tuning."""
    
    def __init__(self, img_size: tuple):
        # Augmentations for RGB
        self.rgb_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations for spectral
        self.spectral_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __call__(self, rgb: torch.Tensor, spectral: torch.Tensor):
        """
        Apply augmentations to the input.
        
        Args:
            rgb (torch.Tensor): RGB input [batch, 3, H, W].
            spectral (torch.Tensor): Spectral input [batch, 1, H, W].
        
        Returns:
            tuple: (augmented_rgb, augmented_spectral)
        """
        rgb = torch.stack([self.rgb_transform(img) for img in rgb])
        spectral = torch.stack([self.spectral_transform(img) for img in spectral])
        return rgb, spectral