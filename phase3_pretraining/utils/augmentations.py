# phase3_pretraining/utils/augmentations.py
import torch
import torchvision.transforms as T

class SimCLRAugmentation:
    """Data augmentation pipeline for SimCLR."""
    
    def __init__(self, img_size: tuple):
        # Augmentations for RGB
        self.rgb_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations for spectral (simpler, as spectral data is typically single-channel)
        self.spectral_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.5], std=[0.5])  # Adjust based on your spectral data distribution
        ])
    
    def __call__(self, rgb: torch.Tensor, spectral: torch.Tensor):
        """
        Apply augmentations to generate two views of the same sample.
        
        Args:
            rgb (torch.Tensor): RGB input [batch, 3, H, W].
            spectral (torch.Tensor): Spectral input [batch, 1, H, W].
        
        Returns:
            tuple: (rgb_view1, spectral_view1, rgb_view2, spectral_view2)
        """
        # Generate two views for RGB
        rgb_view1 = torch.stack([self.rgb_transform(img) for img in rgb])
        rgb_view2 = torch.stack([self.rgb_transform(img) for img in rgb])
        
        # Generate two views for spectral
        spectral_view1 = torch.stack([self.spectral_transform(img) for img in spectral])
        spectral_view2 = torch.stack([self.spectral_transform(img) for img in spectral])
        
        return rgb_view1, spectral_view1, rgb_view2, spectral_view2