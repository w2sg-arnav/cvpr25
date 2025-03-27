import torch
from torchvision import transforms
import logging

# Use a module-specific logger
logger = logging.getLogger(__name__)

class SimCLRAugmentation:
    def __init__(self, img_size: tuple, device: str = "cuda"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),  # Stronger crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),  # Stronger color jitter
            transforms.RandomGrayscale(p=0.2),  # Add grayscale
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=img_size[0]//10*2+1, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
        ])
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of images with shape [B, C, H, W]
        batch_size = x.shape[0]
        # Apply augmentations to each image in the batch
        augmented = torch.stack([self.transform(transforms.ToPILImage()(img)) for img in x])
        # Move to device
        augmented = augmented.to(self.device)
        # Ensure the output requires gradients if the input does
        if x.requires_grad and not augmented.requires_grad:
            logger.warning("Augmented output does not require gradients, but input does. This may break the computation graph.")
        return augmented