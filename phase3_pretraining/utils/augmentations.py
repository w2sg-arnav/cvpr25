import torch
import torchvision.transforms.functional as TF
import logging
from config import DEVICE

logger = logging.getLogger(__name__)

class SimCLRAugmentation:
    def __init__(self, img_size: tuple):
        self.img_size = img_size
        self.device = DEVICE  # Augmentations happen on the specified device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device) # Move input to device before augmentation

        x = TF.resize(x, self.img_size)
        if torch.rand(1) < 0.5:
            x = TF.hflip(x)

        brightness_factor = torch.rand(1).item() * 1.6 + 0.2
        x = TF.adjust_brightness(x, brightness_factor)

        contrast_factor = torch.rand(1).item() * 1.6 + 0.2
        x = TF.adjust_contrast(x, contrast_factor)

        saturation_factor = torch.rand(1).item() * 1.6 + 0.2
        x = TF.adjust_saturation(x, saturation_factor)

        hue_factor = torch.rand(1).item() * 0.4 - 0.2
        x = TF.adjust_hue(x, hue_factor)

        if torch.rand(1) < 0.2:
            x = TF.rgb_to_grayscale(x, num_output_channels=3)

        x = TF.gaussian_blur(x, kernel_size=self.img_size[0]//10*2+1, sigma=(0.1, 2.0))
        return x