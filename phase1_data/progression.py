# progression.py
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

class DiseaseProgressionSimulator:
    """Simulate disease progression stages for cotton leaves."""
    
    def __init__(self):
        self.transforms = {
            'early': T.ColorJitter(brightness=0.1, contrast=0.1),
            'mid': T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5)
            ]),
            'advanced': T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                T.RandomApply([T.GaussianBlur(5, sigma=(0.5, 2.0))], p=0.7),
                lambda x: self._add_lesions(x)
            ])
        }
    
    def _add_lesions(self, img: Image.Image) -> Image.Image:
        """Simulate lesions by adding dark spots."""
        img_np = np.array(img).astype(np.float32)
        h, w = img_np.shape[:2]
        for _ in range(np.random.randint(1, 5)):  # 1-4 lesions
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(5, 15)
            cv2.circle(img_np, (x, y), radius, (50, 30, 20), -1)  # Dark brown spots
        return Image.fromarray(img_np.astype(np.uint8))
    
    def apply(self, img: Image.Image, stage: str) -> Image.Image:
        """Apply stage-specific transformations."""
        if stage not in self.transforms:
            return img
        return self.transforms[stage](img)