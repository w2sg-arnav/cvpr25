# dataset.py
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Tuple, Optional, Dict
import logging

from config import DEFAULT_STAGE_MAP, SPECTRAL_SIZE
from progression import DiseaseProgressionSimulator

class CottonLeafDataset(Dataset):
    """Custom dataset for cotton leaf disease detection with support for original and augmented datasets."""
    
    def __init__(self, root_dir: str, transform: Optional[T.Compose] = None, 
                 stage_map: Optional[Dict[int, str]] = None, apply_progression: bool = False):
        """
        Args:
            root_dir (str): Path to dataset root (e.g., 'Original Dataset/' or 'Augmented Dataset/').
            transform (T.Compose, optional): Transformations for RGB images.
            stage_map (Dict[int, str], optional): Mapping of label integers to stages.
            apply_progression (bool): If True, apply disease progression simulation (useful for original dataset).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.stage_map = stage_map or DEFAULT_STAGE_MAP
        self.apply_progression = apply_progression
        self.progression_simulator = DiseaseProgressionSimulator() if apply_progression else None
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()
    
    def _load_images(self) -> list:
        """Load all image paths and labels, skipping corrupt files."""
        images = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            label = self.class_to_idx[cls]
            for img_name in os.listdir(cls_dir):
                if not img_name.endswith('.jpg'):
                    continue
                img_path = cls_dir / img_name
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Check for corruption
                    spectral_path = cls_dir / f"{img_name.split('.')[0]}_spectral.npy"
                    has_spectral = spectral_path.exists()
                    images.append((str(img_path), str(spectral_path) if has_spectral else None, label))
                except (IOError, SyntaxError) as e:
                    logging.warning(f"Skipping corrupt image: {img_path} - {e}")
        logging.info(f"Loaded {len(images)} valid images from {self.root_dir}")
        return images
    
    def _simulate_ndvi(self, rgb_img: Image.Image) -> torch.Tensor:
        """Simulate NDVI from RGB when spectral data is unavailable."""
        rgb = np.array(rgb_img.convert('RGB')) / 255.0  # Normalize to [0, 1]
        nir = rgb[:, :, 0] * 0.5 + rgb[:, :, 1] * 0.3 + rgb[:, :, 2] * 0.2  # Proxy NIR
        red = rgb[:, :, 0]
        ndvi = (nir - red) / (nir + red + 1e-6)  # Avoid division by zero
        ndvi_tensor = torch.tensor(ndvi, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        # Resize NDVI to match RGB resolution
        resize = T.Resize(SPECTRAL_SIZE, interpolation=T.InterpolationMode.BILINEAR)
        return resize(ndvi_tensor)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, str]:
        """Return RGB tensor, spectral/NDVI tensor, label, and stage."""
        img_path, spectral_path, label = self.images[idx]
        rgb_img = Image.open(img_path).convert('RGB')
        
        # Apply disease progression simulation if enabled
        stage = self.stage_map.get(label, 'unknown')
        if self.apply_progression and self.progression_simulator:
            rgb_img = self.progression_simulator.apply(rgb_img, stage)
        
        # Load or simulate spectral data
        if spectral_path:
            spectral_data = np.load(spectral_path)  # Assume [H, W] or [C, H, W]
            if spectral_data.ndim == 2:
                spectral_data = spectral_data[np.newaxis, ...]  # Add channel dim
            spectral_tensor = torch.tensor(spectral_data, dtype=torch.float32)
            # Resize spectral data to match RGB resolution
            resize = T.Resize(SPECTRAL_SIZE, interpolation=T.InterpolationMode.BILINEAR)
            spectral_tensor = resize(spectral_tensor)
        else:
            spectral_tensor = self._simulate_ndvi(rgb_img)
        
        # Apply transforms to RGB
        if self.transform:
            rgb_tensor = self.transform(rgb_img)
        else:
            rgb_tensor = T.ToTensor()(rgb_img)
        
        return rgb_tensor, spectral_tensor, label, stage