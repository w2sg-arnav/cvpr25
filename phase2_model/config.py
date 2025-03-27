# config.py
import logging
import torch

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# Model configurations
PATCH_SIZE = 16  # Initial patch size for HVT
EMBED_DIM = 168  # Embedding dimension
DEPTHS = [2, 2, 6, 2]  # Number of layers in each stage
NUM_HEADS = [3, 6, 12, 24]  # Number of attention heads per stage
WINDOW_SIZE = 7  # Window size for Swin Transformer
NUM_CLASSES = 6  # Adjust based on SAR-CLD-2024 dataset (e.g., 6 classes)
SPECTRAL_CHANNELS = 1  # Number of spectral channels (e.g., NDVI)
PROGRESSIVE_RESOLUTIONS = [(128, 128), (224, 224), (384, 384)]  # Progressive training resolutions