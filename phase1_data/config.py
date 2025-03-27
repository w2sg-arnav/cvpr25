# config.py
import logging
import torch

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# Dataset paths
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
ORIGINAL_DATASET_ROOT = f"{DATASET_BASE_PATH}/Original Dataset"
AUGMENTED_DATASET_ROOT = f"{DATASET_BASE_PATH}/Augmented Dataset"

# Dataset constants
# Updated stage map to cover more labels (placeholder; replace with actual mapping)
DEFAULT_STAGE_MAP = {
    0: 'early', 1: 'mid', 2: 'advanced',
    3: 'early', 4: 'mid', 5: 'advanced'  # Temporary mapping for labels 3, 4, 5
}
IMAGE_SIZE = (224, 224)  # Default input size for models
SPECTRAL_SIZE = (224, 224)  # Target size for spectral data