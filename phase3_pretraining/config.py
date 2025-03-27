import torch
from phase2_model.config import *  # Import Phase 2 config

# Random seed for reproducibility
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# Pretraining-specific configurations
PROJECTION_DIM = 256  # Updated to match the projection head output in hvt.py
PRETRAIN_LR = 8e-3  # Slightly increased learning rate for controlled descent
PRETRAIN_EPOCHS = 50  # Number of epochs for pretraining
PRETRAIN_BATCH_SIZE = 192  # Batch size to avoid OOM
TEMPERATURE = 0.2  # Adjusted temperature for InfoNCE loss (increased to 0.2)
ACCUM_STEPS = 1  # No accumulation needed with larger batch size

PROGRESSIVE_RESOLUTIONS = [(128, 128), (256, 256), (384, 384)]
FINETUNE_BATCH_SIZE = 32  # Define the batch size here
NUM_CLASSES = 7  # Number of classes in SAR-CLD-2024 dataset
NUM_EPOCHS = 50  # Number of epochs for pretraining