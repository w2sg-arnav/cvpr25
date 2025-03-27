# phase3_pretraining/config.py
import logging
import torch
from phase2_model.config import *  # Import Phase 2 config

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# Pretraining-specific configurations
PROJECTION_DIM = 128  # Dimension of the projection head output
PRETRAIN_LR = 3e-4  # Learning rate for pretraining
PRETRAIN_EPOCHS = 10  # Number of pretraining epochs (for testing; increase for real training)
PRETRAIN_BATCH_SIZE = 4  # Batch size for pretraining
TEMPERATURE = 0.5  # Temperature parameter for InfoNCE loss