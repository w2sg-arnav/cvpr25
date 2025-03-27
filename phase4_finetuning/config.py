# phase4_finetuning/config.py
import logging
import torch
from phase3_pretraining.config import *  # Import Phase 3 config

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# Fine-tuning-specific configurations
FINETUNE_LR = 3e-4  # Learning rate for fine-tuning
FINETUNE_EPOCHS = 30  # Increased to 30 epochs
FINETUNE_BATCH_SIZE = 32  # Batch size for fine-tuning
PRETRAINED_MODEL_PATH = "pretrained_hvt.pth"  # Path to pretrained weights
NUM_CLASSES = 7  # Number of classes in SAR-CLD-2024 dataset