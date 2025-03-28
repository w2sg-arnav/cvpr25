import torch

# Reproducibility
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# Pretraining Hyperparameters
PRETRAIN_LR = 3e-4
PRETRAIN_EPOCHS = 50
PRETRAIN_BATCH_SIZE = 32 # Reduced batch size to avoid OOM
ACCUM_STEPS = 1 # Reduced Accum steps

TEMPERATURE = 0.07 # Tune me!
PROJECTION_DIM = 256

# Image and Dataset Settings
PROGRESSIVE_RESOLUTIONS = [(128, 128), (256, 256), (384, 384)]
NUM_CLASSES = 7

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
LOG_FILE = "pretrain.log"