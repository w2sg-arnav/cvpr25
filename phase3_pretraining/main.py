# phase3_pretraining/main.py
import sys
import os

# Add the parent directory (cvpr25) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from typing import Tuple
from config import PROGRESSIVE_RESOLUTIONS, PRETRAIN_BATCH_SIZE
from models.hvt import DiseaseAwareHVT
from utils.augmentations import SimCLRAugmentation
from utils.losses import InfoNCELoss
from pretrain.trainer import Pretrainer
import logging

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Use the largest resolution for pretraining
    img_size = PROGRESSIVE_RESOLUTIONS[-1]  # (384, 384)
    
    # Initialize model
    hvt_model = DiseaseAwareHVT(img_size=img_size)
    
    # Initialize augmentations and loss
    augmentations = SimCLRAugmentation(img_size)
    loss_fn = InfoNCELoss()
    
    # Initialize pretrainer
    pretrainer = Pretrainer(hvt_model, augmentations, loss_fn, device)
    
    # Dummy data for testing (replace with real dataset later)
    rgb = torch.randn(PRETRAIN_BATCH_SIZE, 3, img_size[0], img_size[1])
    spectral = torch.randn(PRETRAIN_BATCH_SIZE, 1, img_size[0], img_size[1])
    
    # Pretrain the model
    pretrainer.pretrain(rgb, spectral)
    
    # Save the pretrained model
    pretrainer.save_model("pretrained_hvt.pth")

if __name__ == "__main__":
    main()