# phase4_finetuning/main.py
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory (cvpr25) to sys.path before any other imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
logging.info(f"Parent directory added to sys.path: {parent_dir}")
logging.info(f"sys.path: {sys.path}")

# Now perform imports
import torch
from config import PROGRESSIVE_RESOLUTIONS, FINETUNE_BATCH_SIZE, PRETRAINED_MODEL_PATH, NUM_CLASSES
from models.hvt import DiseaseAwareHVT
from models.baseline import InceptionV3Baseline
from utils.augmentations import FinetuneAugmentation
from finetune.trainer import Finetuner

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Use the largest resolution for fine-tuning
    img_size = PROGRESSIVE_RESOLUTIONS[-1]  # (384, 384)
    
    # Initialize models
    hvt_model = DiseaseAwareHVT(img_size=img_size)
    baseline_model = InceptionV3Baseline()
    
    # Load pretrained weights for HVT
    hvt_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    logging.info(f"Loaded pretrained weights for DiseaseAwareHVT from {PRETRAINED_MODEL_PATH}")
    
    # Initialize augmentations
    augmentations = FinetuneAugmentation(img_size)
    
    # Initialize finetuners
    hvt_finetuner = Finetuner(hvt_model, augmentations, device)
    baseline_finetuner = Finetuner(baseline_model, augmentations, device)
    
    # Dummy data for testing (replace with real dataset later)
    rgb = torch.randn(FINETUNE_BATCH_SIZE, 3, img_size[0], img_size[1])
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to [0, 1]
    spectral = torch.randn(FINETUNE_BATCH_SIZE, 1, img_size[0], img_size[1])
    spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min())  # Normalize to [0, 1]
    labels = torch.randint(0, NUM_CLASSES, (FINETUNE_BATCH_SIZE,))  # Random labels (0 to NUM_CLASSES-1)
    
    # Fine-tune HVT
    logging.info("Fine-tuning DiseaseAwareHVT...")
    hvt_finetuner.finetune(rgb, spectral, labels)
    hvt_finetuner.save_model("finetuned_hvt.pth")
    
    # Fine-tune baseline
    logging.info("Fine-tuning InceptionV3Baseline...")
    baseline_finetuner.finetune(rgb, spectral, labels)
    baseline_finetuner.save_model("finetuned_baseline.pth")

if __name__ == "__main__":
    main()