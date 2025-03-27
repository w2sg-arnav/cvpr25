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
from torch.utils.data import DataLoader
from config import PROGRESSIVE_RESOLUTIONS, FINETUNE_BATCH_SIZE, PRETRAINED_MODEL_PATH, NUM_CLASSES
from models.hvt import DiseaseAwareHVT
from models.baseline import InceptionV3Baseline
from utils.augmentations import FinetuneAugmentation
from finetune.trainer import Finetuner
from dataset import SARCLD2024Dataset

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Use the largest resolution for fine-tuning
    img_size = PROGRESSIVE_RESOLUTIONS[-1]  # (384, 384)
    
    # Load dataset
    dataset_root = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    # Verify dataset_root exists
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}. Please check the path.")
    
    train_dataset = SARCLD2024Dataset(dataset_root, img_size, split="train", train_split=0.8)
    val_dataset = SARCLD2024Dataset(dataset_root, img_size, split="val", train_split=0.8)
    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    
    # Log dataset sizes
    logging.info(f"Training dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Update NUM_CLASSES in config based on dataset
    num_classes = len(train_dataset.get_class_names())
    if num_classes != NUM_CLASSES:
        logging.warning(f"NUM_CLASSES in config ({NUM_CLASSES}) does not match dataset classes ({num_classes}). Updating NUM_CLASSES.")
        globals()['NUM_CLASSES'] = num_classes
    
    # Initialize models AFTER updating NUM_CLASSES
    hvt_model = DiseaseAwareHVT(img_size=img_size)
    baseline_model = InceptionV3Baseline()
    
    # Load pretrained weights for HVT, ignoring the classifier head mismatch
    pretrained_dict = torch.load(PRETRAINED_MODEL_PATH)
    model_dict = hvt_model.state_dict()
    
    # Filter out the classifier head (head.weight and head.bias) from the pretrained dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    # Update the model's state dict with the pretrained weights
    model_dict.update(pretrained_dict)
    hvt_model.load_state_dict(model_dict)
    logging.info(f"Loaded pretrained weights for DiseaseAwareHVT from {PRETRAINED_MODEL_PATH} (classifier head excluded due to class mismatch)")
    
    # Initialize augmentations
    augmentations = FinetuneAugmentation(img_size)
    
    # Initialize finetuners
    hvt_finetuner = Finetuner(hvt_model, augmentations, device)
    baseline_finetuner = Finetuner(baseline_model, augmentations, device)
    
    # Fine-tune HVT
    logging.info("Fine-tuning DiseaseAwareHVT...")
    for epoch in range(5):  # Use a fixed number of epochs for simplicity
        # Training loop
        hvt_finetuner.model.train()
        train_loss = 0.0
        for batch_idx, (rgb, spectral, labels) in enumerate(train_loader):
            loss = hvt_finetuner.train_step(rgb, spectral, labels)
            train_loss += loss
        train_loss /= len(train_loader)
        
        # Validation loop
        val_metrics = {"accuracy": 0.0, "f1": 0.0}
        hvt_finetuner.model.eval()
        with torch.no_grad():
            for rgb, spectral, labels in val_loader:
                metrics = hvt_finetuner.evaluate(rgb, spectral, labels)
                val_metrics["accuracy"] += metrics["accuracy"]
                val_metrics["f1"] += metrics["f1"]
        val_metrics["accuracy"] /= len(val_loader)
        val_metrics["f1"] /= len(val_loader)
        
        logging.info(f"Epoch {epoch+1}/5, Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
    
    hvt_finetuner.save_model("finetuned_hvt.pth")
    
    # Fine-tune baseline
    logging.info("Fine-tuning InceptionV3Baseline...")
    for epoch in range(5):  # Use a fixed number of epochs for simplicity
        # Training loop
        baseline_finetuner.model.train()
        train_loss = 0.0
        for batch_idx, (rgb, spectral, labels) in enumerate(train_loader):
            loss = baseline_finetuner.train_step(rgb, spectral, labels)
            train_loss += loss
        train_loss /= len(train_loader)
        
        # Validation loop
        val_metrics = {"accuracy": 0.0, "f1": 0.0}
        baseline_finetuner.model.eval()
        with torch.no_grad():
            for rgb, spectral, labels in val_loader:
                metrics = baseline_finetuner.evaluate(rgb, spectral, labels)
                val_metrics["accuracy"] += metrics["accuracy"]
                val_metrics["f1"] += metrics["f1"]
        val_metrics["accuracy"] /= len(val_loader)
        val_metrics["f1"] /= len(val_loader)
        
        logging.info(f"Epoch {epoch+1}/5, Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
    
    baseline_finetuner.save_model("finetuned_baseline.pth")

if __name__ == "__main__":
    main()