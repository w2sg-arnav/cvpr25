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
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from torch.optim.lr_scheduler import CosineAnnealingLR  # Updated scheduler
from config import PROGRESSIVE_RESOLUTIONS, FINETUNE_BATCH_SIZE, PRETRAINED_MODEL_PATH, NUM_CLASSES
from models.hvt import DiseaseAwareHVT
from models.efficientnet import EfficientNetBaseline
from utils.augmentations import FinetuneAugmentation
from finetune.trainer import Finetuner
from dataset import SARCLD2024Dataset

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    logging.info("Enabled CUDA optimizations (cudnn.benchmark)")
    
    # Use the largest resolution for fine-tuning
    img_size = PROGRESSIVE_RESOLUTIONS[-1]  # (384, 384)
    
    # Load dataset with optimized DataLoader
    dataset_root = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}. Please check the path.")
    
    train_dataset = SARCLD2024Dataset(dataset_root, img_size, split="train", train_split=0.8)
    val_dataset = SARCLD2024Dataset(dataset_root, img_size, split="val", train_split=0.8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Log dataset sizes
    logging.info(f"Training dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Update NUM_CLASSES in config based on dataset
    num_classes = len(train_dataset.get_class_names())
    if num_classes != NUM_CLASSES:
        logging.warning(f"NUM_CLASSES in config ({NUM_CLASSES}) does not match dataset classes ({num_classes}). Updating NUM_CLASSES.")
        globals()['NUM_CLASSES'] = num_classes
    
    # Get class weights for weighted loss
    class_weights = train_dataset.get_class_weights()
    logging.info(f"Class weights for weighted loss: {class_weights}")
    
    # Initialize models AFTER updating NUM_CLASSES
    hvt_model = DiseaseAwareHVT(img_size=img_size).to(device)
    baseline_model = EfficientNetBaseline().to(device)
    
    # Freeze early layers of EfficientNetBaseline (e.g., first 5 blocks)
    for name, param in baseline_model.named_parameters():
        if "features.0" in name or "features.1" in name or "features.2" in name or "features.3" in name or "features.4" in name:
            param.requires_grad = False
    logging.info("Froze early layers (features.0 to features.4) of EfficientNetBaseline")
    
    # Load pretrained weights for HVT, ignoring the classifier head mismatch
    pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    model_dict = hvt_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    hvt_model.load_state_dict(model_dict)
    logging.info(f"Loaded pretrained weights for DiseaseAwareHVT from {PRETRAINED_MODEL_PATH} (classifier head excluded due to class mismatch)")
    
    # Initialize augmentations
    augmentations = FinetuneAugmentation(img_size)
    
    # Initialize finetuners with class weights and different learning rates
    hvt_finetuner = Finetuner(hvt_model, augmentations, device, class_weights=class_weights)
    baseline_finetuner = Finetuner(baseline_model, augmentations, device, class_weights=class_weights)
    # Adjust learning rate for EfficientNetBaseline
    for param_group in baseline_finetuner.optimizer.param_groups:
        param_group['lr'] = 5e-5  # Reduced learning rate for better stability
    logging.info("Set learning rate for EfficientNetBaseline to 5e-5")
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize learning rate scheduler with CosineAnnealingLR
    hvt_scheduler = CosineAnnealingLR(hvt_finetuner.optimizer, T_max=30, eta_min=1e-6)
    baseline_scheduler = CosineAnnealingLR(baseline_finetuner.optimizer, T_max=30, eta_min=1e-6)
    
    # Fine-tune HVT with mixed precision
    logging.info("Fine-tuning DiseaseAwareHVT with mixed precision...")
    best_val_acc = 0.0
    best_model_path = "best_finetuned_hvt.pth"
    for epoch in range(30):
        # Training loop
        hvt_finetuner.model.train()
        train_loss = 0.0
        for batch_idx, (rgb, labels) in enumerate(train_loader):
            rgb, labels = rgb.to(device), labels.to(device)
            
            # Mixed precision training
            with autocast():
                loss = hvt_finetuner.train_step(rgb, labels)
            
            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(hvt_finetuner.optimizer)
            scaler.update()
            hvt_finetuner.optimizer.zero_grad()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation loop
        val_metrics = {"accuracy": 0.0, "f1": 0.0}
        hvt_finetuner.model.eval()
        with torch.no_grad():
            for rgb, labels in val_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                metrics = hvt_finetuner.evaluate(rgb, labels)
                val_metrics["accuracy"] += metrics["accuracy"]
                val_metrics["f1"] += metrics["f1"]
        val_metrics["accuracy"] /= len(val_loader)
        val_metrics["f1"] /= len(val_loader)
        
        # Step the scheduler
        hvt_scheduler.step()
        
        # Early stopping: save the model with the best validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            hvt_finetuner.save_model(best_model_path)
            logging.info(f"New best model saved with Val Accuracy: {best_val_acc:.4f}")
        
        logging.info(f"Epoch {epoch+1}/30, Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
    
    hvt_finetuner.save_model("finetuned_hvt.pth")
    
    # Fine-tune baseline with mixed precision
    logging.info("Fine-tuning EfficientNetBaseline with mixed precision...")
    best_val_acc = 0.0
    best_model_path = "best_finetuned_baseline.pth"
    for epoch in range(30):
        # Training loop
        baseline_finetuner.model.train()
        train_loss = 0.0
        for batch_idx, (rgb, labels) in enumerate(train_loader):
            rgb, labels = rgb.to(device), labels.to(device)
            
            # Mixed precision training
            with autocast():
                loss = baseline_finetuner.train_step(rgb, labels)
            
            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(baseline_finetuner.optimizer)
            scaler.update()
            baseline_finetuner.optimizer.zero_grad()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation loop
        val_metrics = {"accuracy": 0.0, "f1": 0.0}
        baseline_finetuner.model.eval()
        with torch.no_grad():
            for rgb, labels in val_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                metrics = baseline_finetuner.evaluate(rgb, labels)
                val_metrics["accuracy"] += metrics["accuracy"]
                val_metrics["f1"] += metrics["f1"]
        val_metrics["accuracy"] /= len(val_loader)
        val_metrics["f1"] /= len(val_loader)
        
        # Step the scheduler
        baseline_scheduler.step()
        
        # Early stopping: save the model with the best validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            baseline_finetuner.save_model(best_model_path)
            logging.info(f"New best model saved with Val Accuracy: {best_val_acc:.4f}")
        
        logging.info(f"Epoch {epoch+1}/30, Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
    
    baseline_finetuner.save_model("finetuned_baseline.pth")

if __name__ == "__main__":
    main()