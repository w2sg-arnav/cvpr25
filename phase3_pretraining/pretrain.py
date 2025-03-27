import logging
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

# Import the logging setup and configure it once
from utils.logging_setup import setup_logging
setup_logging(log_file="pretrain.log")

# Use a module-specific logger
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append("/teamspace/studios/this_studio/cvpr25/")

# Updated imports
from phase4_finetuning.dataset import SARCLD2024Dataset  # Custom dataset class
from models.hvt import DiseaseAwareHVT  # Custom model class
from utils.augmentations import SimCLRAugmentation  # SimCLR augmentations
from utils.losses import InfoNCELoss  # InfoNCE loss
from pretrain.trainer import Pretrainer  # Pretrainer class
from config import PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, PROGRESSIVE_RESOLUTIONS, NUM_CLASSES

def evaluate_linear_probe(model, train_loader, val_loader, device, num_classes=NUM_CLASSES, epochs=10):
    """Train a linear classifier on top of frozen features and evaluate on validation set."""
    model.eval()  # Freeze the pretrained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add a linear layer for classification
    linear_probe = nn.Linear(model.embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train the linear probe
    for epoch in range(epochs):
        linear_probe.train()
        total_loss = 0.0
        for rgb, labels in train_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            with torch.no_grad():
                features = model(rgb, pretrain=False)  # Get features from frozen model
                features = features.mean(dim=1)  # Global average pooling
            logits = linear_probe(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Linear Probe Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate on validation set
    linear_probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for rgb, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            features = model(rgb, pretrain=False).mean(dim=1)
            logits = linear_probe(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logger.info(f"Linear Probe Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    # Enable TF32 for faster matrix operations on H100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Training will run on CPU, which will be slower.")
    
    img_size = PROGRESSIVE_RESOLUTIONS[1]  # Use 256x256 to reduce memory usage
    logger.info(f"Image size: {img_size}")
    
    # Load dataset (train and val splits)
    dataset_root = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    logger.info(f"Loading dataset from: {dataset_root}")
    train_dataset = SARCLD2024Dataset(dataset_root, img_size, split="train", train_split=0.8, normalize=False)
    val_dataset = SARCLD2024Dataset(dataset_root, img_size, split="val", train_split=0.8, normalize=False)
    logger.info(f"Train dataset loaded. Number of samples: {len(train_dataset)}")
    logger.info(f"Val dataset loaded. Number of samples: {len(val_dataset)}")
    
    batch_size = PRETRAIN_BATCH_SIZE  # Should be 192 from config
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        prefetch_factor=4
    )
    logger.info(f"Train DataLoader created with batch size: {batch_size}")
    logger.info(f"Val DataLoader created with batch size: {batch_size}")
    
    # Initialize model
    logger.info("Initializing DiseaseAwareHVT model...")
    model = DiseaseAwareHVT(img_size=img_size).to(device)
    # Ensure model parameters require gradients
    for param in model.parameters():
        if not param.requires_grad:
            logger.warning("Some model parameters do not require gradients. Enabling gradients for all parameters.")
            param.requires_grad = True
    # Log device of model parameters to confirm GPU usage
    param_device = next(model.parameters()).device
    logger.info(f"Model parameters are on device: {param_device}")
    # Skip torch.compile due to incompatibility with gradient checkpointing
    logger.info("Model initialized and moved to device")
    
    # Initialize augmentations and loss
    augmentations = SimCLRAugmentation(img_size, device=device)  # Pass device
    loss_fn = InfoNCELoss()
    
    # Initialize pretrainer with gradient accumulation
    pretrainer = Pretrainer(model, augmentations, loss_fn, device)
    
    # Start pretraining
    num_epochs = PRETRAIN_EPOCHS
    logger.info(f"Starting pretraining for {num_epochs} epochs...")
    for epoch in pretrainer.pretrain(train_loader, total_epochs=num_epochs):
        # Evaluate representations every 5 epochs
        if epoch % 5 == 0:
            accuracy = evaluate_linear_probe(model, train_loader, val_loader, device)
            # Save checkpoint
            checkpoint_path = f"pretrained_hvt_simclr_epoch_{epoch}.pth"
            pretrainer.save_model(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    pretrainer.save_model("pretrained_hvt_simclr.pth")
    logger.info("Pretrained model saved to pretrained_hvt_simclr.pth")

if __name__ == "__main__":
    main()