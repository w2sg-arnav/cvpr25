import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
logging.info(f"Parent directory added to sys.path: {parent_dir}")
logging.info(f"sys.path: {sys.path}")

# Try to import config, fallback to defaults if it fails
try:
    from config import PROGRESSIVE_RESOLUTIONS, FINETUNE_BATCH_SIZE, PRETRAINED_MODEL_PATH, NUM_CLASSES
except ModuleNotFoundError as e:
    logging.warning(f"Failed to import from config: {e}. Using default values.")
    PROGRESSIVE_RESOLUTIONS = [(128, 128), (256, 256), (384, 384)]
    FINETUNE_BATCH_SIZE = 32
    PRETRAINED_MODEL_PATH = "pretrained_hvt.pth"
    NUM_CLASSES = 7

# Remaining imports
from models.hvt import DiseaseAwareHVT
from models.efficientnet import EfficientNetBaseline
from utils.augmentations import FinetuneAugmentation
from finetune.trainer import Finetuner
from dataset import SARCLD2024Dataset

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logging.info("Enabled CUDA optimizations (cudnn.benchmark)")
    
    # Use the largest resolution for fine-tuning
    img_size = PROGRESSIVE_RESOLUTIONS[-1]  # (384, 384)
    print(f"Image size: {img_size}")
    
    # Load dataset with optimized DataLoader
    dataset_root = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")
    
    train_dataset = SARCLD2024Dataset(dataset_root, img_size, split="train", train_split=0.8, normalize=True)
    val_dataset = SARCLD2024Dataset(dataset_root, img_size, split="val", train_split=0.8, normalize=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
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
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    logging.info(f"Training dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Update NUM_CLASSES based on dataset
    num_classes = len(train_dataset.get_class_names())
    if num_classes != NUM_CLASSES:
        logging.warning(f"NUM_CLASSES in config ({NUM_CLASSES}) does not match dataset classes ({num_classes}). Updating NUM_CLASSES.")
        globals()['NUM_CLASSES'] = num_classes
    
    # Get class weights for weighted loss
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    logging.info(f"Class weights for weighted loss: {class_weights}")
    
    # Initialize models AFTER updating NUM_CLASSES
    hvt_model = DiseaseAwareHVT(img_size=img_size).to(device)
    baseline_model = EfficientNetBaseline().to(device)
    
    # Freeze early layers of EfficientNetBaseline and HVT's EfficientNet
    for name, param in baseline_model.named_parameters():
        if any(f"features.{i}" in name for i in range(5)):
            param.requires_grad = False
    logging.info("Froze early layers (features.0 to features.4) of EfficientNetBaseline")
    for name, param in hvt_model.named_parameters():
        if "efficientnet" in name and any(f"features.{i}" in name for i in range(5)):
            param.requires_grad = False
    logging.info("Froze early layers (features.0 to features.4) of EfficientNet in HVT")
    
    # Load pretrained weights for HVT
    pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    model_dict = hvt_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    hvt_model.load_state_dict(model_dict)
    logging.info(f"Loaded pretrained weights for DiseaseAwareHVT from {PRETRAINED_MODEL_PATH} (classifier head excluded due to class mismatch)")
    
    # Initialize augmentations
    augmentations = FinetuneAugmentation(img_size)
    
    # Initialize finetuners with class weights and label smoothing
    hvt_finetuner = Finetuner(hvt_model, augmentations, device, class_weights=class_weights, label_smoothing=0.1)
    baseline_finetuner = Finetuner(baseline_model, augmentations, device, class_weights=class_weights, label_smoothing=0.1)
    # Adjust learning rate for EfficientNetBaseline
    for param_group in baseline_finetuner.optimizer.param_groups:
        param_group['lr'] = 5e-5
    logging.info("Set learning rate for EfficientNetBaseline to 5e-5")
    
    # Initialize GradScaler and schedulers
    scaler = GradScaler()
    hvt_scheduler = CosineAnnealingLR(hvt_finetuner.optimizer, T_max=50, eta_min=1e-6)
    baseline_scheduler = CosineAnnealingLR(baseline_finetuner.optimizer, T_max=50, eta_min=1e-6)
    
    # Fine-tune HVT with mixed precision
    print("Fine-tuning DiseaseAwareHVT with mixed precision...")
    logging.info("Fine-tuning DiseaseAwareHVT with mixed precision...")
    best_val_acc_hvt = 0.0
    best_model_path_hvt = "best_finetuned_hvt.pth"
    patience = 10
    patience_counter_hvt = 0
    
    for epoch in range(50):
        # Progressive unfreezing for HVT
        if epoch == 10:
            for name, param in hvt_model.named_parameters():
                if "efficientnet.features.5" in name:
                    param.requires_grad = True
            logging.info("Unfroze EfficientNet features.5 in HVT")
        elif epoch == 20:
            for name, param in hvt_model.named_parameters():
                if "efficientnet.features.6" in name:
                    param.requires_grad = True
            logging.info("Unfroze EfficientNet features.6 in HVT")
        elif epoch == 30:
            for name, param in hvt_model.named_parameters():
                if "efficientnet.features.7" in name:
                    param.requires_grad = True
            logging.info("Unfroze EfficientNet features.7 in HVT")
        
        # HVT Training loop with tqdm
        hvt_finetuner.model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"HVT Epoch {epoch+1}/50", file=sys.stdout) as pbar:
            for rgb, labels in train_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                with autocast():
                    loss = hvt_finetuner.train_step(rgb, labels)
                scaler.scale(loss).backward()
                scaler.step(hvt_finetuner.optimizer)
                scaler.update()
                hvt_finetuner.optimizer.zero_grad()
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        train_loss /= len(train_loader)
        
        # HVT Validation
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
        
        hvt_scheduler.step()
        
        print(f"HVT Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        logging.info(f"Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics["accuracy"] > best_val_acc_hvt:
            best_val_acc_hvt = val_metrics["accuracy"]
            hvt_finetuner.save_model(best_model_path_hvt)
            print(f"New best HVT model saved with Val Accuracy: {best_val_acc_hvt:.4f}")
            logging.info(f"New best model saved with Val Accuracy: {best_val_acc_hvt:.4f}")
            patience_counter_hvt = 0
        else:
            patience_counter_hvt += 1
            if patience_counter_hvt >= patience:
                print(f"HVT early stopping triggered after {patience} epochs without improvement")
                break
    
    hvt_finetuner.save_model("finetuned_hvt.pth")
    print("HVT final model saved to finetuned_hvt.pth")
    
    # Fine-tune baseline with mixed precision
    print("Fine-tuning EfficientNetBaseline with mixed precision...")
    logging.info("Fine-tuning EfficientNetBaseline with mixed precision...")
    best_val_acc_baseline = 0.0
    best_model_path_baseline = "best_finetuned_baseline.pth"
    patience_counter_baseline = 0
    
    for epoch in range(50):
        # Baseline Training loop with tqdm
        baseline_finetuner.model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Baseline Epoch {epoch+1}/50", file=sys.stdout) as pbar:
            for rgb, labels in train_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                with autocast():
                    loss = baseline_finetuner.train_step(rgb, labels)
                scaler.scale(loss).backward()
                scaler.step(baseline_finetuner.optimizer)
                scaler.update()
                baseline_finetuner.optimizer.zero_grad()
                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        train_loss /= len(train_loader)
        
        # Baseline Validation
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
        
        baseline_scheduler.step()
        
        print(f"Baseline Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        logging.info(f"Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics["accuracy"] > best_val_acc_baseline:
            best_val_acc_baseline = val_metrics["accuracy"]
            baseline_finetuner.save_model(best_model_path_baseline)
            print(f"New best Baseline model saved with Val Accuracy: {best_val_acc_baseline:.4f}")
            logging.info(f"New best model saved with Val Accuracy: {best_val_acc_baseline:.4f}")
            patience_counter_baseline = 0
        else:
            patience_counter_baseline += 1
            if patience_counter_baseline >= patience:
                print(f"Baseline early stopping triggered after {patience} epochs without improvement")
                break
    
    baseline_finetuner.save_model("finetuned_baseline.pth")
    print("Baseline final model saved to finetuned_baseline.pth")

if __name__ == "__main__":
    main()