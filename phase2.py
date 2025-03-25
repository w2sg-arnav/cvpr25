import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hvt_model import HierarchicalVisionTransformer  # Import HVT
from torchvision.models import Inception_V3_Weights
from torchvision import models
from typing import Optional, Tuple  # Import Optional and Tuple from typing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# H100 Optimization Configuration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.95)  # Allocate 95% of memory

# Load Phase 1 data
SAVE_PATH = "./phase1_checkpoints"
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'phase1_preprocessed_data.pth')
if not os.path.exists(CHECKPOINT_PATH):
    logger.error(f"Phase 1 checkpoint not found at {CHECKPOINT_PATH}")
    raise FileNotFoundError(f"Phase 1 checkpoint not found at {CHECKPOINT_PATH}")

checkpoint_data = torch.load(CHECKPOINT_PATH)
train_dataset = checkpoint_data['train_dataset']
val_dataset = checkpoint_data['val_dataset']
test_dataset = checkpoint_data['test_dataset']
class_names = checkpoint_data['class_names']
num_classes = len(class_names)
has_multimodal = checkpoint_data['has_multimodal']
class_distribution = checkpoint_data['original_stats']['class_distribution']
SPECTRAL_DIR = checkpoint_data['spectral_dir'] # check if there is any spectral data available or not

# Create DataLoaders
batch_size = 64 # Increased Batch size. You can try increasing it more.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Compute class weights for imbalance
class_counts = np.array([class_distribution[name][0] for name in class_names])
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = torch.FloatTensor(class_weights).to(device)
logger.info(f"Class weights: {class_weights.tolist()}")

def get_inception_v3_model(num_classes, weights=Inception_V3_Weights.IMAGENET1K_V1):
    model = models.inception_v3(weights=weights)
    model.aux_logits = False  # Disable auxiliary logits for simplicity
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Freeze all layers except the last 20
    for name, param in model.named_parameters():
        if "Mixed_7" not in name and "fc" not in name:
            param.requires_grad = False
    
    return model.to(device)

# Training Function with Learning Rate Scheduling
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: torch.device, scheduler: Optional[ReduceLROnPlateau] = None, model_name: str = "HVT") -> Tuple[list, list, list, list, float]:
    """Trains the model and validates periodically.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs.
        device (torch.device): Device to use (CPU or GPU).
        scheduler (ReduceLROnPlateau, optional): Learning rate scheduler. Defaults to None.
        model_name (str, optional): Model name for saving checkpoints. Defaults to "HVT".

    Returns:
        Tuple[list, list, list, list, float]: Train losses, validation losses, train accuracies, validation accuracies, and best validation accuracy.
    """
    scaler = GradScaler() # Define scaler
    best_val_acc = 0.0
    best_model_path = f"./phase2_checkpoints/{model_name}_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    accumulation_steps = 2  # You could further reduce if it runs out of memory

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad() # Reset gradients *before* the inner loop

        for i, (inputs, spectral, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            spectral = spectral.to(device) if has_multimodal and SPECTRAL_DIR else None  # Only move to device if spectral data exists

            with autocast():  # Enable mixed precision
                if isinstance(model, HierarchicalVisionTransformer):
                    outputs = model(inputs, spectral)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss = loss / accumulation_steps # Scale the loss

            scaler.scale(loss).backward() # Scale the gradients

            if (i + 1) % accumulation_steps == 0:  # Accumulate gradients
                scaler.step(optimizer)  # Update weights
                scaler.update()
                optimizer.zero_grad() # Clear the gradients

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(), autocast():  # Enable mixed precision
            for inputs, spectral, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                spectral = spectral.to(device) if has_multimodal and SPECTRAL_DIR else None

                if isinstance(model, HierarchicalVisionTransformer):
                    outputs = model(inputs, spectral)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(f"Epoch {epoch+1}/{num_epochs} - {model_name}")
        logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Acc: {best_val_acc:.2f}%")

        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    return train_losses, val_losses, train_accs, val_accs, best_val_acc

# Evaluation Function
def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device, class_names: list, model_name: str = "HVT") -> Tuple[float, dict, np.ndarray]:
    """Evaluates the model on the test set.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(), autocast():  # Use mixed precision for inference
        for inputs, spectral, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            spectral = spectral.to(device) if has_multimodal and SPECTRAL_DIR else None  # Only move to device if spectral data exists

            if isinstance(model, HierarchicalVisionTransformer):
                outputs = model(inputs, spectral)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100 * correct / total

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    logger.info(f"\n{model_name} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    logger.info("\nClassification Report:")
    for class_name, metrics in report.items():
        if class_name in class_names:
            logger.info(f"{class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./phase2_checkpoints/{model_name}_confusion_matrix.png")
    plt.close()

    return test_acc, report, cm

# Efficiency Analysis with Multiple Batch Sizes
def efficiency_analysis(model: nn.Module, input_size: Tuple[int, int, int] = (3, 299, 299), device: str = 'cuda', model_name: str = "HVT"):
    """Analyzes the efficiency (model size and inference time) of a model with different batch sizes.
    """
    model.eval()
    batch_sizes = [1, 64] # Increased batch size
    efficiency_metrics = {}

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        spectral_input = torch.randn(batch_size, 299, 299).to(device) if has_multimodal and SPECTRAL_DIR else None  # Condition on Spectral Data

        # Model size
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)  # MB

        # Inference time
        num_trials = 100
        start_time = time.time()
        with torch.no_grad(), autocast():  # Enable mixed precision
            for _ in range(num_trials):
                if isinstance(model, HierarchicalVisionTransformer):
                    outputs = model(dummy_input, spectral_input)
                else:
                    outputs = model(dummy_input)
        avg_time = (time.time() - start_time) / num_trials * 1000  # ms

        efficiency_metrics[batch_size] = {'model_size': model_size, 'inference_time': avg_time}
        logger.info(f"\n{model_name} Efficiency Analysis (Batch Size {batch_size}):")
        logger.info(f"Model Size: {model_size:.2f} MB")
        logger.info(f"Average Inference Time ({device}): {avg_time:.2f} ms/sample")

    return efficiency_metrics

# Main Execution
if __name__ == "__main__":
    try:
        # Step 1: Initialize HVT
        hvt_model = HierarchicalVisionTransformer(
            num_classes=num_classes,
            img_size=299,
            patch_sizes=[16, 8, 4],
            embed_dims=[768, 384, 192],
            num_heads=24,
            num_layers=16,
        has_multimodal=has_multimodal,  # set the has_multimodal flag here for if we want spectral analysis
        ).to(device)
        
        # Inception model implementation
        inception_model = get_inception_v3_model(num_classes)

        # Step 2: Define Loss Function, Optimizer, and Scheduler (HVT)
        criterion_hvt = nn.CrossEntropyLoss(weight=class_weights)
        optimizer_hvt = optim.Adam(hvt_model.parameters(), lr=0.0001)
        scheduler_hvt = ReduceLROnPlateau(optimizer_hvt, mode='min', factor=0.1, patience=5)

        logger.info("Training HVT architecture...")
        hvt_train_losses, hvt_val_losses, hvt_train_accs, hvt_val_accs, hvt_best_val_acc = train_model(
            hvt_model, train_loader, val_loader, criterion_hvt, optimizer_hvt, num_epochs=10,
            device=device, scheduler=scheduler_hvt, model_name="HVT"
        )

        # Step 3: Evaluate HVT
        hvt_model.load_state_dict(torch.load("./phase2_checkpoints/HVT_best.pth"))
        hvt_test_acc, hvt_report, hvt_cm = evaluate_model(hvt_model, test_loader, criterion_hvt, device, class_names, model_name="HVT")
        hvt_metrics = efficiency_analysis(hvt_model, device=device, model_name="HVT")
    
        inception_model.load_state_dict(torch.load("./phase2_checkpoints/InceptionV3_best.pth"))
        test_acc, report, cm = evaluate_model(inception_model, test_loader, criterion_hvt, device, class_names, model_name="InceptionV3")
        inception_metrics = efficiency_analysis(inception_model, device=device, model_name="InceptionV3")
    
        # Step 4: Save Results
        phase2_results = {
            'inception': {
                'test_acc': test_acc,
                'report': report,
                'confusion_matrix': cm,
                'efficiency_metrics': inception_metrics
            },
            'hvt': {
                'train_losses': hvt_train_losses,
                'val_losses': hvt_val_losses,
                'train_accs': hvt_train_accs,
                'val_accs': hvt_val_accs,
                'best_val_acc': hvt_best_val_acc,
                'test_acc': hvt_test_acc,
                'report': hvt_report,
                'confusion_matrix': hvt_cm,
                'efficiency_metrics': hvt_metrics
            }
        }
        torch.save(phase2_results, "./phase2_checkpoints/phase2_results.pth")

        logger.info("Phase 2 completed: HVT architecture trained and evaluated.")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise