# train_inception.py
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from torchvision.models import Inception_V3_Weights
from torch.cuda.amp import GradScaler, autocast
import logging
from dataset_utils import CottonLeafDataset  # Import the custom dataset class

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Load Preprocessed Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

output_dir = 'phase2_inception_v3_results'
os.makedirs(output_dir, exist_ok=True)

try:
    checkpoint = torch.load(os.path.join('./phase1_checkpoints', 'phase1_preprocessed_data.pth'))
except FileNotFoundError as e:
    logger.error(f"Preprocessed data not found: {e}")
    raise FileNotFoundError("Run Phase 1 first to generate preprocessed data.")

combined_train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']
class_names = checkpoint['class_names']
has_multimodal = checkpoint['has_multimodal']

# Adjust class names to 7 classes (excluding unnamed if present)
if len(class_names) > 7:
    class_names = class_names[:-1]  # Remove unnamed class
num_classes = len(class_names)

# Recreate DataLoaders
train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Training set size: {len(combined_train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")
logger.info(f"Test set size: {len(test_dataset)}")

# Step 2: Load and Modify Inception V3
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
for param in list(model.parameters())[:-20]:  # Freeze early layers
    param.requires_grad = False

# Adjust final classifier for 7 classes with class weights
in_features = model.fc.in_features
class_counts = [checkpoint['original_stats']['class_distribution'][cls][0] for cls in class_names]
# Fix: Compute class weights correctly
class_counts_tensor = torch.tensor(class_counts, dtype=torch.float)
class_weights = (1.0 / class_counts_tensor) / torch.sum(1.0 / class_counts_tensor)
model.fc = nn.Sequential(
    nn.BatchNorm1d(in_features),
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)
logger.info(f"Inception V3 model loaded and modified for {num_classes} classes.")

# Step 3: Define Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
scaler = GradScaler() if torch.cuda.is_available() else None

# Step 4: Training Loop with Early Stopping and Mixed Precision
num_epochs = 30
best_val_acc = 0.0
best_model_path = os.path.join(output_dir, 'best_inception_v3.pth')
patience = 7
trigger_times = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, spectral, labels in train_loader if has_multimodal else zip(train_loader, [None] * len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast() if scaler else torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle auxiliary outputs
                outputs = outputs[0]
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_accuracy)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, spectral, labels in val_loader if has_multimodal else zip(val_loader, [None] * len(val_loader)):
            images, labels = images.to(device), labels.to(device)
            with autocast() if scaler else torch.no_grad():
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_accuracy)

    scheduler.step()
    logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }, best_model_path)
        logger.info(f"Best model saved at epoch {epoch+1} with val_acc: {val_accuracy:.2f}%")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

training_time = time.time() - start_time
logger.info(f"Training completed in {training_time/60:.2f} minutes")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

# Step 5: Evaluate on Test Set
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation accuracy: {checkpoint['val_accuracy']:.2f}%")

model.eval()
test_correct, test_total, test_loss = 0, 0, 0.0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, spectral, labels in test_loader if has_multimodal else zip(test_loader, [None] * len(test_loader)):
        images, labels = images.to(device), labels.to(device)
        with autocast() if scaler else torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100 * test_correct / test_total
logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Detailed Evaluation
logger.info("\nPer-class accuracy:")
class_correct = [0] * num_classes
class_total = [0] * num_classes
for pred, label in zip(all_preds, all_labels):
    class_total[label] += 1
    if pred == label:
        class_correct[label] += 1
for i in range(num_classes):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        logger.info(f"{class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

logger.info("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Save results
with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"Inception V3 Baseline Results\n\n")
    f.write(f"Training time: {training_time/60:.2f} minutes\n")
    f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Test accuracy: {test_accuracy:.2f}%\n")
    f.write("\nPer-class accuracy:\n")
    for i in range(num_classes):
        if class_total[i] > 0:
            f.write(f"{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})\n")
    f.write("\nClassification Report:\n")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    for cls in class_names:
        f.write(f"{cls}:\n{report[str(class_names.index(cls))]}\n")

torch.save({
    'model_state_dict': model.state_dict(),
    'test_accuracy': test_accuracy,
    'class_report': report,
    'confusion_matrix': cm,
    'training_history': history
}, os.path.join(output_dir, 'inception_v3_baseline.pth'))

logger.info(f"Phase 2 complete! Inception V3 baseline model trained and evaluated with test accuracy: {test_accuracy:.2f}%")
logger.info(f"Results saved to {output_dir}/")