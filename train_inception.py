import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import Inception_V3_Weights
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Define CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Step 1: Load the Preprocessed Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for results
output_dir = 'inception_v3_results'
os.makedirs(output_dir, exist_ok=True)

checkpoint = torch.load('dataset_splits.pth')
combined_train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']

# Calculate class weights for imbalance
def calculate_class_weights(dataset):
    class_counts = np.zeros(7)  # 7 classes
    for _, label in dataset.data:
        class_counts[label] += 1
    
    # Print class distribution
    class_names = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage',
                   'Leaf Hopper Jassids', 'Leaf Redding', 'Leaf Variegation']
    print("Class distribution in training set:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {name}: {int(count)} samples")
    
    total_samples = np.sum(class_counts)
    class_counts[class_counts == 0] = 1  # Prevent division by zero
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights).to(device)

train_class_weights = calculate_class_weights(combined_train_dataset)
print("Class weights:", train_class_weights)

# Create per-sample weights for WeightedRandomSampler
sample_weights = [train_class_weights[label].item() for _, label in combined_train_dataset.data]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(combined_train_dataset), replacement=True)

# DataLoaders with proper workers for better performance
num_workers = 4 if torch.cuda.is_available() else 0
train_loader = DataLoader(combined_train_dataset, batch_size=32, sampler=sampler, 
                         num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                       num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                        num_workers=num_workers, pin_memory=True)

print(f"Training set size: {len(combined_train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Step 2: Load and Modify Inception V3 with Dropout and BatchNorm
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

# Freeze early layers to prevent overfitting with smaller dataset
for param in list(model.parameters())[:-20]:  # Keep last few layers trainable
    param.requires_grad = False

# Modify final layers for classification
num_classes = 7
# Replace final classifier with a sequence including BatchNorm and Dropout
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.BatchNorm1d(in_features),
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)
print("Inception V3 model loaded and modified for 7 classes with enhanced regularization.")

# Step 3: Define Loss Function, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss(weight=train_class_weights)
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# Using CosineAnnealingLR for better convergence
scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

# Step 4: Implement the Training Loop with Early Stopping and Mixed Precision
num_epochs = 30
best_val_loss = float('inf')
best_val_acc = 0.0
best_model_path = os.path.join(output_dir, 'best_inception_v3.pth')
patience = 7  # Increased patience for better convergence
trigger_times = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# Enable mixed precision for faster training if available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if isinstance(outputs, tuple):  # Handle auxiliary outputs
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle auxiliary outputs
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_accuracy)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_accuracy)

    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save both by loss and accuracy for better model selection
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, best_model_path)
        print(f"Best model saved at epoch {epoch+1} with val_acc: {val_accuracy:.2f}%")
        trigger_times = 0
    elif val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print(f"Training completed in {training_time/60:.2f} minutes")

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
# Load the best model for evaluation
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation accuracy: {checkpoint['val_accuracy']:.2f}%")

model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0
all_preds = []
all_labels = []
class_correct = [0] * 7
class_total = [0] * 7

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1
        
        # Save for classification report
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Per-class accuracy
class_names = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage',
               'Leaf Hopper Jassids', 'Leaf Redding', 'Leaf Variegation']
print("\nPer-class accuracy:")
for i in range(7):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{class_names[i]}: No samples")

# Detailed Classification Report
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
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

# Save all results to a text file
with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"Inception V3 Model for Cotton Disease Classification\n\n")
    f.write(f"Training time: {training_time/60:.2f} minutes\n")
    f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Test accuracy: {test_accuracy:.2f}%\n\n")
    
    f.write("Per-class accuracy:\n")
    for i in range(7):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            f.write(f"{class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})\n")
    
    f.write("\nClassification Report:\n")
    for cls in class_names:
        f.write(f"{cls}:\n")
        f.write(f"  Precision: {report[cls]['precision']:.4f}\n")
        f.write(f"  Recall: {report[cls]['recall']:.4f}\n")
        f.write(f"  F1-score: {report[cls]['f1-score']:.4f}\n")
        f.write(f"  Support: {report[cls]['support']}\n")
    
    f.write(f"\nMacro avg:\n")
    f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
    f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
    f.write(f"  F1-score: {report['macro avg']['f1-score']:.4f}\n")
    
    f.write(f"\nWeighted avg:\n")
    f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
    f.write(f"  Recall: {report['weighted avg']['recall']:.4f}\n")
    f.write(f"  F1-score: {report['weighted avg']['f1-score']:.4f}\n")

# Plot ROC curve for multi-class classification
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

# Get predicted probabilities
model.eval()
y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(7):
    y_test_binary = (np.array(y_true) == i).astype(int)
    fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(np.eye(7)[y_true].ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(7)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(7):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 7

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curves
plt.figure(figsize=(12, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average (AUC = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label=f'macro-average (AUC = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(7), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Save model architecture summary if torchsummary is available
try:
    from torchsummary import summary
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        # Redirect stdout to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        summary(model, (3, 299, 299), device=device)
        sys.stdout = original_stdout
except ImportError:
    print("torchsummary not available, skipping model architecture summary")

# Calculate and save inference time metrics
inference_times = []
model.eval()
with torch.no_grad():
    # Warm-up
    for _ in range(10):
        dummy_input = torch.rand(1, 3, 299, 299).to(device)
        _ = model(dummy_input)
    
    # Measure inference time
    num_runs = 100
    for _ in range(num_runs):
        dummy_input = torch.rand(1, 3, 299, 299).to(device)
        start = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        inference_times.append(end - start)

avg_inference_time = sum(inference_times) / len(inference_times)
with open(os.path.join(output_dir, 'inference_metrics.txt'), 'w') as f:
    f.write(f"Average inference time per image: {avg_inference_time*1000:.2f} ms\n")
    f.write(f"Throughput: {1/avg_inference_time:.2f} images/second\n")

print("\nPhase 2 complete! Inception V3 baseline model trained and evaluated.")
print(f"Test accuracy: {test_accuracy:.2f}%")
print(f"Results saved to {output_dir}/")

# Checkpoint for next phase
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_accuracy,
    'class_report': report,
    'confusion_matrix': cm,
    'training_history': history
}, os.path.join(output_dir, 'inception_v3_final.pth'))