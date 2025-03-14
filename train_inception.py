import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from torchvision.models import Inception_V3_Weights
import numpy as np

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

checkpoint = torch.load('dataset_splits.pth')
combined_train_dataset = checkpoint['train_dataset']
val_dataset = checkpoint['val_dataset']
test_dataset = checkpoint['test_dataset']

# Calculate class weights for imbalance
def calculate_class_weights(dataset):
    class_counts = np.zeros(7)  # 7 classes
    for _, label in dataset.data:  # Access the label from each (image, label) tuple
        class_counts[label] += 1
    total_samples = np.sum(class_counts)
    # Avoid division by zero for empty classes (though unlikely here)
    class_counts[class_counts == 0] = 1  # Prevent division by zero
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights).to(device)

train_class_weights = calculate_class_weights(combined_train_dataset)
print("Class weights:", train_class_weights)

# Create per-sample weights for WeightedRandomSampler
sample_weights = [train_class_weights[label].item() for _, label in combined_train_dataset.data]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(combined_train_dataset), replacement=True)

train_loader = DataLoader(combined_train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training set size: {len(combined_train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Step 2: Load and Modify Inception V3 with Dropout
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
num_classes = 7
model.dropout = nn.Dropout(p=0.5)  # Add dropout to reduce overfitting
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
print("Inception V3 model loaded and modified for 7 classes with dropout.")

# Step 3: Define Loss Function, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss(weight=train_class_weights)  # Weighted loss for imbalance
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Lower lr and weight decay for regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)

# Step 4: Implement the Training Loop with Early Stopping
num_epochs = 30
best_val_loss = float('inf')
best_model_path = 'best_inception_v3.pth'
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
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

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

# Step 5: Evaluate on Test Set
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Step 6: Per-Class Performance Analysis
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage',
               'Leaf Hopper Jassids', 'Leaf Redding', 'Leaf Variegation']
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))