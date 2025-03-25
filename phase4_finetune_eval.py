# phase4_finetune_eval.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import numpy as np
import logging
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from data_utils import CottonLeafDataset, get_transforms
from hvt_model import HierarchicalVisionTransformer
from phase3_ssl_hvt import SSLHierarchicalVisionTransformer
import torchvision
from ptflops import get_model_complexity_info

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class FinetuneCottonLeafDataset(Dataset):
    def __init__(self, samples, transform=None, rare_transform=None, rare_classes=None, has_multimodal=False, is_test=False, resolution=288, label_to_stage=None):
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes or []
        self.has_multimodal = has_multimodal
        self.is_test = is_test
        self.resolution = resolution
        self.label_to_stage = label_to_stage  # Mapping from original labels to stages

        # Validate and clean samples
        self.samples = []
        invalid_indices = []
        reported_length = len(samples) if isinstance(samples, (list, tuple)) else 0
        logger.info(f"Reported length of input samples: {reported_length}")

        for idx in range(reported_length):
            try:
                sample = samples[idx]
                logger.debug(f"Sample {idx} type: {type(sample)}, value: {sample}")

                # Handle different sample formats
                if isinstance(sample, dict):
                    img = sample.get('img')
                    spectral = sample.get('spectral')
                    label = sample.get('label')
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    img, spectral, label = sample[:3]
                else:
                    logger.warning(f"Sample {idx} has unexpected format: {sample}")
                    invalid_indices.append(idx)
                    continue

                # Validate img and spectral
                if img is None:
                    logger.warning(f"Sample {idx} has None img")
                    invalid_indices.append(idx)
                    continue

                if self.has_multimodal and spectral is None:
                    logger.debug(f"Sample {idx} has None spectral, but has_multimodal is True. Proceeding with None spectral.")
                    spectral = None

                # Convert img and spectral to tensors if they aren't already
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)

                if img.dim() < 2:
                    logger.warning(f"Sample {idx} img has {img.dim()} dimensions, reshaping to 2D")
                    img = img.view(-1, 1)

                if self.has_multimodal and spectral is not None:
                    if not isinstance(spectral, torch.Tensor):
                        spectral = transforms.ToTensor()(spectral)
                    if spectral.dim() < 2:
                        logger.warning(f"Sample {idx} spectral has {spectral.dim()} dimensions, reshaping to 2D")
                        spectral = spectral.view(-1, 1)

                self.samples.append((img, spectral, label))
            except Exception as e:
                logger.warning(f"Error accessing or processing sample at index {idx}: {e}")
                invalid_indices.append(idx)
                continue

        if invalid_indices:
            logger.warning(f"Found {len(invalid_indices)} invalid samples at indices: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}")
        logger.info(f"Dataset initialized with {len(self.samples)} valid samples")

        if len(self.samples) == 0:
            raise ValueError("No valid samples found in the dataset. Cannot proceed with training.")

        self.progression_stages = {
            'early': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.1), 1.1),
            'mid': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.3), 1.3),
            'advanced': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.5), 1.5)
        }

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomResizedCrop(size=(self.resolution, self.resolution), scale=(0.7, 1.0)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            logger.error(f"Index {idx} is out of range for dataset with {len(self.samples)} samples")
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self.samples)} samples")
        try:
            img, spectral, label = self.samples[idx]

            # Map the original label to a stage
            if self.label_to_stage is not None:
                stage = self.label_to_stage.get(label)
                if stage is None:
                    logger.warning(f"Label {label} at index {idx} does not map to a valid stage. Using 'mid' as default.")
                    stage = 'mid'
            else:
                stage = 'mid'  # Fallback if no mapping is provided

            # Ensure img is a tensor
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            
            if img.dim() == 2:
                img = img.unsqueeze(0)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            if img.shape[0] != 3 or img.shape[1] != self.resolution or img.shape[2] != self.resolution:
                img = transforms.Resize((self.resolution, self.resolution))(img)
                if img.shape[0] != 3:
                    img = img[:3, :, :]

            if not self.is_test:
                img = self.augmentation(img)

            if self.transform:
                for t in self.transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        img = t(img)

            if not self.is_test:
                img = self.progression_stages[stage](img)

            if not self.is_test and label in self.rare_classes and self.rare_transform:
                for t in self.rare_transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        img = t(img)

            if self.has_multimodal and spectral is not None:
                if not isinstance(spectral, torch.Tensor):
                    spectral = transforms.ToTensor()(spectral)
                if spectral.dim() == 2:
                    spectral = spectral.unsqueeze(0)
                if spectral.shape[0] != 1 or spectral.shape[1] != self.resolution or spectral.shape[2] != self.resolution:
                    spectral = transforms.Resize((self.resolution, self.resolution))(spectral)
                    spectral = spectral[:1, :, :]

            return img, spectral, stage

        except Exception as e:
            logger.warning(f"Error loading sample at index {idx}: {e}. Returning placeholder.")
            return torch.zeros((3, self.resolution, self.resolution)), torch.zeros((1, self.resolution, self.resolution)) if self.has_multimodal else None, 'mid'

class CrossModalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb_logits, spectral_logits):
        rgb_probs = F.softmax(rgb_logits, dim=1)
        spectral_probs = F.softmax(spectral_logits, dim=1)
        kl_loss = F.kl_div(rgb_probs.log(), spectral_probs, reduction='batchmean')
        return kl_loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_finetune_model(model, train_loader, val_loader, criterion, consistency_criterion, optimizer, scheduler, num_epochs, device, train_mean, train_std):
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_model_path = "./phase4_checkpoints/finetuned_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    patience = 5
    epochs_no_improve = 0
    early_stop = False

    logger.info(f"Training DataLoader has {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Validation DataLoader has {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_consistency_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, spectral, stage) in enumerate(train_loader):
            images = images.to(device)
            spectral = spectral.to(device) if spectral is not None else None
            stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

            if images.min() >= 0 and images.max() <= 1:
                images = (images - train_mean) / train_std

            optimizer.zero_grad()
            
            with autocast():
                images, stage_a, stage_b, lam = mixup_data(images, stage, alpha=1.0, device=device)
                _, rgb_logits = model(images, spectral)
                _, spectral_logits = model(images, None) if spectral is not None else rgb_logits
                class_loss = lam * criterion(rgb_logits, stage_a) + (1 - lam) * criterion(rgb_logits, stage_b)
                consistency_loss = consistency_criterion(rgb_logits, spectral_logits) if spectral is not None else 0.0
                total_loss = class_loss + 1.0 * consistency_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += class_loss.item() * images.size(0)
            running_consistency_loss += consistency_loss.item() * images.size(0) if spectral is not None else 0.0
            _, predicted = torch.max(rgb_logits, 1)
            total += stage.size(0)
            correct += (predicted == stage_a).sum().item()

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Class Loss: {class_loss.item():.4f} | Consistency Loss: {consistency_loss.item():.4f}")
                logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB | Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_consistency_loss = running_consistency_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training Class Loss: {epoch_loss:.4f}, Consistency Loss: {epoch_consistency_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, spectral, stage in val_loader:
                images = images.to(device)
                spectral = spectral.to(device) if spectral is not None else None
                stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

                if images.min() >= 0 and images.max() <= 1:
                    images = (images - train_mean) / train_std

                _, logits = model(images, spectral)
                loss = criterion(logits, stage)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += stage.size(0)
                val_correct += (predicted == stage).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True
                break

        scheduler.step(val_loss)
        logger.info(f"Learning rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if early_stop:
            break

    return best_model_path

def mc_dropout_inference(model, images, spectral, num_samples=10):
    model.eval()
    model.apply(lambda m: setattr(m, 'training', True))
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            _, logits = model(images, spectral)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
    predictions = torch.stack(predictions)
    mean_probs = predictions.mean(dim=0)
    uncertainty = predictions.var(dim=0).mean(dim=1)
    return mean_probs, uncertainty

def grad_cam(model, images, target_class, device):
    model.eval()
    images = images.requires_grad_(True)
    _, logits = model(images, None)
    model.zero_grad()
    logits[:, target_class].sum().backward()
    gradients = images.grad
    activations = model.base_model.features(images, None)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)
    grad_cam_map = F.relu(grad_cam_map)
    grad_cam_map = F.interpolate(grad_cam_map, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
    grad_cam_map = grad_cam_map / (grad_cam_map.max() + 1e-8)
    return grad_cam_map

def evaluate_model(model, test_loader, criterion, device, train_mean, train_std, class_names, robustness_tests=False):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    if robustness_tests:
        degradation_transforms = [
            ("Blur", transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))),
            ("Noise", lambda x: x + torch.randn_like(x) * 0.1),
            ("Occlusion", lambda x: x * (torch.rand_like(x) > 0.2).float()),
            ("Lighting", transforms.ColorJitter(brightness=0.5))
        ]
    else:
        degradation_transforms = [("Normal", lambda x: x)]

    for test_name, degradation in degradation_transforms:
        logger.info(f"Evaluating with {test_name} degradation...")
        test_loss = 0.0
        all_preds = []
        all_labels = []
        all_uncertainties = []

        with torch.no_grad():
            for images, spectral, stage in test_loader:
                images = images.to(device)
                spectral = spectral.to(device) if spectral is not None else None
                stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

                if images.min() >= 0 and images.max() <= 1:
                    images = (images - train_mean) / train_std

                images = degradation(images)

                probs, uncertainty = mc_dropout_inference(model, images, spectral)
                loss = criterion(probs.log(), stage)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(stage.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        n = len(all_labels)
        acc_se = np.sqrt((accuracy * (1 - accuracy)) / n)
        ci_lower, ci_upper = stats.norm.interval(0.95, loc=accuracy, scale=acc_se)

        logger.info(f"{test_name} Test Loss: {test_loss:.4f}")
        logger.info(f"{test_name} Test Accuracy: {accuracy:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        logger.info(f"{test_name} Test F1 Score: {f1:.4f}")
        logger.info(f"{test_name} Average Uncertainty: {np.mean(all_uncertainties):.4f}")
        logger.info(f"{test_name} Confusion Matrix:")
        logger.info(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix ({test_name})')
        plt.savefig(f'./phase4_checkpoints/confusion_matrix_{test_name.lower()}.png')
        plt.close()

    return test_loss, accuracy, f1, cm, all_uncertainties

def evaluate_baselines(test_loader, device, train_mean, train_std):
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 3)
    resnet = resnet.to(device)
    resnet.eval()

    vit = models.vit_b_16(pretrained=True)
    vit.heads = nn.Linear(vit.heads.head.in_features, 3)
    vit = vit.to(device)
    vit.eval()

    criterion = nn.CrossEntropyLoss()
    baselines = {"ResNet-50": resnet, "ViT": vit}
    results = {}

    for name, model in baselines.items():
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, spectral, stage in test_loader:
                images = images.to(device)
                stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

                if images.min() >= 0 and images.max() <= 1:
                    images = (images - train_mean) / train_std

                logits = model(images)
                loss = criterion(logits, stage)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(stage.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        results[name] = {"loss": test_loss, "accuracy": accuracy, "f1": f1}
        logger.info(f"Baseline {name} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    return results

def ablation_study(model, test_loader, criterion, device, train_mean, train_std, class_names):
    model.eval()
    test_loss_rgb = 0.0
    all_preds_rgb = []
    all_labels_rgb = []

    with torch.no_grad():
        for images, spectral, stage in test_loader:
            images = images.to(device)
            stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

            if images.min() >= 0 and images.max() <= 1:
                images = (images - train_mean) / train_std

            _, logits = model(images, None)
            loss = criterion(logits, stage)

            test_loss_rgb += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            all_preds_rgb.extend(predicted.cpu().numpy())
            all_labels_rgb.extend(stage.cpu().numpy())

    test_loss_rgb = test_loss_rgb / len(test_loader.dataset)
    accuracy_rgb = accuracy_score(all_labels_rgb, all_preds_rgb)
    f1_rgb = f1_score(all_labels_rgb, all_preds_rgb, average='weighted')

    logger.info(f"Ablation (RGB Only) - Test Loss: {test_loss_rgb:.4f}, Accuracy: {accuracy_rgb:.4f}, F1 Score: {f1_rgb:.4f}")

    return {"RGB_Only": {"loss": test_loss_rgb, "accuracy": accuracy_rgb, "f1": f1_rgb}}

if __name__ == "__main__":
    try:
        PHASE1_CHECKPOINT_PATH = "./phase1_checkpoints/phase1_preprocessed_data.pth"
        HVT_CHECKPOINT_PATH = "./phase2_checkpoints/HVT_best.pth"
        SSL_CHECKPOINT_PATH = "./phase3_checkpoints/ssl_hvt_best.pth"
        PHASE4_SAVE_PATH = "./phase4_checkpoints"
        os.makedirs(PHASE4_SAVE_PATH, exist_ok=True)

        # Step 1: Load and Inspect Phase 1 Data
        checkpoint_data = torch.load(PHASE1_CHECKPOINT_PATH)
        train_dataset = checkpoint_data['train_dataset']
        val_dataset = checkpoint_data['val_dataset']
        test_dataset = checkpoint_data['test_dataset']
        rare_classes = checkpoint_data['rare_classes']
        has_multimodal = checkpoint_data['has_multimodal']
        train_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        train_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        class_names = checkpoint_data['class_names']
        num_classes = len(class_names)
        logger.info("Successfully loaded Phase 1 data")
        logger.info(f"Phase 1 checkpoint contents: {list(checkpoint_data.keys())}")
        logger.info(f"Reported train_dataset length: {len(train_dataset)}")
        logger.info(f"Reported val_dataset length: {len(val_dataset)}")
        logger.info(f"Reported test_dataset length: {len(test_dataset)}")
        logger.info(f"has_multimodal: {has_multimodal}")
        logger.info(f"train_dataset type: {type(train_dataset)}")
        logger.info(f"class_names: {class_names}")

        # Define mapping from original labels to stages
        # Assuming class_names is something like ['healthy', 'early_disease', 'mid_disease', 'advanced_disease']
        # and labels are 3, 4, 5, 6 respectively
        label_to_stage = {
            3: 'healthy',  # Will be filtered out
            4: 'early',
            5: 'mid',
            6: 'advanced'
        }
        logger.info(f"Label to stage mapping: {label_to_stage}")

        # Convert datasets to lists of samples, filtering out 'healthy' samples
        logger.info("Converting train_dataset to list of samples...")
        train_samples = []
        for idx in range(len(train_dataset)):
            try:
                sample = train_dataset[idx]
                label = sample[2]  # Label is the third element in the tuple
                if label_to_stage.get(label) == 'healthy':
                    continue  # Skip 'healthy' samples
                train_samples.append(sample)
            except Exception as e:
                logger.warning(f"Error accessing train_dataset[{idx}]: {e}")
                continue
        logger.info(f"Converted train_dataset to list with {len(train_samples)} samples after filtering")

        logger.info("Converting val_dataset to list of samples...")
        val_samples = []
        for idx in range(len(val_dataset)):
            try:
                sample = val_dataset[idx]
                label = sample[2]
                if label_to_stage.get(label) == 'healthy':
                    continue
                val_samples.append(sample)
            except Exception as e:
                logger.warning(f"Error accessing val_dataset[{idx}]: {e}")
                continue
        logger.info(f"Converted val_dataset to list with {len(val_samples)} samples after filtering")

        logger.info("Converting test_dataset to list of samples...")
        test_samples = []
        for idx in range(len(test_dataset)):
            try:
                sample = test_dataset[idx]
                label = sample[2]
                if label_to_stage.get(label) == 'healthy':
                    continue
                test_samples.append(sample)
            except Exception as e:
                logger.warning(f"Error accessing test_dataset[{idx}]: {e}")
                continue
        logger.info(f"Converted test_dataset to list with {len(test_samples)} samples after filtering")

        # Additional debugging: inspect the first few and last few samples
        logger.info("Inspecting first 5 samples of train_samples:")
        for i in range(min(5, len(train_samples))):
            try:
                sample = train_samples[i]
                logger.info(f"Sample {i} type: {type(sample)}, value: {sample}")
            except Exception as e:
                logger.warning(f"Error accessing train_samples[{i}]: {e}")
        logger.info("Inspecting last 5 samples of train_samples:")
        for i in range(max(0, len(train_samples) - 5), len(train_samples)):
            try:
                sample = train_samples[i]
                logger.info(f"Sample {i} type: {type(sample)}, value: {sample}")
            except Exception as e:
                logger.warning(f"Error accessing train_samples[{i}]: {e}")

        # Step 2: Initialize and Load HVT Model
        hvt_model = HierarchicalVisionTransformer(
            num_classes=num_classes,
            img_size=288,
            patch_sizes=[16, 8, 4],
            embed_dims=[768, 384, 192],
            num_heads=8,
            num_layers=16,
            has_multimodal=has_multimodal,
            dropout=0.5
        ).to(device)
        
        if os.path.exists(HVT_CHECKPOINT_PATH):
            state_dict = torch.load(HVT_CHECKPOINT_PATH, map_location=device)
            hvt_model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded pretrained HVT weights")
        else:
            logger.warning("HVT checkpoint not found, initializing new model")

        # Step 3: Initialize SSL Model and Load Pretrained Weights
        ssl_model = SSLHierarchicalVisionTransformer(hvt_model, num_classes=3).to(device)
        if os.path.exists(SSL_CHECKPOINT_PATH):
            state_dict = torch.load(SSL_CHECKPOINT_PATH, map_location=device)
            ssl_model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded SSL pretrained weights")
        else:
            logger.error("SSL checkpoint not found!")
            raise FileNotFoundError("SSL checkpoint not found!")

        for param in ssl_model.parameters():
            param.requires_grad = True

        # Step 4: Train at Native Resolution (288x288)
        logger.info("Training at resolution 288x288...")
        train_transforms, rare_transforms, test_transforms = get_transforms()
        finetune_train_dataset = FinetuneCottonLeafDataset(
            train_samples,
            transform=train_transforms,
            rare_transform=rare_transforms,
            rare_classes=rare_classes,
            has_multimodal=has_multimodal,
            is_test=False,
            resolution=288,
            label_to_stage=label_to_stage
        )
        finetune_val_dataset = FinetuneCottonLeafDataset(
            val_samples,
            transform=train_transforms,
            rare_transform=rare_transforms,
            rare_classes=rare_classes,
            has_multimodal=has_multimodal,
            is_test=False,
            resolution=288,
            label_to_stage=label_to_stage
        )
        finetune_test_dataset = FinetuneCottonLeafDataset(
            test_samples,
            transform=test_transforms,
            rare_transform=rare_transforms,
            rare_classes=rare_classes,
            has_multimodal=has_multimodal,
            is_test=True,
            resolution=288,
            label_to_stage=label_to_stage
        )

        # Log class distribution using explicit indices
        stages = []
        for idx in range(len(finetune_train_dataset)):
            try:
                _, _, stage = finetune_train_dataset[idx]
                stages.append(stage)
            except Exception as e:
                logger.warning(f"Error accessing finetune_train_dataset[{idx}]: {e}")
                continue
        stage_counts = {s: stages.count(s) for s in ['early', 'mid', 'advanced']}
        logger.info(f"Training set stage distribution: {stage_counts}")

        train_loader = DataLoader(finetune_train_dataset, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(finetune_val_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(finetune_test_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)
        logger.info("DataLoaders created")

        criterion = nn.CrossEntropyLoss()
        consistency_criterion = CrossModalConsistencyLoss()
        optimizer = optim.AdamW(ssl_model.parameters(), lr=1e-4, weight_decay=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        logger.info("Optimizer, loss functions, and scheduler initialized")

        best_model_path = train_finetune_model(
            ssl_model, train_loader, val_loader, 
            criterion, consistency_criterion, optimizer, scheduler, num_epochs=20, device=device,
            train_mean=train_mean, train_std=train_std
        )

        # Step 5: Load Best Model for Final Evaluation
        ssl_model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best fine-tuned model for evaluation")

        # Step 6: Evaluate on Test Set with Robustness Tests
        class_names = ['early', 'mid', 'advanced']
        logger.info("Evaluating on test set with robustness tests...")
        test_loss, test_accuracy, test_f1, cm, uncertainties = evaluate_model(
            ssl_model, test_loader, criterion, device, train_mean, train_std, class_names, robustness_tests=True
        )

        # Step 7: Baseline Comparisons
        logger.info("Evaluating baseline models...")
        baseline_results = evaluate_baselines(test_loader, device, train_mean, train_std)

        # Step 8: Ablation Study
        logger.info("Conducting ablation study...")
        ablation_results = ablation_study(ssl_model, test_loader, criterion, device, train_mean, train_std, class_names)

        # Step 9: Computational Efficiency Analysis
        logger.info("Computing computational efficiency...")
        macs, params = get_model_complexity_info(ssl_model, (3, 288, 288), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info(f"Computational Complexity: {macs} MACs, {params} parameters")

        # Step 10: Grad-CAM Visualization
        logger.info("Generating Grad-CAM visualizations...")
        ssl_model.eval()
        with torch.no_grad():
            for i, (images, spectral, stage) in enumerate(test_loader):
                if i >= 5:
                    break
                images = images.to(device)
                stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)
                grad_cam_map = grad_cam(ssl_model, images, target_class=stage[0].item(), device=device)
                img = images[0].cpu().permute(1, 2, 0).numpy()
                heatmap = grad_cam_map[0].cpu().squeeze().numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.title(f"Grad-CAM (Class: {class_names[stage[0].item()]})")
                plt.savefig(f"./phase4_checkpoints/gradcam_sample_{i}.png")
                plt.close()

        # Step 11: Save Final Model and Results
        final_model_path = os.path.join(PHASE4_SAVE_PATH, "finetuned_final.pth")
        torch.save({
            'model_state_dict': ssl_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'confusion_matrix': cm,
            'uncertainties': uncertainties,
            'baseline_results': baseline_results,
            'ablation_results': ablation_results,
            'class_names': class_names
        }, final_model_path)
        logger.info(f"Final model and results saved to {final_model_path}")

        logger.info("Phase 4 completed successfully!")

    except Exception as e:
        logger.error(f"Error in Phase 4: {e}")
        raise