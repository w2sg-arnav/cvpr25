from hvt_model import HierarchicalVisionTransformer, SSLHierarchicalVisionTransformer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import numpy as np
import logging
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from copy import deepcopy
import cv2
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

class EnhancedFinetuneDataset(Dataset):
    def __init__(self, samples, transform=None, rare_transform=None, rare_classes=None, 
                 has_multimodal=False, is_test=False, resolution=288, label_to_stage=None):
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes or []
        self.has_multimodal = has_multimodal
        self.is_test = is_test
        self.resolution = resolution
        self.label_to_stage = label_to_stage

        # Validate and clean samples
        self.samples = []
        for idx in range(len(samples)):
            try:
                sample = samples[idx]
                if isinstance(sample, dict):
                    img = sample.get('img')
                    spectral = sample.get('spectral')
                    label = sample.get('label')
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    img, spectral, label = sample[:3]
                else:
                    continue

                if img is None:
                    continue

                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)

                if img.dim() < 2:
                    img = img.view(-1, 1)

                if self.has_multimodal and spectral is not None:
                    if not isinstance(spectral, torch.Tensor):
                        spectral = transforms.ToTensor()(spectral)
                    if spectral.dim() < 2:
                        spectral = spectral.view(-1, 1)

                self.samples.append((img, spectral, label))
            except Exception:
                continue

        logger.info(f"Dataset initialized with {len(self.samples)} valid samples")

        # Enhanced augmentation strategies
        self.progression_stages = {
            'early': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.3), 1.3),
            'mid': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.5), 1.5),
            'advanced': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.8), 1.8)
        }

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomResizedCrop(size=(self.resolution, self.resolution), scale=(0.8, 1.0)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, spectral, label = self.samples[idx]
        stage = self.label_to_stage.get(label, 'mid')

        # Convert to 3-channel if needed
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # Resize and ensure proper dimensions
        img = transforms.Resize((self.resolution, self.resolution))(img)
        img = img[:3, :, :]  # Ensure 3 channels

        if not self.is_test:
            img = self.augmentation(img)
            img = self.progression_stages[stage](img)

        if self.transform:
            for t in self.transform.transforms:
                if not isinstance(t, transforms.ToTensor):
                    img = t(img)

        if not self.is_test and label in self.rare_classes and self.rare_transform:
            for t in self.rare_transform.transforms:
                if not isinstance(t, transforms.ToTensor):
                    img = t(img)

        if self.has_multimodal and spectral is not None:
            if not isinstance(spectral, torch.Tensor):
                spectral = transforms.ToTensor()(spectral)
            if spectral.dim() == 2:
                spectral = spectral.unsqueeze(0)
            spectral = transforms.Resize((self.resolution, self.resolution))(spectral)
            spectral = spectral[:1, :, :]  # Ensure 1 channel

        return img, spectral, stage

class EnhancedFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None, label_smoothing=0.15, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        if self.alpha is not None:
            weights = self.alpha[targets].unsqueeze(1)
            log_probs = log_probs * weights
        
        pt = torch.exp(log_probs) * targets_smooth
        pt = pt.sum(dim=1)
        focal_loss = -((1 - pt) ** self.gamma) * log_probs * targets_smooth
        
        if self.reduction == 'mean':
            return focal_loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CrossModalConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, rgb_logits, spectral_logits):
        rgb_probs = F.softmax(rgb_logits / self.temperature, dim=1)
        spectral_probs = F.softmax(spectral_logits / self.temperature, dim=1)
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

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[rand_index], lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def train_epoch(model, train_loader, criterion, consistency_criterion, optimizer, 
               scheduler, scaler, device, train_mean, train_std, ema_model=None):
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
            # Apply MixUp or CutMix
            if np.random.rand() < 0.5:
                images, stage_a, stage_b, lam = mixup_data(images, stage, alpha=1.0, device=device)
            else:
                images, stage_a, stage_b, lam = cutmix_data(images, stage, alpha=1.0)
            
            _, rgb_logits = model(images, spectral)
            _, spectral_logits = model(images, None) if spectral is not None else (None, rgb_logits)
            
            class_loss = lam * criterion(rgb_logits, stage_a) + (1 - lam) * criterion(rgb_logits, stage_b)
            consistency_loss = consistency_criterion(rgb_logits, spectral_logits) if spectral is not None else 0.0
            total_loss = class_loss + 0.5 * consistency_loss

        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if ema_model is not None:
            model_ema(model, ema_model)
        
        scheduler.step()
        
        running_loss += class_loss.item() * images.size(0)
        running_consistency_loss += consistency_loss.item() * images.size(0) if spectral is not None else 0.0
        _, predicted = torch.max(rgb_logits, 1)
        total += stage.size(0)
        correct += (lam * (predicted == stage_a).sum().item() + 
                   (1 - lam) * (predicted == stage_b).sum().item())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_consistency_loss = running_consistency_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_consistency_loss, epoch_acc

def model_ema(model, ema_model, decay=0.999):
    model_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())
    for name in ema_params:
        ema_params[name].data = decay * ema_params[name].data + (1 - decay) * model_params[name].data

def validate(model, val_loader, criterion, device, train_mean, train_std, class_names):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, spectral, stage in val_loader:
            images = images.to(device)
            spectral = spectral.to(device) if spectral is not None else None
            stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

            if images.min() >= 0 and images.max() <= 1:
                images = (images - train_mean) / train_std

            # Test-time augmentation
            outputs = []
            for _ in range(3):
                aug_images = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
                ])(images)
                
                _, logits = model(aug_images, spectral)
                outputs.append(F.softmax(logits, dim=1))
            
            avg_probs = torch.mean(torch.stack(outputs), dim=0)
            loss = criterion(avg_probs.log(), stage)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(avg_probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(stage.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc_dict = {class_names[i]: acc for i, acc in enumerate(per_class_acc)}
    
    return val_loss, val_acc, f1, per_class_acc_dict, cm

def train_finetune_model(model, train_loader, val_loader, criterion, consistency_criterion, 
                        optimizer, scheduler, num_epochs, device, train_mean, train_std, class_names):
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_path = "./phase4_checkpoints/finetuned_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Create EMA model
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)

    patience = 5
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        train_loss, train_cons_loss, train_acc = train_epoch(
            model, train_loader, criterion, consistency_criterion, 
            optimizer, scheduler, scaler, device, train_mean, train_std, ema_model
        )
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                   f"Cons Loss: {train_cons_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Validate both regular and EMA model
        val_loss, val_acc, val_f1, per_class_acc, cm = validate(
            model, val_loader, criterion, device, train_mean, train_std, class_names
        )
        ema_val_loss, ema_val_acc, ema_val_f1, _, _ = validate(
            ema_model, val_loader, criterion, device, train_mean, train_std, class_names
        )
        
        logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"EMA Validation - Loss: {ema_val_loss:.4f}, Acc: {ema_val_acc:.4f}, F1: {ema_val_f1:.4f}")
        logger.info(f"Per-class accuracy: {per_class_acc}")
        
        # Save best model (regular or EMA)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model with Val Acc: {best_val_acc:.4f}")
            epochs_no_improve = 0
        elif ema_val_acc > best_val_acc:
            best_val_acc = ema_val_acc
            torch.save(ema_model.state_dict(), best_model_path)
            logger.info(f"Saved new best EMA model with Val Acc: {best_val_acc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True
                break

        if early_stop:
            break

    return best_model_path

def main():
    try:
        # Configuration
        PHASE1_CHECKPOINT_PATH = "./phase1_checkpoints/phase1_preprocessed_data.pth"
        HVT_CHECKPOINT_PATH = "./phase2_checkpoints/HVT_best.pth"
        SSL_CHECKPOINT_PATH = "./phase3_checkpoints/ssl_hvt_best.pth"
        PHASE4_SAVE_PATH = "./phase4_checkpoints"
        os.makedirs(PHASE4_SAVE_PATH, exist_ok=True)

        # Load data
        checkpoint_data = torch.load(PHASE1_CHECKPOINT_PATH)
        train_dataset = checkpoint_data['train_dataset']
        val_dataset = checkpoint_data['val_dataset']
        test_dataset = checkpoint_data['test_dataset']
        rare_classes = checkpoint_data['rare_classes']
        has_multimodal = checkpoint_data['has_multimodal']
        train_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        train_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        class_names = ['early', 'mid', 'advanced']
        num_classes = len(class_names)

        # Label to stage mapping
        label_to_stage = {
            0: 'early', 1: 'advanced', 2: 'healthy', 
            3: 'damage', 4: 'early', 5: 'mid', 6: 'advanced'
        }

        # Combine and split data
        all_samples = []
        all_stages = []
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    label = sample[2] if isinstance(sample, (list, tuple)) else sample['label']
                    stage = label_to_stage.get(label)
                    if stage in ['early', 'mid', 'advanced']:
                        all_samples.append(sample)
                        all_stages.append(stage)
                except Exception:
                    continue

        # Stratified split
        train_val_samples, test_samples, train_val_stages, test_stages = train_test_split(
            all_samples, all_stages, test_size=0.1, stratify=all_stages, random_state=42
        )
        train_samples, val_samples, train_stages, val_stages = train_test_split(
            train_val_samples, train_val_stages, test_size=0.1111, stratify=train_val_stages, random_state=42
        )

        # Create datasets
        train_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = EnhancedFinetuneDataset(
            train_samples, transform=train_transforms, rare_classes=rare_classes,
            has_multimodal=has_multimodal, resolution=288, label_to_stage=label_to_stage
        )
        val_dataset = EnhancedFinetuneDataset(
            val_samples, transform=train_transforms, rare_classes=rare_classes,
            has_multimodal=has_multimodal, is_test=True, resolution=288, label_to_stage=label_to_stage
        )
        test_dataset = EnhancedFinetuneDataset(
            test_samples, transform=test_transforms, rare_classes=rare_classes,
            has_multimodal=has_multimodal, is_test=True, resolution=288, label_to_stage=label_to_stage
        )

        # Handle class imbalance
        stage_counts = {'early': 0, 'mid': 0, 'advanced': 0}
        for _, _, stage in train_dataset:
            stage_counts[stage] += 1

        class_weights = torch.tensor([
            1.0 / stage_counts['early'],
            1.0 / stage_counts['mid'],
            1.0 / stage_counts['advanced']
        ]).to(device)
        class_weights = class_weights / class_weights.sum()

        # Create weighted sampler
        sample_weights = torch.zeros(len(train_dataset))
        for idx, (_, _, stage) in enumerate(train_dataset):
            sample_weights[idx] = class_weights[{'early': 0, 'mid': 1, 'advanced': 2}[stage]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, 
                                num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                               num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                                num_workers=8, pin_memory=True)

        # Initialize model
        hvt_model = HierarchicalVisionTransformer(
            num_classes=num_classes,
            img_size=288,
            patch_sizes=[16, 8, 4],
            embed_dims=[768, 384, 192],
            num_heads=12,
            num_layers=12,
            mlp_ratio=4.0,
            has_multimodal=has_multimodal,
            dropout=0.1
        ).to(device)

        if os.path.exists(HVT_CHECKPOINT_PATH):
            hvt_model.load_state_dict(torch.load(HVT_CHECKPOINT_PATH, map_location=device))
            logger.info("Loaded pretrained HVT weights")

        ssl_model = SSLHierarchicalVisionTransformer(hvt_model, num_classes=3).to(device)
        if os.path.exists(SSL_CHECKPOINT_PATH):
            ssl_model.load_state_dict(torch.load(SSL_CHECKPOINT_PATH, map_location=device))
            logger.info("Loaded SSL pretrained weights")

        # Loss functions and optimizer
        criterion = EnhancedFocalLoss(gamma=1.5, alpha=class_weights, label_smoothing=0.15)
        consistency_criterion = CrossModalConsistencyLoss(temperature=0.1)
        
        optimizer = optim.AdamW(
            ssl_model.parameters(),
            lr=3e-5,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            steps_per_epoch=len(train_loader),
            epochs=50,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=10000
        )

        # Training
        best_model_path = train_finetune_model(
            ssl_model, train_loader, val_loader, criterion, consistency_criterion,
            optimizer, scheduler, num_epochs=50, device=device,
            train_mean=train_mean, train_std=train_std, class_names=class_names
        )

        # Evaluation
        ssl_model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, test_f1, _, test_cm = validate(
            ssl_model, test_loader, criterion, device, train_mean, train_std, class_names
        )
        
        logger.info(f"Final Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Save final model
        final_model_path = os.path.join(PHASE4_SAVE_PATH, "finetuned_final.pth")
        torch.save({
            'model_state_dict': ssl_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc,
                'f1_score': test_f1,
                'confusion_matrix': test_cm
            },
            'class_names': class_names
        }, final_model_path)
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    main()