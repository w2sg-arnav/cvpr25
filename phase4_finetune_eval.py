import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import logging
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision.transforms import autoaugment, RandAugment

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from hvt_model import SwinTransformer, SSLHierarchicalVisionTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class EnhancedFinetuneDataset(Dataset):
    def __init__(self, samples, transform=None, rare_classes=None, has_multimodal=False, 
                 is_test=False, resolution=260, label_to_stage=None):
        self.transform = transform
        self.rare_classes = rare_classes or []
        self.has_multimodal = has_multimodal
        self.is_test = is_test
        self.resolution = resolution
        self.label_to_stage = label_to_stage or {}

        self.samples = []
        for sample in samples:
            try:
                if isinstance(sample, dict):
                    img, spectral, label = sample.get('img'), sample.get('spectral'), sample.get('label')
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    img, spectral, label = sample[:3]
                else:
                    continue

                if img is None or label not in self.label_to_stage:
                    continue

                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)

                if self.has_multimodal and spectral is not None:
                    if not isinstance(spectral, torch.Tensor):
                        spectral = transforms.ToTensor()(spectral)
                    if spectral.dim() == 2:
                        spectral = spectral.unsqueeze(0)

                self.samples.append((img, spectral, label))
            except Exception as e:
                logger.warning(f"Skipping invalid sample: {e}")

        logger.info(f"Dataset initialized with {len(self.samples)} valid samples")

        self.progression_stages = {
            'early': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.3), 1.3),
            'mid': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.5), 1.5),
            'advanced': lambda x: TF.adjust_brightness(TF.adjust_contrast(x, 1.8), 1.8)
        }

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            RandAugment(num_ops=3, magnitude=12),
            autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2),
            transforms.RandomResizedCrop(size=(resolution, resolution), scale=(0.7, 1.0)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
        ])

        self.spectral_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, spectral, label = self.samples[idx]
        stage = self.label_to_stage[label]

        img = transforms.Resize((self.resolution, self.resolution))(img)
        img = img[:3, :, :]

        if not self.is_test:
            img = (img * 255).to(torch.uint8)
            img = self.augmentation(img)
            img = img.to(torch.float32) / 255.0
            img = self.progression_stages[stage](img)

        if self.transform:
            img = self.transform(img)

        if self.has_multimodal and spectral is not None:
            spectral = transforms.Resize((self.resolution, self.resolution))(spectral)
            spectral = spectral[:1, :, :]
            if not self.is_test:
                spectral = self.spectral_augmentation(spectral)
                noise = torch.randn_like(spectral) * 0.05
                spectral = torch.clamp(spectral + noise, 0, 1)

        return img, spectral, stage

class EnhancedFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None, label_smoothing=0.2, reduction='mean'):
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

        pt = (torch.exp(log_probs) * targets_smooth).sum(dim=1)
        pt = pt.unsqueeze(1)

        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -focal_weight * log_probs * targets_smooth

        if self.reduction == 'mean':
            return focal_loss.sum() / inputs.size(0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CrossModalConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, rgb_logits, spectral_logits):
        rgb_probs = F.softmax(rgb_logits / self.temperature, dim=1)
        spectral_probs = F.softmax(spectral_logits / self.temperature, dim=1)
        return F.kl_div(rgb_probs.log(), spectral_probs, reduction='batchmean')

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
                scheduler, scaler, device, train_mean, train_std, ema_model=None, 
                accum_steps=4, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0
    running_cons_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    if epoch is not None and total_epochs is not None:
        criterion.gamma = 1.5 + 1.5 * (epoch / total_epochs)

    for batch_idx, (images, spectral, stage) in enumerate(train_loader):
        images = images.to(device)
        spectral = spectral.to(device) if spectral is not None else None
        stage = torch.tensor([{'early': 0, 'mid': 1, 'advanced': 2}[s] for s in stage]).to(device)

        images = (images - train_mean) / train_std

        with autocast():
            if np.random.rand() < 0.7:
                if np.random.rand() < 0.5:
                    images, stage_a, stage_b, lam = mixup_data(images, stage, alpha=1.2, device=device)
                else:
                    images, stage_a, stage_b, lam = cutmix_data(images, stage, alpha=1.2)
            else:
                stage_a, stage_b, lam = stage, stage, 1.0

            _, rgb_logits = model(images, spectral)
            _, spectral_logits = model(images, None) if spectral is not None else (None, rgb_logits)

            class_loss = lam * criterion(rgb_logits, stage_a) + (1 - lam) * criterion(rgb_logits, stage_b)
            consistency_loss = consistency_criterion(rgb_logits, spectral_logits) if spectral is not None else 0.0
            total_loss = (class_loss + 0.5 * consistency_loss) / accum_steps

        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if ema_model is not None:
            model_ema(model, ema_model)

        running_loss += class_loss.item() * images.size(0)
        running_cons_loss += consistency_loss.item() * images.size(0) if spectral is not None else 0.0
        _, predicted = torch.max(rgb_logits, 1)
        total += stage.size(0)
        correct += (lam * (predicted == stage_a).sum().item() + 
                   (1 - lam) * (predicted == stage_b).sum().item())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_cons_loss = running_cons_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_cons_loss, epoch_acc

def model_ema(model, ema_model, decay=0.995):
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

            images = (images - train_mean) / train_std

            outputs = []
            for _ in range(5):
                aug_images = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=30),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
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
    per_class_acc = {class_names[i]: cm[i, i] / max(cm[i].sum(), 1) for i in range(len(class_names))}
    per_class_f1 = {class_names[i]: f for i, f in enumerate(f1_score(all_labels, all_preds, average=None))}

    return val_loss, val_acc, f1, per_class_acc, per_class_f1, cm

def train_finetune_model(model, train_loader, val_loader, criterion, consistency_criterion, 
                         optimizer, scheduler, num_epochs, device, train_mean, train_std, class_names):
    scaler = GradScaler()
    best_val_acc = 0.0
    best_model_path = "./phase4_checkpoints/finetuned_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)

    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_cons_loss, train_acc = train_epoch(
            model, train_loader, criterion, consistency_criterion, optimizer, scheduler, 
            scaler, device, train_mean, train_std, ema_model, epoch=epoch, total_epochs=num_epochs
        )
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                    f"Cons Loss: {train_cons_loss:.4f}, Acc: {train_acc:.2f}%")

        val_loss, val_acc, val_f1, per_class_acc, per_class_f1, cm = validate(
            model, val_loader, criterion, device, train_mean, train_std, class_names
        )
        ema_val_loss, ema_val_acc, ema_val_f1, ema_per_class_acc, ema_per_class_f1, ema_cm = validate(
            ema_model, val_loader, criterion, device, train_mean, train_std, class_names
        )

        logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"EMA Validation - Loss: {ema_val_loss:.4f}, Acc: {ema_val_acc:.4f}, F1: {ema_val_f1:.4f}")
        logger.info(f"Per-class accuracy: {per_class_acc}")
        logger.info(f"Per-class F1: {per_class_f1}")

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
                break

    return best_model_path

def load_model_weights(model, checkpoint_path, strict=False, ignore_keys=None):
    ignore_keys = ignore_keys or []
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in checkpoint.items():
        if k in ignore_keys or k not in model_dict:
            continue
        if v.shape == model_dict[k].shape:
            pretrained_dict[k] = v
        else:
            logger.warning(f"Shape mismatch for {k}: checkpoint {v.shape}, model {model_dict[k].shape}")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=strict)
    logger.info(f"Loaded weights from {checkpoint_path}")

def main():
    try:
        torch.cuda.empty_cache()

        PHASE1_CHECKPOINT_PATH = "./phase1_checkpoints/phase1_preprocessed_data.pth"
        HVT_CHECKPOINT_PATH = "./phase2_checkpoints/HVT_best.pth"
        SSL_CHECKPOINT_PATH = "./phase3_checkpoints/ssl_hvt_best.pth"
        PHASE4_SAVE_PATH = "./phase4_checkpoints"
        os.makedirs(PHASE4_SAVE_PATH, exist_ok=True)

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

        label_to_stage = {0: 'early', 5: 'mid', 6: 'advanced'}

        all_samples = []
        all_stages = []
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    label = sample[2] if isinstance(sample, (list, tuple)) else sample['label']
                    stage = label_to_stage.get(label)
                    if stage:
                        all_samples.append(sample)
                        all_stages.append(stage)
                except Exception:
                    continue

        train_val_samples, test_samples, train_val_stages, test_stages = train_test_split(
            all_samples, all_stages, test_size=0.1, stratify=all_stages, random_state=42
        )
        train_samples, val_samples, train_stages, val_stages = train_test_split(
            train_val_samples, train_val_stages, test_size=0.1111, stratify=train_val_stages, random_state=42
        )

        logger.info(f"Unique stages: {set(train_stages + val_stages + test_stages)}")

        train_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = EnhancedFinetuneDataset(
            train_samples, transform=train_transforms, rare_classes=rare_classes,
            has_multimodal=has_multimodal, resolution=260, label_to_stage=label_to_stage
        )
        val_dataset = EnhancedFinetuneDataset(
            val_samples, transform=train_transforms, rare_classes=rare_classes,
            has_multimodal=has_multimodal, is_test=True, resolution=260, label_to_stage=label_to_stage
        )
        test_dataset = EnhancedFinetuneDataset(
            test_samples, transform=test_transforms, rare_classes=rare_classes,
            has_multimodal=has_multimodal, is_test=True, resolution=260, label_to_stage=label_to_stage
        )

        stage_counts = {'early': 0, 'mid': 0, 'advanced': 0}
        for _, _, stage in train_dataset:
            stage_counts[stage] += 1
        logger.info(f"Stage counts: {stage_counts}")

        class_weights = torch.tensor([1.0 / max(stage_counts[c], 1) for c in class_names]).to(device)
        class_weights = class_weights * torch.tensor([2.0, 2.0, 1.0]).to(device)
        class_weights = class_weights / class_weights.sum()

        sample_weights = torch.tensor([class_weights[{'early': 0, 'mid': 1, 'advanced': 2}[s]] 
                                     for _, _, s in train_dataset])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        batch_size = 8
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                                 num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=8, pin_memory=True)

        hvt_model = SwinTransformer(
            num_classes=num_classes, img_size=260, patch_size=16, embed_dim=168,
            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=8,  # Changed to 8
            drop_rate=0.2, drop_path_rate=0.3, has_multimodal=has_multimodal
        ).to(device)

        load_model_weights(hvt_model, HVT_CHECKPOINT_PATH, strict=False, 
                          ignore_keys=['head.weight', 'head.bias'])
        hvt_model.head = nn.Linear(hvt_model.num_features, num_classes).to(device)

        ssl_model = SSLHierarchicalVisionTransformer(hvt_model, num_classes=num_classes).to(device)
        load_model_weights(ssl_model, SSL_CHECKPOINT_PATH, strict=False, ignore_keys=[
            'base_model.head.weight', 'base_model.head.bias',
            'classification_head.weight', 'classification_head.bias',
            'projection_head.0.weight', 'projection_head.0.bias',
            'projection_head.2.weight', 'projection_head.2.bias'
        ])

        final_embed_dim = ssl_model.base_model.num_features
        ssl_model.projection_head = nn.Sequential(
            nn.Linear(final_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        ).to(device)
        ssl_model.classification_head = nn.Linear(final_embed_dim, num_classes).to(device)

        dummy_rgb = torch.randn(2, 3, 260, 260).to(device)
        _, logits = ssl_model(dummy_rgb)
        logger.info(f"Model output shape: {logits.shape}")

        criterion = EnhancedFocalLoss(gamma=1.5, alpha=class_weights, label_smoothing=0.2)
        consistency_criterion = CrossModalConsistencyLoss(temperature=0.07)
        optimizer = optim.AdamW(ssl_model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_model_path = train_finetune_model(
            ssl_model, train_loader, val_loader, criterion, consistency_criterion,
            optimizer, scheduler, num_epochs=50, device=device,
            train_mean=train_mean, train_std=train_std, class_names=class_names
        )

        ssl_model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, test_f1, per_class_acc, per_class_f1, test_cm = validate(
            ssl_model, test_loader, criterion, device, train_mean, train_std, class_names
        )

        logger.info(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        logger.info(f"Test Per-class accuracy: {per_class_acc}")
        logger.info(f"Test Per-class F1: {per_class_f1}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(PHASE4_SAVE_PATH, 'confusion_matrix.png'))
        plt.close()

        final_model_path = os.path.join(PHASE4_SAVE_PATH, "finetuned_final.pth")
        torch.save({
            'model_state_dict': ssl_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_metrics': {'loss': test_loss, 'accuracy': test_acc, 'f1_score': test_f1,
                            'per_class_acc': per_class_acc, 'per_class_f1': per_class_f1,
                            'confusion_matrix': test_cm},
            'class_names': class_names
        }, final_model_path)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    main()