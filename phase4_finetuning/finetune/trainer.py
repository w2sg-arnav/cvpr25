# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
import logging
from utils.metrics import compute_metrics

class Finetuner:
    def __init__(self, model, augmentations, device, class_weights=None, label_smoothing=0.0):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=label_smoothing  # Added label smoothing support
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)  # Use AdamW with weight decay
    
    def train_step(self, rgb, labels):
        # Forward pass
        rgb, labels = rgb.to(self.device), labels.to(self.device)
        
        # Apply augmentations
        rgb = self.augmentations(rgb)
        
        # Model prediction
        outputs = self.model(rgb)
        
        # Compute loss
        loss = self.criterion(outputs, labels)
        
        return loss
    
    def evaluate(self, rgb, labels):
        rgb, labels = rgb.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(rgb)
            _, preds = torch.max(outputs, 1)
        
        metrics = compute_metrics(preds.cpu().numpy(), labels.cpu().numpy())
        return metrics
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")