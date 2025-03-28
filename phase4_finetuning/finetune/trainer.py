import torch
import torch.nn as nn
import logging
from utils.metrics import compute_metrics

class Finetuner:
    def __init__(self, model, augmentations, device, class_weights=None, label_smoothing=0.0, 
                 optimizer_class=torch.optim.AdamW, optimizer_params=None):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=label_smoothing  # Added label smoothing support
        )
        # Use optimizer_class and optimizer_params, with defaults if None
        if optimizer_params is None:
            optimizer_params = {"lr": 3e-4, "weight_decay": 0.01}
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
    
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
    
    def evaluate(self, rgb, labels, custom_preds=None):
        rgb, labels = rgb.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            if custom_preds is None:
                outputs = self.model(rgb)
                _, preds = torch.max(outputs, 1)
            else:
                # Use custom predictions (e.g., for TTA)
                _, preds = torch.max(custom_preds, 1)
        
        metrics = compute_metrics(preds.cpu().numpy(), labels.cpu().numpy())
        return metrics
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")