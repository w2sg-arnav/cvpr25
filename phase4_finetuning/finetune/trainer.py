# phase4_finetuning/finetune/trainer.py
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Tuple, Dict
from config import FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE
from utils.metrics import compute_metrics
import logging
from models.hvt import DiseaseAwareHVT
from models.baseline import InceptionV3Baseline

class Finetuner:
    """Finetuner for supervised learning."""
    
    def __init__(self, model: nn.Module, augmentations, device: str = "cpu"):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=FINETUNE_LR)
        self.device = device
    
    def train_step(self, rgb: torch.Tensor, spectral: torch.Tensor, labels: torch.Tensor) -> float:
        """Perform one training step."""
        self.model.train()
        rgb, spectral, labels = rgb.to(self.device), spectral.to(self.device), labels.to(self.device)
        
        # Apply augmentations
        rgb, spectral = self.augmentations(rgb, spectral)
        
        # Forward pass (handle different model types)
        if isinstance(self.model, DiseaseAwareHVT):
            logits = self.model(rgb, spectral, pretrain=False)
        elif isinstance(self.model, InceptionV3Baseline):
            logits = self.model(rgb, spectral)  # Pass rgb and spectral separately
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        
        loss = self.criterion(logits, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, rgb: torch.Tensor, spectral: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Evaluate the model."""
        self.model.eval()
        rgb, spectral, labels = rgb.to(self.device), spectral.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, DiseaseAwareHVT):
                logits = self.model(rgb, spectral, pretrain=False)
            elif isinstance(self.model, InceptionV3Baseline):
                logits = self.model(rgb, spectral)  # Pass rgb and spectral separately
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            
            metrics = compute_metrics(logits, labels)
        
        return metrics
    
    def finetune(self, rgb: torch.Tensor, spectral: torch.Tensor, labels: torch.Tensor, epochs: int = FINETUNE_EPOCHS):
        """Finetune the model."""
        for epoch in range(epochs):
            loss = self.train_step(rgb, spectral, labels)
            metrics = self.evaluate(rgb, spectral, labels)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    def save_model(self, path: str):
        """Save the fine-tuned model weights."""
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")