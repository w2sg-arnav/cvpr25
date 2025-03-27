# phase3_pretraining/pretrain/trainer.py
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Tuple
from config import PRETRAIN_LR, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE
import logging

class Pretrainer:
    """Pretrainer for self-supervised learning with SimCLR."""
    
    def __init__(self, model: nn.Module, augmentations, loss_fn: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.optimizer = Adam(self.model.parameters(), lr=PRETRAIN_LR)
        self.device = device
    
    def train_step(self, rgb: torch.Tensor, spectral: torch.Tensor) -> float:
        """Perform one training step."""
        self.model.train()
        rgb, spectral = rgb.to(self.device), spectral.to(self.device)
        
        # Generate two augmented views
        rgb_view1, spectral_view1, rgb_view2, spectral_view2 = self.augmentations(rgb, spectral)
        
        # Forward pass for both views
        features1 = self.model(rgb_view1, spectral_view1, pretrain=True)
        features2 = self.model(rgb_view2, spectral_view2, pretrain=True)
        
        # Compute loss
        loss = self.loss_fn(features1, features2)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def pretrain(self, rgb: torch.Tensor, spectral: torch.Tensor, epochs: int = PRETRAIN_EPOCHS):
        """Pretrain the model."""
        for epoch in range(epochs):
            loss = self.train_step(rgb, spectral)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def save_model(self, path: str):
        """Save the pretrained model weights."""
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")