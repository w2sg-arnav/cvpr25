import torch
import torch.nn as nn
from torch.optim import Adam
from config import PRETRAIN_LR, ACCUM_STEPS
import logging

# Use a module-specific logger
logger = logging.getLogger(__name__)

class Pretrainer:
    def __init__(self, model: nn.Module, augmentations, loss_fn, device: str):
        self.model = model
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        
        # Use Adam optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=PRETRAIN_LR,
            weight_decay=1e-4
        )
        
        # Verify that all model parameters require gradients
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logger.warning(f"Parameter {name} does not require gradients. Enabling gradients.")
                param.requires_grad = True
        
        self.accum_steps = ACCUM_STEPS
        self.step_count = 0
    
    def train_step(self, rgb: torch.Tensor, spectral: torch.Tensor = None):
        self.model.train()
        rgb = rgb.to(self.device)
        if spectral is not None:
            spectral = spectral.to(self.device)
        
        # Apply augmentations to create two views
        rgb_view1, rgb_view2 = self.augmentations(rgb), self.augmentations(rgb)
        spectral_view1 = spectral_view2 = None
        if spectral is not None:
            spectral_view1, spectral_view2 = self.augmentations(spectral), self.augmentations(spectral)
        
        # Forward pass for both views
        features1 = self.model(rgb_view1, spectral_view1, pretrain=True)
        features2 = self.model(rgb_view2, spectral_view2, pretrain=True)
        
        # Compute InfoNCE loss
        loss = self.loss_fn(features1, features2) / self.accum_steps
        
        # Verify that the loss requires gradients
        if not loss.requires_grad:
            logger.error("Loss does not require gradients. Check the computation graph.")
            raise RuntimeError("Loss does not require gradients.")
        
        loss.backward()
        
        self.step_count += 1
        if self.step_count % self.accum_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accum_steps
    
    def pretrain(self, train_loader, total_epochs: int):
        for epoch in range(1, total_epochs + 1):
            total_loss = 0.0
            num_batches = len(train_loader)
            
            for batch_idx, (rgb, _) in enumerate(train_loader, 1):
                batch_loss = self.train_step(rgb, None)  # spectral=None
                total_loss += batch_loss
                
                if batch_idx % 10 == 0 or batch_idx == num_batches:
                    print(f"Epoch {epoch}/{total_epochs}, Batch {batch_idx}/{num_batches}, Batch Loss: {batch_loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{total_epochs}, Pretrain Loss: {avg_loss:.4f}")
            yield epoch
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)