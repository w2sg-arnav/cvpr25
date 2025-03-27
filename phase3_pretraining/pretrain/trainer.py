import torch
import torch.nn as nn
from torch.optim import Optimizer
from config import PRETRAIN_LR, ACCUM_STEPS

# Custom LARS Optimizer
class LARS(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=1e-4, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param = p.data

                # Apply weight decay
                if group['weight_decay'] > 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                # Compute local learning rate using LARS
                param_norm = torch.norm(param)
                grad_norm = torch.norm(grad)
                trust_ratio = group['eta'] * param_norm / (grad_norm + 1e-6)
                trust_ratio = torch.clamp(trust_ratio, 0.0, 1.0)

                # Update momentum
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(param)
                momentum_buffer = self.state[p]['momentum_buffer']
                momentum_buffer.mul_(group['momentum']).add_(grad, alpha=1.0)

                # Update parameter
                scaled_lr = group['lr'] * trust_ratio
                param.add_(momentum_buffer, alpha=-scaled_lr)

class Pretrainer:
    def __init__(self, model: nn.Module, augmentations, loss_fn, device: str):
        self.model = model
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.device = device
        
        # Use LARS optimizer
        self.optimizer = LARS(
            self.model.parameters(),
            lr=PRETRAIN_LR,
            momentum=0.9,
            weight_decay=1e-4,
            eta=0.001
        )
        
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