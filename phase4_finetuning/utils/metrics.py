# phase4_finetuning/utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

def compute_metrics(preds, labels):
    """
    Compute evaluation metrics (accuracy and F1 score) for predictions and labels.
    
    Args:
        preds: Predicted class indices (numpy array or torch tensor, 1D or 2D)
        labels: Ground truth labels (numpy array or torch tensor, 1D)
    
    Returns:
        dict: Dictionary containing accuracy and F1 score
    """
    # Convert to numpy if inputs are torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # If preds is 2D (e.g., raw logits or probabilities), take argmax to get class indices
    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1)  # Use axis for numpy arrays
    
    # Ensure labels are 1D
    if labels.ndim != 1:
        labels = np.argmax(labels, axis=1) if labels.ndim == 2 else labels.flatten()
    
    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')  # Macro F1 for multi-class
    
    return {"accuracy": accuracy, "f1": f1}