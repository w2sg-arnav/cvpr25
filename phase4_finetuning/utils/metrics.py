# phase4_finetuning/utils/metrics.py
import torch
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    Compute accuracy and F1 score.
    
    Args:
        preds (torch.Tensor): Predicted logits or probabilities [batch, num_classes].
        labels (torch.Tensor): Ground truth labels [batch].
    
    Returns:
        dict: Metrics including accuracy and F1 score.
    """
    preds = preds.argmax(dim=1).cpu().numpy()  # Get predicted class indices
    labels = labels.cpu().numpy()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "f1": f1}