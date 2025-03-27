# main.py
import torch
import logging
from config import PROGRESSIVE_RESOLUTIONS, NUM_CLASSES
from models.hvt import DiseaseAwareHVT
from models.baseline import InceptionV3Baseline

def test_model(model, img_size: Tuple[int, int], batch_size: int = 4):
    """Test the model with dummy inputs."""
    rgb = torch.randn(batch_size, 3, img_size[0], img_size[1])
    spectral = torch.randn(batch_size, 1, img_size[0], img_size[1])
    
    model.eval()
    with torch.no_grad():
        logits = model(rgb, spectral)
        logging.info(f"Model: {model.__class__.__name__}, Input size: {img_size}, Output shape: {logits.shape}")
        assert logits.shape == (batch_size, NUM_CLASSES), f"Expected output shape {(batch_size, NUM_CLASSES)}, got {logits.shape}"

def main():
    # Test DiseaseAwareHVT with progressive resolutions
    for img_size in PROGRESSIVE_RESOLUTIONS:
        hvt_model = DiseaseAwareHVT(img_size=img_size)
        test_model(hvt_model, img_size)
    
    # Test Inception V3 baseline (using the largest resolution)
    inception_model = InceptionV3Baseline()
    test_model(inception_model, PROGRESSIVE_RESOLUTIONS[-1])

if __name__ == "__main__":
    main()