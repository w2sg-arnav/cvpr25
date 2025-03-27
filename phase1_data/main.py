# main.py
import torch
from torch.utils.data import DataLoader
from dataset import CottonLeafDataset
from transforms import get_transforms
from config import ORIGINAL_DATASET_ROOT, AUGMENTED_DATASET_ROOT
import logging
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
from progression import DiseaseProgressionSimulator

# Create a directory for saving visualizations
VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def save_visualization(original_img: torch.Tensor, progressed_imgs: dict, idx: int):
    """
    Save a visualization of the original and progressed images.
    
    Args:
        original_img (torch.Tensor): Original RGB image tensor [C, H, W].
        progressed_imgs (dict): Dictionary of stage:progressed_image_tensor.
        idx (int): Index of the image for naming the saved file.
    """
    # Denormalize the image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    original_img = original_img * std + mean  # Undo normalization
    original_img = original_img.clamp(0, 1)  # Ensure values are in [0, 1]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_img.permute(1, 2, 0).numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Plot each progressed image
    for i, (stage, img) in enumerate(progressed_imgs.items()):
        img = img * std + mean  # Denormalize
        img = img.clamp(0, 1)
        axes[i + 1].imshow(img.permute(1, 2, 0).numpy())
        axes[i + 1].set_title(f"Stage: {stage}")
        axes[i + 1].axis('off')
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"progression_sample_{idx}.png"))
    plt.close(fig)
    logging.info(f"Saved visualization for sample {idx} to {VISUALIZATION_DIR}/progression_sample_{idx}.png")

def test_dataset(root_dir: str, apply_progression: bool = False):
    """Test loading and processing a dataset."""
    train_transforms = get_transforms(train=True)
    dataset = CottonLeafDataset(
        root_dir=root_dir,
        transform=train_transforms,
        apply_progression=apply_progression
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    for rgb, spectral, labels, stages in dataloader:
        logging.info(f"Dataset: {root_dir}")
        logging.info(f"Batch: RGB {rgb.shape}, Spectral {spectral.shape}, Labels {labels}, Stages {stages}")
        
        # If progression is applied, visualize the effect on the first image in the batch
        if apply_progression:
            simulator = DiseaseProgressionSimulator()
            original_img = rgb[0]  # First image in the batch
            original_pil = T.ToPILImage()(original_img)  # Convert to PIL for progression
            
            # Apply progression for each stage
            progressed_imgs = {}
            for stage in ['early', 'mid', 'advanced']:
                progressed_pil = simulator.apply(original_pil, stage)
                progressed_tensor = train_transforms(progressed_pil)
                progressed_imgs[stage] = progressed_tensor
            
            # Save visualization
            save_visualization(original_img, progressed_imgs, idx=0)
        break  # Test one batch

def main():
    # Test original dataset with on-the-fly progression simulation
    logging.info("Testing Original Dataset with progression simulation...")
    test_dataset(ORIGINAL_DATASET_ROOT, apply_progression=True)
    
    # Test augmented dataset without additional progression
    logging.info("Testing Augmented Dataset...")
    test_dataset(AUGMENTED_DATASET_ROOT, apply_progression=False)

if __name__ == "__main__":
    main()