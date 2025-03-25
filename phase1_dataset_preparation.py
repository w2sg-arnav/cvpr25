import os
import torch
import logging
from data_utils import analyze_dataset, CottonLeafDataset, load_and_split_dataset, load_augmented_dataset, visualize_batch, get_transforms, create_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Step 1: Set Up the Environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define global variables
DATA_ROOT = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
SAVE_PATH = "./phase1_checkpoints"

if not os.path.exists(DATA_ROOT):
    logger.error(f"Dataset root {DATA_ROOT} not found.")
    raise FileNotFoundError(f"Dataset root {DATA_ROOT} not found.")

os.makedirs(SAVE_PATH, exist_ok=True)

# Step 2: Analyze Datasets
original_stats = analyze_dataset(os.path.join(DATA_ROOT, "Original Dataset"), "Original", save_path=SAVE_PATH)
augmented_stats = analyze_dataset(os.path.join(DATA_ROOT, "Augmented Dataset"), "Augmented", save_path=SAVE_PATH)

# Step 3: Define Transformations
train_transforms, rare_transforms, val_test_transforms = get_transforms()

# Step 4: Load and Split Dataset
SPECTRAL_DIR = os.path.join(DATA_ROOT, "Spectral Dataset")
SIMULATE_SPECTRAL = True # Set to True if you want to simulate data if the directory does not exist
if not os.path.exists(SPECTRAL_DIR):
    logger.warning("Spectral directory not found.")
    SPECTRAL_DIR = None # Set to None to simulate the spectral data

original_train_dataset, original_val_dataset, original_test_dataset, class_names, rare_classes = load_and_split_dataset(
    os.path.join(DATA_ROOT, "Original Dataset"),
    train_transforms,
    rare_transforms,
    val_test_transforms,
    spectral_dir=SPECTRAL_DIR,
    simulate_spectral=SIMULATE_SPECTRAL,
    corrupt_images=original_stats['corrupt_images']
)

# Step 5: Load Augmented Dataset
augmented_dataset = load_augmented_dataset(
    os.path.join(DATA_ROOT, "Augmented Dataset"),
    train_transforms,
    rare_transforms,
    rare_classes,
    spectral_dir=SPECTRAL_DIR,
    simulate_spectral=SIMULATE_SPECTRAL,
    corrupt_images=augmented_stats['corrupt_images']
)

# Step 6: Combine Datasets for Training
combined_train_dataset = CottonLeafDataset(
    original_train_dataset.samples + augmented_dataset.samples,
    transform=train_transforms,
    rare_transform=rare_transforms,
    rare_classes=rare_classes,
    spectral_dir=SPECTRAL_DIR,
    simulate_spectral=SIMULATE_SPECTRAL
)

# Step 7: Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    combined_train_dataset,
    original_val_dataset,
    original_test_dataset,
    batch_size=32
)

# Step 8: Visualize and Save
has_multimodal = SPECTRAL_DIR is not None or SIMULATE_SPECTRAL
visualize_batch(train_loader, title="Training Samples", has_multimodal=has_multimodal, save_path=SAVE_PATH)
visualize_batch(val_loader, title="Validation Samples", has_multimodal=has_multimodal, save_path=SAVE_PATH)
visualize_batch(test_loader, title="Test Samples", has_multimodal=has_multimodal, save_path=SAVE_PATH)

# Step 9: Save Preprocessed Data with Metadata
checkpoint_data = {
    'train_dataset': combined_train_dataset,
    'val_dataset': original_val_dataset,
    'test_dataset': original_test_dataset,
    'class_names': class_names,
    'rare_classes': rare_classes,
    'original_stats': original_stats,
    'augmented_stats': augmented_stats,
    'has_multimodal': has_multimodal,
    'spectral_dir': SPECTRAL_DIR,  # Save spectral data path
    'simulate_spectral': SIMULATE_SPECTRAL, # If spectral data should be simulated if directory does not exist
    'metadata': {
        'version': '1.1',
        'transforms': {
            'train_transforms': str(train_transforms),
            'rare_transforms': str(rare_transforms),
            'val_test_transforms': str(val_test_transforms)
        }
    }
}
torch.save(checkpoint_data, os.path.join(SAVE_PATH, 'phase1_preprocessed_data.pth'))

# Step 10: Summary Log
logger.info("Phase 1 Summary:")
logger.info(f"Training set size: {len(combined_train_dataset)}")
logger.info(f"Validation set size: {len(original_val_dataset)}")
logger.info(f"Test set size: {len(original_test_dataset)}")
logger.info(f"Number of classes: {len(class_names)}")
logger.info(f"Class names: {class_names}")
logger.info(f"Rare classes: {[class_names[i] for i in rare_classes]}")
logger.info(f"Class imbalance ratio (original): {original_stats['class_imbalance_ratio']:.2f}")
logger.info(f"Multimodal support: {has_multimodal}")
logger.info(f"Simulate Spectral: {SIMULATE_SPECTRAL}") # Log the data simulation status
logger.info("Phase 1 completed: Dataset prepared with class-specific augmentations and multimodal integration set up.")