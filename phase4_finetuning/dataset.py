# phase4_finetuning/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import logging

class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., '/teamspace/studios/this_studio/cvpr25/data/sar_cld_2024').
            img_size (tuple): Desired image size (height, width), e.g., (384, 384).
            split (str): 'train' or 'val' to specify the dataset split.
            train_split (float): Fraction of data to use for training (e.g., 0.8 for 80% train, 20% val).
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        
        # Define class names and labels
        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf", 
            "Herbicide Growth Damage", "Leaf Hopper Jassids", 
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Check if root_dir exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")
        
        logging.info(f"Loading dataset from: {root_dir}")
        
        # Traverse both Original and Augmented datasets
        for dataset_type in ["Original Dataset", "Augmented Dataset"]:
            dataset_path = os.path.join(root_dir, dataset_type)
            if not os.path.exists(dataset_path):
                logging.warning(f"Dataset path does not exist, skipping: {dataset_path}")
                continue
            
            logging.info(f"Scanning dataset: {dataset_type}")
            for class_name in self.classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    logging.warning(f"Class path does not exist, skipping: {class_path}")
                    continue
                
                logging.info(f"Scanning class: {class_name}")
                for img_name in os.listdir(class_path):
                    # Make file extension check case-insensitive
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        # Convert to numpy arrays for splitting
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        # Check if any images were found
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the dataset at {root_dir}. Please check the directory structure and file extensions.")
        
        logging.info(f"Total images found: {len(self.image_paths)}")
        
        # Split into train and validation sets
        indices = np.arange(len(self.image_paths))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * self.train_split)
        if self.split == "train":
            self.indices = indices[:split_idx]
        else:  # val
            self.indices = indices[split_idx:]
        
        logging.info(f"{self.split.capitalize()} split size: {len(self.indices)} samples")
        
        # Define preprocessing transforms
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])
        
        # Spectral transform (if spectral data is single-channel)
        self.spectral_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from the split indices
        actual_idx = self.indices[idx]
        
        # Load image
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        
        # Load image as a numpy array to check channels
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Extract RGB (first 3 channels)
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            rgb = Image.fromarray(img_array[:, :, :3])  # Take first 3 channels as RGB
        else:
            rgb = img.convert("RGB")
        rgb = self.transform(rgb)
        
        # Extract spectral data (if available, e.g., 4th channel)
        if img_array.ndim == 3 and img_array.shape[2] > 3:
            # Use the 4th channel as spectral data
            spectral = Image.fromarray(img_array[:, :, 3])
            spectral = self.spectral_transform(spectral)
        else:
            # Fallback: Create a dummy spectral channel
            spectral = torch.randn(1, self.img_size[0], self.img_size[1])
            spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min())  # Normalize to [0, 1]
        
        return rgb, spectral, torch.tensor(label, dtype=torch.long)

    def get_class_names(self):
        return self.classes