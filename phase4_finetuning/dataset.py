# phase4_finetuning/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import logging
from collections import Counter

class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8):
        """
        Args:
            root_dir (str): Root directory of the dataset.
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
        
        # Log class distribution
        class_counts = Counter(self.labels)
        logging.info("Class distribution:")
        for idx, count in class_counts.items():
            class_name = self.classes[idx]
            logging.info(f"Class {class_name}: {count} samples ({count/len(self.labels)*100:.2f}%)")
        
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
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from the split indices
        actual_idx = self.indices[idx]
        
        # Load image
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        
        # Load image as RGB
        img = Image.open(img_path).convert("RGB")
        rgb = self.transform(img)
        
        return rgb, torch.tensor(label, dtype=torch.long)

    def get_class_names(self):
        return self.classes
    
    def get_class_weights(self):
        # Compute class weights for imbalanced dataset
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        weights = [total_samples / (len(self.classes) * class_counts[i]) for i in range(len(self.classes))]
        return torch.tensor(weights, dtype=torch.float)