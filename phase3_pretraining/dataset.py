import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class SARCLD2024Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: tuple, split: str = "train", train_split: float = 0.8, normalize: bool = True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        self.normalize = normalize

        self.classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory does not exist: {root_dir}")

        logger.info(f"Loading dataset from: {root_dir}")

        for dataset_type in ["Original Dataset", "Augmented Dataset"]:
            dataset_path = os.path.join(root_dir, dataset_type)
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset path does not exist, skipping: {dataset_path}")
                continue

            logger.info(f"Scanning dataset: {dataset_type}")
            for class_name in self.classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    logger.warning(f"Class path does not exist, skipping: {class_path}")
                    continue

                logger.info(f"Scanning class: {class_name}")
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in the dataset at {root_dir}")

        class_counts = Counter(self.labels)
        logger.info("Class distribution:")
        for idx, count in class_counts.items():
            class_name = self.classes[idx]
            logger.info(f"Class {class_name}: {count} samples ({count/len(self.labels)*100:.2f}%)")

        logger.info(f"Total images found: {len(self.image_paths)}")

        indices = np.arange(len(self.image_paths))
        np.random.seed(42)
        np.random.shuffle(indices)

        split_idx = int(len(indices) * self.train_split)
        if self.split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        logger.info(f"{self.split.capitalize()} split size: {len(self.indices)} samples")

        transforms_list = [
            T.Resize(self.img_size),
            T.ToTensor(),
        ]
        if self.normalize:
            transforms_list.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_paths[actual_idx]
        label = self.labels[actual_idx]
        img = Image.open(img_path).convert("RGB")
        rgb = self.transform(img)

        #***REMOVE THIS***
        #rgb.requires_grad_(True)

        return rgb, torch.tensor(label, dtype=torch.long)