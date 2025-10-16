import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from pathlib import Path

class ImageFolderWithLandmarks(Dataset):
    """
    Dataset que carrega:
    - Imagem alinhada
    - Label da identidade
    - Landmarks normalizados (para loss auxiliar)
    """
    
    def __init__(self, root, landmarks_json, transform=None):
        self.root = root
        self.transform = transform
        
        # Carrega landmarks
        with open(landmarks_json, 'r') as f:
            self.landmarks_data = json.load(f)
        
        # Constrói lista de samples
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        root_path = Path(root)
        class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
            
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))
            
            for img_file in image_files:
                key = f"{class_name}/{img_file.name}"
                
                # Apenas adiciona se tiver landmarks
                if key in self.landmarks_data:
                    self.samples.append({
                        'path': str(img_file),
                        'label': idx,
                        'landmarks': self.landmarks_data[key],
                        'key': key
                    })
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Carrega imagem
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Landmarks como tensor
        landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
        
        label = sample['label']
        
        return img, label, landmarks


def create_validation_split_with_landmarks(dataset, val_split=0.1, seed=42):
    """
    Split considerando landmarks
    """
    from torch.utils.data import Subset
    import random
    
    random.seed(seed)
    
    # Organiza por classe
    class_indices = {}
    for idx, sample in enumerate(dataset.samples):
        label = sample['label']
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_split))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset