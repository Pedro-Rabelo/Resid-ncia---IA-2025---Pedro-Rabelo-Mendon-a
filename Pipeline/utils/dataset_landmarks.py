"""
Dataset loader for VGGFace2 with facial landmarks support.
Optimized exclusively for VGGFace2 112x112 aligned faces.

Author: Adapted for VGGFace2 training pipeline
Date: 2025
"""

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class ImageFolderWithLandmarks(Dataset):
    """
    Dataset loader for VGGFace2 with facial landmarks.
    
    Expected structure:
        root/
        ├── identity_1/
        │   ├── img_001.jpg
        │   └── img_002.jpg
        └── identity_2/
            └── img_003.jpg
    
    Landmarks JSON format:
        {
            "identity_1/img_001.jpg": [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5],
            ...
        }
    
    Args:
        root (str): Path to VGGFace2 aligned images directory
        landmarks_json (str): Path to landmarks JSON file
        transform (callable, optional): Transform to apply to images
        min_images_per_class (int): Minimum images per identity (default: 2)
    
    Returns:
        tuple: (image_tensor, label, landmarks_tensor)
    """
    
    def __init__(
        self,
        root: str,
        landmarks_json: str,
        transform: Optional[callable] = None,
        min_images_per_class: int = 2
    ):
        self.root = Path(root)
        self.transform = transform
        self.min_images_per_class = min_images_per_class
        
        # Validate paths
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {root}")
        
        if not Path(landmarks_json).exists():
            raise FileNotFoundError(f"Landmarks JSON not found: {landmarks_json}")
        
        # Load landmarks
        print(f"Loading landmarks from: {landmarks_json}")
        with open(landmarks_json, 'r') as f:
            self.landmarks_data = json.load(f)
        
        print(f"Landmarks loaded: {len(self.landmarks_data):,} entries")
        
        # Build dataset
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        self._build_dataset()
        self._filter_small_classes()
        
        print(f"Dataset ready: {len(self.samples):,} samples from {len(self.classes):,} identities")
    
    def _build_dataset(self):
        """Build sample list from directory structure"""
        root_path = Path(self.root)
        class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        print(f"Scanning {len(class_dirs):,} identity folders...")
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
            
            # Find all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(class_dir.glob(ext))
            
            # Add samples with landmarks
            for img_file in image_files:
                key = f"{class_name}/{img_file.name}"
                
                if key in self.landmarks_data:
                    self.samples.append({
                        'path': str(img_file),
                        'label': idx,
                        'landmarks': self.landmarks_data[key],
                        'key': key,
                        'class_name': class_name
                    })
    
    def _filter_small_classes(self):
        """Remove identities with fewer than min_images_per_class"""
        if self.min_images_per_class <= 1:
            return
        
        print(f"Filtering identities with < {self.min_images_per_class} images...")
        
        # Count samples per class
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Find valid classes
        valid_classes = {
            cls for cls, count in class_counts.items() 
            if count >= self.min_images_per_class
        }
        
        # Filter samples
        original_count = len(self.samples)
        self.samples = [
            s for s in self.samples 
            if s['class_name'] in valid_classes
        ]
        
        # Rebuild class mapping
        self.classes = sorted(list(valid_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Update labels
        for sample in self.samples:
            sample['label'] = self.class_to_idx[sample['class_name']]
        
        removed = original_count - len(self.samples)
        removed_classes = len(class_counts) - len(valid_classes)
        
        print(f"Removed {removed:,} samples from {removed_classes:,} identities")
    
    def get_num_classes(self) -> int:
        """Return number of unique identities"""
        return len(self.classes)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Load and return a sample
        
        Returns:
            image: Transformed image tensor [3, H, W]
            label: Identity class index (int)
            landmarks: Normalized landmarks tensor [10] - [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
        """
        sample = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {sample['path']}: {e}")
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Convert landmarks to tensor
        landmarks = torch.tensor(sample['landmarks'], dtype=torch.float32)
        
        # Label
        label = sample['label']
        
        return img, label, landmarks
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample (useful for debugging)"""
        return {
            'path': self.samples[idx]['path'],
            'class_name': self.samples[idx]['class_name'],
            'label': self.samples[idx]['label'],
            'key': self.samples[idx]['key']
        }


def create_validation_split_with_landmarks(
    dataset: ImageFolderWithLandmarks,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Split dataset into train/validation preserving class distribution.
    
    Ensures each identity appears in both train and validation sets
    (if it has enough samples).
    
    Args:
        dataset: ImageFolderWithLandmarks instance
        val_split: Fraction of data for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset: Subset for training
        val_dataset: Subset for validation
    """
    random.seed(seed)
    
    # Organize indices by class
    class_indices = {}
    for idx, sample in enumerate(dataset.samples):
        label = sample['label']
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # Split each class
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_split))
        
        # Ensure at least 1 sample in validation if possible
        if split_point == len(indices) and len(indices) > 1:
            split_point = len(indices) - 1
        
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"\nDataset split summary:")
    print(f"  Training samples:   {len(train_indices):,}")
    print(f"  Validation samples: {len(val_indices):,}")
    print(f"  Split ratio:        {(1-val_split)*100:.1f}% / {val_split*100:.1f}%")
    
    return train_dataset, val_dataset


def verify_dataset_integrity(
    root: str,
    landmarks_json: str,
    sample_size: int = 100
) -> Dict:
    """
    Verify dataset integrity by checking random samples.
    
    Args:
        root: Path to dataset root
        landmarks_json: Path to landmarks JSON
        sample_size: Number of random samples to check
    
    Returns:
        dict: Statistics about dataset integrity
    """
    print(f"Verifying dataset integrity...")
    
    dataset = ImageFolderWithLandmarks(
        root=root,
        landmarks_json=landmarks_json,
        transform=None
    )
    
    stats = {
        'total_samples': len(dataset),
        'num_classes': dataset.get_num_classes(),
        'checked_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'errors': []
    }
    
    # Check random samples
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    
    for idx in indices:
        try:
            img, label, landmarks = dataset[idx]
            
            # Validate shapes
            assert landmarks.shape == (10,), f"Invalid landmarks shape: {landmarks.shape}"
            assert 0 <= label < dataset.get_num_classes(), f"Invalid label: {label}"
            
            stats['valid_samples'] += 1
        except Exception as e:
            stats['invalid_samples'] += 1
            stats['errors'].append(f"Sample {idx}: {str(e)}")
        
        stats['checked_samples'] += 1
    
    print(f"\nIntegrity check results:")
    print(f"  Total samples:   {stats['total_samples']:,}")
    print(f"  Checked samples: {stats['checked_samples']}")
    print(f"  Valid samples:   {stats['valid_samples']}")
    print(f"  Invalid samples: {stats['invalid_samples']}")
    
    if stats['errors']:
        print(f"\nErrors found:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    return stats


# Example usage
if __name__ == '__main__':
    # Test dataset loading
    print("="*70)
    print("TESTING VGGFACE2 DATASET LOADER")
    print("="*70 + "\n")
    
    try:
        dataset = ImageFolderWithLandmarks(
            root='data/train/vggface2_aligned_112x112',
            landmarks_json='data/train/vggface2_landmarks.json',
            transform=None,
            min_images_per_class=2
        )
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset):,}")
        print(f"   Total identities: {dataset.get_num_classes():,}")
        
        # Test sample loading
        img, label, landmarks = dataset[0]
        print(f"\n✅ Sample loaded successfully!")
        print(f"   Image shape: {img.size}")
        print(f"   Label: {label}")
        print(f"   Landmarks shape: {landmarks.shape}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")