"""
Chest X-Ray Dataset Loader
Optimized for medical imaging with proper preprocessing and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class ChestXrayDataset(Dataset):
    """
    Dataset class for chest X-ray images
    Handles NORMAL vs PNEUMONIA classification with medical-grade preprocessing
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (380, 380),  # EfficientNet-B4 optimal size
        class_names: List[str] = None
    ):
        """
        Initialize chest X-ray dataset
        
        Args:
            data_dir: Path to data directory containing class subdirectories
            transform: Optional transform to apply to images
            image_size: Target image size (height, width)
            class_names: List of class names (default: ['NORMAL', 'PNEUMONIA'])
        """
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        self.class_names = class_names or ['NORMAL', 'PNEUMONIA']
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_counts = {}
        
        self._load_dataset()
        
        # Setup default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms()
        
        logger.info(f"Loaded dataset from {data_dir}")
        logger.info(f"Total samples: {len(self.image_paths)}")
        logger.info(f"Class distribution: {self.class_counts}")
    
    def _load_dataset(self):
        """Load all image paths and labels from the directory structure"""
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all image files in this class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            class_images = []
            
            for ext in image_extensions:
                class_images.extend(list(class_dir.glob(f'*{ext}')))
                class_images.extend(list(class_dir.glob(f'*{ext.upper()}')))
            
            # Add to dataset
            class_idx = self.class_to_idx[class_name]
            for img_path in class_images:
                self.image_paths.append(str(img_path))
                self.labels.append(class_idx)
            
            self.class_counts[class_name] = len(class_images)
            logger.info(f"Found {len(class_images)} images in {class_name} class")
        
        # Handle augmented folders (like NORMAL_augmented)
        augmented_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir() and 'augmented' in d.name.lower()]
        
        for aug_dir in augmented_dirs:
            # Determine base class from directory name
            base_class = None
            for class_name in self.class_names:
                if class_name.lower() in aug_dir.name.lower():
                    base_class = class_name
                    break
            
            if base_class is None:
                logger.warning(f"Could not determine base class for {aug_dir}")
                continue
            
            # Get all image files in augmented directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            aug_images = []
            
            for ext in image_extensions:
                aug_images.extend(list(aug_dir.glob(f'*{ext}')))
                aug_images.extend(list(aug_dir.glob(f'*{ext.upper()}')))
            
            # Add to dataset with base class label
            class_idx = self.class_to_idx[base_class]
            for img_path in aug_images:
                self.image_paths.append(str(img_path))
                self.labels.append(class_idx)
            
            # Update class counts
            self.class_counts[f"{base_class}_augmented"] = len(aug_images)
            logger.info(f"Found {len(aug_images)} augmented images for {base_class} class")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms for medical images"""
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic quality checks
            if image.shape[0] < 32 or image.shape[1] < 32:
                raise ValueError(f"Image too small: {image.shape}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        
        # Load image and label
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            try:
                # Handle different transform types
                if hasattr(self.transform, '__call__'):
                    # Check if it's albumentations transform
                    if hasattr(self.transform, 'processors'):
                        # Albumentations transform
                        transformed = self.transform(image=image)
                        image = transformed['image']
                    else:
                        # Torchvision transform
                        image = self.transform(image)
                else:
                    # Fallback to default transform
                    image = self._get_default_transforms()(image)
            except Exception as e:
                logger.warning(f"Transform failed for {image_path}: {e}")
                # Use default transform as fallback
                image = self._get_default_transforms()(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training
        
        Returns:
            Tensor of class weights
        """
        
        # Count samples per class
        class_counts = torch.zeros(len(self.class_names))
        for label in self.labels:
            class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.labels)
        weights = total_samples / (len(self.class_names) * class_counts)
        
        # Handle zero counts
        weights[class_counts == 0] = 0.0
        
        logger.info(f"Calculated class weights: {weights.tolist()}")
        return weights
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a specific sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with sample information
        """
        
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range")
        
        image_path = Path(self.image_paths[idx])
        label = self.labels[idx]
        class_name = self.idx_to_class[label]
        
        # Get image info
        try:
            image = cv2.imread(str(image_path))
            if image is not None:
                height, width, channels = image.shape
                file_size = image_path.stat().st_size
            else:
                height = width = channels = file_size = 0
        except:
            height = width = channels = file_size = 0
        
        return {
            'index': idx,
            'image_path': str(image_path),
            'filename': image_path.name,
            'class_name': class_name,
            'class_index': label,
            'image_height': height,
            'image_width': width,
            'image_channels': channels,
            'file_size_bytes': file_size,
            'is_augmented': 'augmented' in str(image_path).lower()
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        
        stats = {
            'total_samples': len(self.image_paths),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_counts': {},
            'class_percentages': {},
            'image_statistics': {
                'mean_width': 0,
                'mean_height': 0,
                'min_width': float('inf'),
                'max_width': 0,
                'min_height': float('inf'),
                'max_height': 0
            }
        }
        
        # Calculate class distribution
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = sum(1 for label in self.labels if label == class_idx)
            stats['class_counts'][class_name] = count
            stats['class_percentages'][class_name] = (count / len(self.labels)) * 100
        
        # Sample image statistics (check first 100 images for performance)
        sample_size = min(100, len(self.image_paths))
        widths, heights = [], []
        
        for i in range(sample_size):
            try:
                image = cv2.imread(self.image_paths[i])
                if image is not None:
                    h, w = image.shape[:2]
                    widths.append(w)
                    heights.append(h)
            except:
                continue
        
        if widths and heights:
            stats['image_statistics'] = {
                'mean_width': np.mean(widths),
                'mean_height': np.mean(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'samples_analyzed': len(widths)
            }
        
        return stats


def create_data_loaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        dataset_path: Path to the dataset directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    dataset_path = Path(dataset_path)
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        data_dir=dataset_path / "train",
        transform=train_transform
    )
    
    val_dataset = ChestXrayDataset(
        data_dir=dataset_path / "val",
        transform=val_transform
    )
    
    test_dataset = ChestXrayDataset(
        data_dir=dataset_path / "test",
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info("Data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Analyze a dataset and return comprehensive statistics
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary with dataset analysis
    """
    
    analysis = {}
    dataset_path = Path(dataset_path)
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        
        if split_path.exists():
            try:
                dataset = ChestXrayDataset(data_dir=split_path)
                stats = dataset.get_dataset_statistics()
                analysis[split] = stats
                
                logger.info(f"{split.upper()} split analysis:")
                logger.info(f"  Total samples: {stats['total_samples']}")
                for class_name, count in stats['class_counts'].items():
                    percentage = stats['class_percentages'][class_name]
                    logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error analyzing {split} split: {e}")
                analysis[split] = {'error': str(e)}
        else:
            logger.warning(f"{split} split not found at {split_path}")
            analysis[split] = {'error': 'Split not found'}
    
    return analysis


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ChestXrayDataset...")
    
    # Test with sample data path
    dataset_path = "data/chest_xray_final"
    
    if Path(dataset_path).exists():
        # Test dataset creation
        train_dataset = ChestXrayDataset(
            data_dir=Path(dataset_path) / "train"
        )
        
        print(f"Dataset loaded: {len(train_dataset)} samples")
        print(f"Classes: {train_dataset.class_names}")
        print(f"Class counts: {train_dataset.class_counts}")
        
        # Test sample loading
        if len(train_dataset) > 0:
            sample_image, sample_label = train_dataset[0]
            print(f"Sample shape: {sample_image.shape}")
            print(f"Sample label: {sample_label} ({train_dataset.idx_to_class[sample_label]})")
            
            # Test sample info
            sample_info = train_dataset.get_sample_info(0)
            print(f"Sample info: {sample_info['filename']}, {sample_info['class_name']}")
        
        # Test class weights
        weights = train_dataset.get_class_weights()
        print(f"Class weights: {weights}")
        
        # Test dataset statistics
        stats = train_dataset.get_dataset_statistics()
        print(f"Dataset statistics: {stats['total_samples']} samples")
        
        print("✅ Dataset loading working correctly!")
    else:
        print(f"Dataset path not found: {dataset_path}")
        print("Please ensure the dataset is available for testing")