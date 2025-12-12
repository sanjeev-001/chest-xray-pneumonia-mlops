"""
Image Preprocessor for chest X-ray images
Handles resizing, normalization, and augmentation
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import albumentations as A
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing"""
    target_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    maintain_aspect_ratio: bool = True
    padding_color: int = 0  # Black padding


@dataclass
class ProcessingResult:
    """Result of image processing"""
    success: bool
    processed_image: Optional[np.ndarray] = None
    original_shape: Optional[Tuple[int, int]] = None
    final_shape: Optional[Tuple[int, int]] = None
    processing_steps: List[str] = None
    error_message: Optional[str] = None


class ImagePreprocessor:
    """Handles preprocessing of chest X-ray images"""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )
        
        # PyTorch transforms for tensor conversion
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def preprocess_image(self, image_path: Union[str, Path], 
                        return_tensor: bool = False) -> ProcessingResult:
        """
        Preprocess a single chest X-ray image
        
        Args:
            image_path: Path to the image file
            return_tensor: Whether to return PyTorch tensor
            
        Returns:
            ProcessingResult with processed image and metadata
        """
        processing_steps = []
        
        try:
            # Load image
            image_path = Path(image_path)
            if not image_path.exists():
                return ProcessingResult(
                    success=False,
                    error_message=f"Image file not found: {image_path}"
                )
            
            # Read image using OpenCV
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return ProcessingResult(
                    success=False,
                    error_message=f"Failed to load image: {image_path}"
                )
            
            original_shape = image.shape
            processing_steps.append("loaded_grayscale")
            
            # Apply CLAHE for contrast enhancement
            if self.config.apply_clahe:
                image = self.clahe.apply(image)
                processing_steps.append("applied_clahe")
            
            # Resize image
            if self.config.maintain_aspect_ratio:
                image = self._resize_with_aspect_ratio(image)
                processing_steps.append("resized_with_aspect_ratio")
            else:
                image = cv2.resize(image, self.config.target_size)
                processing_steps.append("resized_direct")
            
            # Convert to 3-channel for compatibility with pre-trained models
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                processing_steps.append("converted_to_rgb")
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            processing_steps.append("normalized_pixels")
            
            # Convert to tensor if requested
            if return_tensor:
                # Convert numpy array to PIL Image for torchvision transforms
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
                image = self.to_tensor(pil_image)
                processing_steps.append("converted_to_tensor")
            
            return ProcessingResult(
                success=True,
                processed_image=image,
                original_shape=original_shape,
                final_shape=image.shape if not return_tensor else tuple(image.shape),
                processing_steps=processing_steps
            )
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_steps=processing_steps
            )
    
    def preprocess_batch(self, image_paths: List[Union[str, Path]], 
                        return_tensors: bool = False) -> List[ProcessingResult]:
        """
        Preprocess a batch of images
        
        Args:
            image_paths: List of paths to image files
            return_tensors: Whether to return PyTorch tensors
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for image_path in image_paths:
            result = self.preprocess_image(image_path, return_tensor=return_tensors)
            results.append(result)
        
        return results
    
    def get_augmentation_pipeline(self, training: bool = True) -> A.Compose:
        """
        Get augmentation pipeline for training or validation
        
        Args:
            training: Whether this is for training (applies augmentations)
            
        Returns:
            Albumentations composition pipeline
        """
        if training:
            # Training augmentations
            return A.Compose([
                A.Rotate(limit=10, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Resize(
                    height=self.config.target_size[0],
                    width=self.config.target_size[1]
                ),
                A.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                )
            ])
        else:
            # Validation/test - only resize and normalize
            return A.Compose([
                A.Resize(
                    height=self.config.target_size[0],
                    width=self.config.target_size[1]
                ),
                A.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                )
            ])
    
    def apply_augmentation(self, image: np.ndarray, 
                          augmentation_pipeline: A.Compose) -> np.ndarray:
        """
        Apply augmentation pipeline to an image
        
        Args:
            image: Input image as numpy array
            augmentation_pipeline: Albumentations pipeline
            
        Returns:
            Augmented image
        """
        # Ensure image is in correct format for albumentations
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        augmented = augmentation_pipeline(image=image)
        return augmented['image']
    
    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio with padding
        
        Args:
            image: Input image
            
        Returns:
            Resized image with padding
        """
        h, w = image.shape[:2]
        target_h, target_w = self.config.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full(
            (target_h, target_w) + image.shape[2:],
            self.config.padding_color,
            dtype=image.dtype
        )
        
        # Calculate padding offsets to center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        if len(image.shape) == 2:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
        
        return padded
    
    def calculate_statistics(self, image_paths: List[Union[str, Path]]) -> Dict:
        """
        Calculate dataset statistics for normalization
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary with mean, std, and other statistics
        """
        pixel_values = []
        valid_images = 0
        
        for image_path in image_paths:
            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # Normalize to [0, 1]
                    normalized = image.astype(np.float32) / 255.0
                    pixel_values.extend(normalized.flatten())
                    valid_images += 1
            except Exception as e:
                logger.warning(f"Failed to process {image_path} for statistics: {e}")
        
        if not pixel_values:
            raise ValueError("No valid images found for statistics calculation")
        
        pixel_array = np.array(pixel_values)
        
        return {
            "mean": float(np.mean(pixel_array)),
            "std": float(np.std(pixel_array)),
            "min": float(np.min(pixel_array)),
            "max": float(np.max(pixel_array)),
            "valid_images": valid_images,
            "total_pixels": len(pixel_values)
        }
    
    def validate_preprocessing_config(self) -> List[str]:
        """
        Validate preprocessing configuration
        
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if self.config.target_size[0] <= 0 or self.config.target_size[1] <= 0:
            warnings.append("Target size must be positive")
        
        if len(self.config.normalize_mean) != 3 or len(self.config.normalize_std) != 3:
            warnings.append("Normalization mean and std must have 3 values for RGB")
        
        if self.config.clahe_clip_limit <= 0:
            warnings.append("CLAHE clip limit must be positive")
        
        if any(x <= 0 for x in self.config.clahe_tile_grid_size):
            warnings.append("CLAHE tile grid size must be positive")
        
        return warnings