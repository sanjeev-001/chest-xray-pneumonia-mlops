"""
Data Validator for chest X-ray images
Handles image quality checks and format validation
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import os

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    min_image_size: Tuple[int, int] = (64, 64)
    max_image_size: Tuple[int, int] = (4096, 4096)
    min_file_size_bytes: int = 1024  # 1KB
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB
    allowed_formats: List[str] = None
    min_brightness: float = 0.01
    max_brightness: float = 0.99
    min_contrast: float = 0.01
    blur_threshold: float = 100.0  # Laplacian variance threshold
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


@dataclass
class ValidationResult:
    """Result of image validation"""
    is_valid: bool
    file_path: Path
    file_size_bytes: int
    image_dimensions: Optional[Tuple[int, int]] = None
    format: Optional[str] = None
    brightness_score: Optional[float] = None
    contrast_score: Optional[float] = None
    blur_score: Optional[float] = None
    issues: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class DatasetValidationSummary:
    """Summary of dataset validation"""
    total_files: int
    valid_files: int
    invalid_files: int
    validation_results: List[ValidationResult]
    common_issues: Dict[str, int]
    dataset_statistics: Dict
    recommendations: List[str]


class DataValidator:
    """Validates chest X-ray images for quality and format compliance"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate_image(self, image_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a single image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ValidationResult with validation details
        """
        image_path = Path(image_path)
        issues = []
        warnings = []
        
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            file_path=image_path,
            file_size_bytes=0,
            issues=issues,
            warnings=warnings
        )
        
        try:
            # Check if file exists
            if not image_path.exists():
                issues.append("File does not exist")
                result.is_valid = False
                return result
            
            # Check file size
            file_size = image_path.stat().st_size
            result.file_size_bytes = file_size
            
            if file_size < self.config.min_file_size_bytes:
                issues.append(f"File too small: {file_size} bytes < {self.config.min_file_size_bytes}")
                result.is_valid = False
            
            if file_size > self.config.max_file_size_bytes:
                issues.append(f"File too large: {file_size} bytes > {self.config.max_file_size_bytes}")
                result.is_valid = False
            
            # Check file format
            file_extension = image_path.suffix.lower()
            result.format = file_extension
            
            if file_extension not in self.config.allowed_formats:
                issues.append(f"Unsupported format: {file_extension}")
                result.is_valid = False
                return result
            
            # Try to load and validate image
            image = self._load_image_safely(image_path)
            if image is None:
                issues.append("Cannot load image - corrupted or invalid format")
                result.is_valid = False
                return result
            
            # Check image dimensions
            height, width = image.shape[:2]
            result.image_dimensions = (width, height)
            
            if width < self.config.min_image_size[0] or height < self.config.min_image_size[1]:
                issues.append(f"Image too small: {width}x{height} < {self.config.min_image_size}")
                result.is_valid = False
            
            if width > self.config.max_image_size[0] or height > self.config.max_image_size[1]:
                warnings.append(f"Image very large: {width}x{height} > {self.config.max_image_size}")
            
            # Check image quality metrics
            quality_checks = self._check_image_quality(image)
            result.brightness_score = quality_checks['brightness']
            result.contrast_score = quality_checks['contrast']
            result.blur_score = quality_checks['blur']
            
            # Validate brightness
            if quality_checks['brightness'] < self.config.min_brightness:
                issues.append(f"Image too dark: brightness {quality_checks['brightness']:.3f}")
                result.is_valid = False
            elif quality_checks['brightness'] > self.config.max_brightness:
                issues.append(f"Image too bright: brightness {quality_checks['brightness']:.3f}")
                result.is_valid = False
            
            # Validate contrast
            if quality_checks['contrast'] < self.config.min_contrast:
                warnings.append(f"Low contrast: {quality_checks['contrast']:.3f}")
            
            # Validate blur
            if quality_checks['blur'] < self.config.blur_threshold:
                warnings.append(f"Potentially blurry image: blur score {quality_checks['blur']:.1f}")
            
            # Check for medical image characteristics
            medical_checks = self._check_medical_image_characteristics(image)
            if medical_checks['warnings']:
                warnings.extend(medical_checks['warnings'])
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            issues.append(f"Validation error: {str(e)}")
            result.is_valid = False
        
        # Final validation status
        result.is_valid = len(issues) == 0
        
        return result
    
    def validate_dataset(self, dataset_path: Union[str, Path]) -> DatasetValidationSummary:
        """
        Validate an entire dataset
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            DatasetValidationSummary with comprehensive validation results
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Find all image files
        image_files = []
        for ext in self.config.allowed_formats:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in dataset")
        
        # Validate each image
        validation_results = []
        for image_file in image_files:
            result = self.validate_image(image_file)
            validation_results.append(result)
        
        # Calculate summary statistics
        valid_files = sum(1 for r in validation_results if r.is_valid)
        invalid_files = len(validation_results) - valid_files
        
        # Count common issues
        common_issues = {}
        for result in validation_results:
            for issue in result.issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        # Calculate dataset statistics
        dataset_stats = self._calculate_dataset_statistics(validation_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, common_issues)
        
        return DatasetValidationSummary(
            total_files=len(validation_results),
            valid_files=valid_files,
            invalid_files=invalid_files,
            validation_results=validation_results,
            common_issues=common_issues,
            dataset_statistics=dataset_stats,
            recommendations=recommendations
        )
    
    def validate_directory_structure(self, dataset_path: Union[str, Path]) -> Dict:
        """
        Validate dataset directory structure for ML training
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with structure validation results
        """
        dataset_path = Path(dataset_path)
        
        structure_info = {
            "has_train_dir": False,
            "has_val_dir": False,
            "has_test_dir": False,
            "class_directories": [],
            "structure_type": "unknown",
            "recommendations": []
        }
        
        # Check for common ML directory structures
        subdirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        
        # Check for train/val/test structure
        if "train" in subdirs:
            structure_info["has_train_dir"] = True
        if "val" in subdirs or "validation" in subdirs:
            structure_info["has_val_dir"] = True
        if "test" in subdirs:
            structure_info["has_test_dir"] = True
        
        # Check for class-based structure
        potential_classes = ["NORMAL", "PNEUMONIA", "normal", "pneumonia"]
        for subdir in subdirs:
            if subdir in potential_classes or any(cls in subdir.lower() for cls in ["normal", "pneumonia"]):
                structure_info["class_directories"].append(subdir)
        
        # Determine structure type
        if structure_info["has_train_dir"]:
            structure_info["structure_type"] = "train_val_test"
        elif structure_info["class_directories"]:
            structure_info["structure_type"] = "class_based"
        else:
            structure_info["structure_type"] = "flat"
        
        # Generate recommendations
        if not structure_info["has_train_dir"] and not structure_info["class_directories"]:
            structure_info["recommendations"].append("Consider organizing images into class directories (NORMAL/PNEUMONIA)")
        
        if structure_info["has_train_dir"] and not structure_info["has_val_dir"]:
            structure_info["recommendations"].append("Consider adding validation directory for model evaluation")
        
        return structure_info
    
    def _load_image_safely(self, image_path: Path) -> Optional[np.ndarray]:
        """Safely load an image file"""
        try:
            # Try OpenCV first
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image
            
            # Try PIL as fallback
            with Image.open(image_path) as pil_image:
                # Convert to grayscale if needed
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                return np.array(pil_image)
                
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _check_image_quality(self, image: np.ndarray) -> Dict:
        """Check various image quality metrics"""
        # Brightness (mean pixel value)
        brightness = np.mean(image) / 255.0
        
        # Contrast (standard deviation of pixel values)
        contrast = np.std(image) / 255.0
        
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'blur': blur_score
        }
    
    def _check_medical_image_characteristics(self, image: np.ndarray) -> Dict:
        """Check for characteristics typical of medical images"""
        warnings = []
        
        # Check for very uniform regions (might indicate poor quality)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        max_bin_count = np.max(hist)
        total_pixels = image.shape[0] * image.shape[1]
        
        if max_bin_count / total_pixels > 0.5:
            warnings.append("Image has very uniform pixel distribution")
        
        # Check aspect ratio (chest X-rays typically have certain proportions)
        height, width = image.shape
        aspect_ratio = width / height
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        return {'warnings': warnings}
    
    def _calculate_dataset_statistics(self, validation_results: List[ValidationResult]) -> Dict:
        """Calculate comprehensive dataset statistics"""
        valid_results = [r for r in validation_results if r.is_valid]
        
        if not valid_results:
            return {"error": "No valid images found"}
        
        # Dimension statistics
        dimensions = [r.image_dimensions for r in valid_results if r.image_dimensions]
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        
        # File size statistics
        file_sizes = [r.file_size_bytes for r in valid_results]
        
        # Quality metrics
        brightness_scores = [r.brightness_score for r in valid_results if r.brightness_score is not None]
        contrast_scores = [r.contrast_score for r in valid_results if r.contrast_score is not None]
        blur_scores = [r.blur_score for r in valid_results if r.blur_score is not None]
        
        return {
            "dimensions": {
                "width_mean": np.mean(widths) if widths else 0,
                "width_std": np.std(widths) if widths else 0,
                "height_mean": np.mean(heights) if heights else 0,
                "height_std": np.std(heights) if heights else 0,
                "min_width": min(widths) if widths else 0,
                "max_width": max(widths) if widths else 0,
                "min_height": min(heights) if heights else 0,
                "max_height": max(heights) if heights else 0
            },
            "file_sizes": {
                "mean_bytes": np.mean(file_sizes) if file_sizes else 0,
                "median_bytes": np.median(file_sizes) if file_sizes else 0,
                "min_bytes": min(file_sizes) if file_sizes else 0,
                "max_bytes": max(file_sizes) if file_sizes else 0
            },
            "quality": {
                "brightness_mean": np.mean(brightness_scores) if brightness_scores else 0,
                "contrast_mean": np.mean(contrast_scores) if contrast_scores else 0,
                "blur_mean": np.mean(blur_scores) if blur_scores else 0
            }
        }
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                common_issues: Dict[str, int]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        total_files = len(validation_results)
        invalid_ratio = sum(1 for r in validation_results if not r.is_valid) / total_files
        
        if invalid_ratio > 0.1:
            recommendations.append(f"High invalid file ratio ({invalid_ratio:.1%}). Consider data cleaning.")
        
        # Check for common issues
        for issue, count in common_issues.items():
            if count / total_files > 0.05:  # More than 5% of files
                if "too small" in issue:
                    recommendations.append("Many images are too small. Consider upsampling or adjusting minimum size.")
                elif "too large" in issue:
                    recommendations.append("Many images are too large. Consider downsampling for efficiency.")
                elif "dark" in issue or "bright" in issue:
                    recommendations.append("Brightness issues detected. Consider histogram equalization.")
        
        return recommendations