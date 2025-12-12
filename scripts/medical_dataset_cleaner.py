#!/usr/bin/env python3
"""
Comprehensive Medical Dataset Cleaner for Chest X-Ray Pneumonia Dataset
Implements all cleaning, validation, and preprocessing steps for medical imaging
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import shutil
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import hashlib
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
from skimage.metrics import structural_similarity as ssim
import imagehash
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDatasetCleaner:
    """Comprehensive medical dataset cleaner with quality assessment and standardization"""
    
    def __init__(self, source_path: str, target_path: str = "data/chest_xray_processed"):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.target_path.mkdir(parents=True, exist_ok=True)
        
        # Cleaning statistics
        self.stats = {
            'original_counts': {},
            'corrupted_removed': 0,
            'duplicates_removed': 0,
            'blurry_removed': 0,
            'exposure_issues_fixed': 0,
            'exposure_issues_removed': 0,
            'contrast_enhanced': 0,
            'images_processed': 0,
            'augmented_images': 0,
            'final_counts': {},
            'quality_metrics': defaultdict(list),
            'processing_time': 0
        }
        
        # Quality thresholds
        self.blur_threshold = 100  # Variance of Laplacian threshold
        self.brightness_low = 0.1  # Too dark threshold
        self.brightness_high = 0.9  # Too bright threshold
        self.contrast_threshold = 0.15  # Low contrast threshold
        
        # Target image specifications
        self.target_size = (224, 224)
        self.target_format = 'JPEG'
        
    def validate_images(self, split_path: Path) -> List[Path]:
        """Step 1: Validate images and remove corrupted ones"""
        logger.info(f"🧹 Validating images in {split_path.name}...")
        
        valid_images = []
        corrupted_count = 0
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            logger.info(f"  Validating {class_name} images...")
            
            for img_path in class_dir.glob('*'):
                if not img_path.is_file():
                    continue
                    
                try:
                    # Check file size
                    if img_path.stat().st_size == 0:
                        logger.warning(f"Zero-byte file: {img_path}")
                        corrupted_count += 1
                        continue
                    
                    # Try to load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Unreadable image: {img_path}")
                        corrupted_count += 1
                        continue
                    
                    # Basic shape validation
                    if len(img.shape) < 2 or img.shape[0] < 32 or img.shape[1] < 32:
                        logger.warning(f"Invalid dimensions: {img_path}")
                        corrupted_count += 1
                        continue
                    
                    valid_images.append(img_path)
                    
                except Exception as e:
                    logger.warning(f"Error validating {img_path}: {e}")
                    corrupted_count += 1
        
        self.stats['corrupted_removed'] += corrupted_count
        logger.info(f"  ✅ Validated: {len(valid_images)} valid, {corrupted_count} corrupted removed")
        
        return valid_images
    
    def calculate_image_hash(self, img_path: Path) -> str:
        """Calculate perceptual hash for duplicate detection"""
        try:
            with Image.open(img_path) as img:
                # Convert to grayscale for consistent hashing
                if img.mode != 'L':
                    img = img.convert('L')
                # Calculate perceptual hash
                hash_value = imagehash.phash(img, hash_size=16)
                return str(hash_value)
        except Exception as e:
            logger.warning(f"Error calculating hash for {img_path}: {e}")
            return None
    
    def detect_duplicates(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """Step 2: Detect duplicate and near-duplicate images"""
        logger.info("🔁 Detecting duplicates using perceptual hashing...")
        
        hash_groups = defaultdict(list)
        
        for img_path in image_paths:
            img_hash = self.calculate_image_hash(img_path)
            if img_hash:
                hash_groups[img_hash].append(img_path)
        
        # Find groups with duplicates
        duplicates = {hash_val: paths for hash_val, paths in hash_groups.items() if len(paths) > 1}
        
        logger.info(f"  Found {len(duplicates)} duplicate groups affecting {sum(len(paths) for paths in duplicates.values())} images")
        
        return duplicates
    
    def remove_duplicates_balanced(self, duplicates: Dict[str, List[Path]]) -> List[Path]:
        """Remove duplicates while maintaining class balance"""
        logger.info("  Removing duplicates while preserving class balance...")
        
        images_to_keep = []
        removed_count = 0
        
        for hash_val, duplicate_paths in duplicates.items():
            # Group by class
            class_groups = defaultdict(list)
            for path in duplicate_paths:
                class_name = path.parent.name
                class_groups[class_name].append(path)
            
            # Keep one from each class if possible, otherwise keep just one
            for class_name, class_paths in class_groups.items():
                images_to_keep.append(class_paths[0])  # Keep first one
                removed_count += len(class_paths) - 1
        
        self.stats['duplicates_removed'] += removed_count
        logger.info(f"  ✅ Removed {removed_count} duplicate images")
        
        return images_to_keep
    
    def assess_image_quality(self, img_path: Path) -> Dict:
        """Step 3: Assess image quality (blur, exposure)"""
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return {'valid': False, 'reason': 'unreadable'}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Calculate quality metrics
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            
            # Store metrics for reporting
            self.stats['quality_metrics']['blur_scores'].append(blur_score)
            self.stats['quality_metrics']['brightness'].append(brightness)
            self.stats['quality_metrics']['contrast'].append(contrast)
            
            # Quality assessment
            quality_issues = []
            
            if blur_score < self.blur_threshold:
                quality_issues.append('blurry')
            
            if brightness < self.brightness_low:
                quality_issues.append('underexposed')
            elif brightness > self.brightness_high:
                quality_issues.append('overexposed')
            
            if contrast < self.contrast_threshold:
                quality_issues.append('low_contrast')
            
            return {
                'valid': True,
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'issues': quality_issues,
                'path': img_path
            }
            
        except Exception as e:
            logger.warning(f"Error assessing quality for {img_path}: {e}")
            return {'valid': False, 'reason': 'error'}
    
    def fix_exposure_issues(self, img: np.ndarray, brightness: float) -> Tuple[np.ndarray, bool]:
        """Fix exposure issues automatically if recoverable"""
        try:
            if brightness < self.brightness_low:
                # Too dark - apply gamma correction
                gamma = 1.5
                corrected = np.power(img / 255.0, 1/gamma) * 255.0
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                return corrected, True
            
            elif brightness > self.brightness_high:
                # Too bright - apply gamma correction
                gamma = 0.7
                corrected = np.power(img / 255.0, 1/gamma) * 255.0
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                return corrected, True
            
            return img, False
            
        except Exception as e:
            logger.warning(f"Error fixing exposure: {e}")
            return img, False
    
    def apply_clahe_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Step 5: Apply CLAHE for contrast enhancement"""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Convert back to 3-channel if original was 3-channel
            if len(img.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error applying CLAHE: {e}")
            return img
    
    def standardize_image(self, img_path: Path, target_path: Path, quality_info: Dict) -> bool:
        """Step 4: Standardize image format, size, and normalization"""
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Convert to RGB (3-channel)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Fix exposure issues if needed
            if 'underexposed' in quality_info['issues'] or 'overexposed' in quality_info['issues']:
                img, fixed = self.fix_exposure_issues(img, quality_info['brightness'])
                if fixed:
                    self.stats['exposure_issues_fixed'] += 1
            
            # Apply CLAHE if low contrast
            if 'low_contrast' in quality_info['issues']:
                img = self.apply_clahe_enhancement(img)
                self.stats['contrast_enhanced'] += 1
            
            # Resize while maintaining aspect ratio
            h, w = img.shape[:2]
            aspect_ratio = w / h
            
            if aspect_ratio > 1:  # Width > Height
                new_w = self.target_size[0]
                new_h = int(new_w / aspect_ratio)
            else:  # Height >= Width
                new_h = self.target_size[1]
                new_w = int(new_h * aspect_ratio)
            
            # Resize
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size
            pad_h = (self.target_size[1] - new_h) // 2
            pad_w = (self.target_size[0] - new_w) // 2
            
            img_padded = np.pad(img_resized, 
                              ((pad_h, self.target_size[1] - new_h - pad_h),
                               (pad_w, self.target_size[0] - new_w - pad_w),
                               (0, 0)), 
                              mode='constant', constant_values=0)
            
            # Normalize to 0-1 range (will be saved as 0-255 JPEG)
            img_normalized = img_padded.astype(np.float32) / 255.0
            img_final = (img_normalized * 255).astype(np.uint8)
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JPEG
            target_path = target_path.with_suffix('.jpeg')
            img_pil = Image.fromarray(img_final)
            img_pil.save(target_path, 'JPEG', quality=95)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error standardizing {img_path}: {e}")
            return False
    
    def augment_normal_class(self, normal_images: List[Path], target_count: int, output_dir: Path) -> int:
        """Step 6: Augment NORMAL class to balance dataset"""
        logger.info(f"⚡ Augmenting NORMAL class from {len(normal_images)} to ~{target_count} images...")
        
        # Create augmented directory
        aug_dir = output_dir / "NORMAL_augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)
        
        augmented_count = 0
        images_needed = target_count - len(normal_images)
        
        if images_needed <= 0:
            logger.info("  No augmentation needed - NORMAL class already sufficient")
            return 0
        
        # Augmentation parameters
        augmentations = [
            ('flip_h', lambda img: cv2.flip(img, 1)),
            ('flip_v', lambda img: cv2.flip(img, 0)),
            ('rotate_10', lambda img: self._rotate_image(img, 10)),
            ('rotate_-10', lambda img: self._rotate_image(img, -10)),
            ('zoom_in', lambda img: self._zoom_image(img, 1.1)),
            ('zoom_out', lambda img: self._zoom_image(img, 0.9)),
            ('bright_up', lambda img: self._adjust_brightness(img, 1.15)),
            ('bright_down', lambda img: self._adjust_brightness(img, 0.85)),
        ]
        
        # Generate augmented images
        aug_per_original = max(1, images_needed // len(normal_images))
        
        for i, img_path in enumerate(normal_images):
            if augmented_count >= images_needed:
                break
                
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Apply multiple augmentations to this image
                for j, (aug_name, aug_func) in enumerate(augmentations[:aug_per_original]):
                    if augmented_count >= images_needed:
                        break
                    
                    try:
                        aug_img = aug_func(img.copy())
                        
                        # Save augmented image
                        aug_filename = f"{img_path.stem}_{aug_name}.jpeg"
                        aug_path = aug_dir / aug_filename
                        
                        # Standardize augmented image
                        aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                        aug_img_pil = Image.fromarray(aug_img_rgb)
                        aug_img_resized = aug_img_pil.resize(self.target_size, Image.Resampling.LANCZOS)
                        aug_img_resized.save(aug_path, 'JPEG', quality=95)
                        
                        augmented_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error augmenting {img_path} with {aug_name}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error processing {img_path} for augmentation: {e}")
        
        logger.info(f"  ✅ Generated {augmented_count} augmented NORMAL images")
        return augmented_count
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _zoom_image(self, img: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom image by given factor"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        if zoom_factor > 1:  # Zoom in
            resized = cv2.resize(img, (new_w, new_h))
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h + h, start_w:start_w + w]
        else:  # Zoom out
            resized = cv2.resize(img, (new_w, new_h))
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return np.pad(resized, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w), (0, 0)), mode='reflect')
    
    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness"""
        adjusted = img.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def process_split(self, split_name: str) -> Dict:
        """Process a complete split (train/val/test)"""
        logger.info(f"\n🔄 Processing {split_name} split...")
        
        split_source = self.source_path / split_name
        split_target = self.target_path / split_name
        
        if not split_source.exists():
            logger.warning(f"Split {split_name} not found")
            return {}
        
        split_stats = {'original': {}, 'final': {}, 'removed': {}, 'augmented': 0}
        
        # Step 1: Validate all images
        valid_images = self.validate_images(split_source)
        
        # Count original images by class
        for img_path in valid_images:
            class_name = img_path.parent.name
            split_stats['original'][class_name] = split_stats['original'].get(class_name, 0) + 1
        
        # Step 2: Detect and remove duplicates
        duplicates = self.detect_duplicates(valid_images)
        if duplicates:
            valid_images = self.remove_duplicates_balanced(duplicates)
        
        # Step 3 & 4: Quality assessment and standardization
        processed_images = {'NORMAL': [], 'PNEUMONIA': []}
        
        for img_path in valid_images:
            class_name = img_path.parent.name
            
            # Assess quality
            quality_info = self.assess_image_quality(img_path)
            
            if not quality_info['valid']:
                continue
            
            # Check if image should be removed due to quality issues
            if 'blurry' in quality_info['issues'] and quality_info['blur_score'] < self.blur_threshold:
                self.stats['blurry_removed'] += 1
                split_stats['removed'][f'{class_name}_blurry'] = split_stats['removed'].get(f'{class_name}_blurry', 0) + 1
                continue
            
            # Check for severe exposure issues that can't be fixed
            if quality_info['brightness'] < 0.05 or quality_info['brightness'] > 0.95:
                self.stats['exposure_issues_removed'] += 1
                split_stats['removed'][f'{class_name}_exposure'] = split_stats['removed'].get(f'{class_name}_exposure', 0) + 1
                continue
            
            # Standardize and save image
            target_path = split_target / class_name / f"{img_path.stem}.jpeg"
            
            if self.standardize_image(img_path, target_path, quality_info):
                processed_images[class_name].append(target_path)
                self.stats['images_processed'] += 1
        
        # Step 6: Augment NORMAL class for training split only
        if split_name == 'train':
            normal_count = len(processed_images['NORMAL'])
            pneumonia_count = len(processed_images['PNEUMONIA'])
            
            if normal_count < pneumonia_count:
                augmented = self.augment_normal_class(
                    [p for p in valid_images if p.parent.name == 'NORMAL'],
                    pneumonia_count,
                    split_target
                )
                split_stats['augmented'] = augmented
                self.stats['augmented_images'] += augmented
        
        # Count final images
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = split_target / class_name
            if class_dir.exists():
                count = len([f for f in class_dir.glob('*.jpeg')])
                split_stats['final'][class_name] = count
        
        # Count augmented images separately
        aug_dir = split_target / "NORMAL_augmented"
        if aug_dir.exists():
            aug_count = len([f for f in aug_dir.glob('*.jpeg')])
            split_stats['final']['NORMAL_augmented'] = aug_count
        
        return split_stats
    
    def generate_quality_report(self) -> Dict:
        """Step 7: Generate comprehensive quality report"""
        logger.info("📊 Generating quality report...")
        
        # Calculate final counts
        final_counts = {}
        for split in ['train', 'val', 'test']:
            split_dir = self.target_path / split
            if split_dir.exists():
                final_counts[split] = {}
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        count = len([f for f in class_dir.glob('*.jpeg')])
                        final_counts[split][class_dir.name] = count
        
        # Quality statistics
        quality_stats = {}
        if self.stats['quality_metrics']['blur_scores']:
            quality_stats = {
                'average_blur_score': np.mean(self.stats['quality_metrics']['blur_scores']),
                'average_brightness': np.mean(self.stats['quality_metrics']['brightness']),
                'average_contrast': np.mean(self.stats['quality_metrics']['contrast']),
                'blur_score_std': np.std(self.stats['quality_metrics']['blur_scores']),
                'brightness_std': np.std(self.stats['quality_metrics']['brightness']),
                'contrast_std': np.std(self.stats['quality_metrics']['contrast'])
            }
        
        report = {
            'processing_summary': {
                'source_path': str(self.source_path),
                'target_path': str(self.target_path),
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': self.stats['processing_time']
            },
            'cleaning_statistics': {
                'images_processed': self.stats['images_processed'],
                'corrupted_removed': self.stats['corrupted_removed'],
                'duplicates_removed': self.stats['duplicates_removed'],
                'blurry_removed': self.stats['blurry_removed'],
                'exposure_issues_fixed': self.stats['exposure_issues_fixed'],
                'exposure_issues_removed': self.stats['exposure_issues_removed'],
                'contrast_enhanced': self.stats['contrast_enhanced'],
                'augmented_images': self.stats['augmented_images']
            },
            'final_dataset_structure': final_counts,
            'quality_statistics': quality_stats,
            'thresholds_used': {
                'blur_threshold': self.blur_threshold,
                'brightness_low': self.brightness_low,
                'brightness_high': self.brightness_high,
                'contrast_threshold': self.contrast_threshold
            }
        }
        
        return report
    
    def create_visualizations(self, report: Dict):
        """Create visualization plots for the report"""
        logger.info("📈 Creating visualization plots...")
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Dataset Cleaning Report - Chest X-Ray Pneumonia', fontsize=16, fontweight='bold')
            
            # Plot 1: Class distribution before/after
            if 'final_dataset_structure' in report:
                train_data = report['final_dataset_structure'].get('train', {})
                
                classes = []
                counts = []
                colors = []
                
                for class_name, count in train_data.items():
                    classes.append(class_name.replace('_', '\n'))
                    counts.append(count)
                    if 'NORMAL' in class_name:
                        colors.append('#2E86AB')
                    else:
                        colors.append('#A23B72')
                
                axes[0, 0].bar(classes, counts, color=colors, alpha=0.8)
                axes[0, 0].set_title('Final Class Distribution (Training Set)', fontweight='bold')
                axes[0, 0].set_ylabel('Number of Images')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add count labels on bars
                for i, count in enumerate(counts):
                    axes[0, 0].text(i, count + max(counts) * 0.01, str(count), 
                                   ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Quality metrics distribution
            if self.stats['quality_metrics']['brightness']:
                axes[0, 1].hist(self.stats['quality_metrics']['brightness'], bins=30, 
                               alpha=0.7, color='#F18F01', edgecolor='black')
                axes[0, 1].axvline(self.brightness_low, color='red', linestyle='--', 
                                  label=f'Low threshold ({self.brightness_low})')
                axes[0, 1].axvline(self.brightness_high, color='red', linestyle='--', 
                                  label=f'High threshold ({self.brightness_high})')
                axes[0, 1].set_title('Brightness Distribution', fontweight='bold')
                axes[0, 1].set_xlabel('Brightness (0-1)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
            
            # Plot 3: Blur scores distribution
            if self.stats['quality_metrics']['blur_scores']:
                axes[1, 0].hist(self.stats['quality_metrics']['blur_scores'], bins=30, 
                               alpha=0.7, color='#C73E1D', edgecolor='black')
                axes[1, 0].axvline(self.blur_threshold, color='red', linestyle='--', 
                                  label=f'Blur threshold ({self.blur_threshold})')
                axes[1, 0].set_title('Blur Score Distribution', fontweight='bold')
                axes[1, 0].set_xlabel('Blur Score (Variance of Laplacian)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
            
            # Plot 4: Cleaning summary
            cleaning_stats = report['cleaning_statistics']
            categories = ['Processed', 'Corrupted\nRemoved', 'Duplicates\nRemoved', 
                         'Blurry\nRemoved', 'Exposure\nFixed', 'Contrast\nEnhanced', 'Augmented']
            values = [cleaning_stats['images_processed'], cleaning_stats['corrupted_removed'],
                     cleaning_stats['duplicates_removed'], cleaning_stats['blurry_removed'],
                     cleaning_stats['exposure_issues_fixed'], cleaning_stats['contrast_enhanced'],
                     cleaning_stats['augmented_images']]
            
            colors_bar = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#FF9800', '#9C27B0']
            
            bars = axes[1, 1].bar(categories, values, color=colors_bar, alpha=0.8)
            axes[1, 1].set_title('Cleaning Operations Summary', fontweight='bold')
            axes[1, 1].set_ylabel('Number of Images')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                                   str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('dataset_cleaning_report.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("  ✅ Visualization saved as 'dataset_cleaning_report.png'")
            
        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")
    
    def clean_dataset(self) -> Dict:
        """Main method to clean the entire dataset"""
        logger.info("🚀 Starting comprehensive medical dataset cleaning...")
        start_time = datetime.now()
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_stats = self.process_split(split)
            self.stats[f'{split}_stats'] = split_stats
        
        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Generate comprehensive report
        report = self.generate_quality_report()
        
        # Create visualizations
        self.create_visualizations(report)
        
        # Save report
        with open('medical_dataset_cleaning_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Dataset cleaning completed in {self.stats['processing_time']:.2f} seconds")
        logger.info(f"📄 Report saved as 'medical_dataset_cleaning_report.json'")
        
        return report


def main():
    """Main function to run the medical dataset cleaner"""
    print("🏥 MEDICAL DATASET CLEANER - CHEST X-RAY PNEUMONIA")
    print("=" * 60)
    print("Performing comprehensive cleaning, validation, and preprocessing...")
    
    # Configuration
    source_dataset = "data/chest_xray_cleaned"
    processed_dataset = "data/chest_xray_processed"
    
    # Initialize cleaner
    cleaner = MedicalDatasetCleaner(source_dataset, processed_dataset)
    
    # Run cleaning process
    report = cleaner.clean_dataset()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 CLEANING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    cleaning_stats = report['cleaning_statistics']
    print(f"📊 PROCESSING SUMMARY:")
    print(f"   Images processed: {cleaning_stats['images_processed']:,}")
    print(f"   Corrupted removed: {cleaning_stats['corrupted_removed']:,}")
    print(f"   Duplicates removed: {cleaning_stats['duplicates_removed']:,}")
    print(f"   Blurry images removed: {cleaning_stats['blurry_removed']:,}")
    print(f"   Exposure issues fixed: {cleaning_stats['exposure_issues_fixed']:,}")
    print(f"   Contrast enhanced: {cleaning_stats['contrast_enhanced']:,}")
    print(f"   Images augmented: {cleaning_stats['augmented_images']:,}")
    
    print(f"\n📁 FINAL DATASET STRUCTURE:")
    for split, classes in report['final_dataset_structure'].items():
        print(f"   {split.upper()}:")
        for class_name, count in classes.items():
            print(f"     {class_name}: {count:,} images")
    
    print(f"\n🎯 READY FOR TRAINING!")
    print(f"   ✅ All images standardized to 224x224 RGB")
    print(f"   ✅ Quality validated and enhanced")
    print(f"   ✅ Class balance improved with augmentation")
    print(f"   ✅ Medical imaging best practices applied")
    
    return report

if __name__ == "__main__":
    report = main()