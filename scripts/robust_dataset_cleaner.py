#!/usr/bin/env python3
"""
Robust Medical Dataset Cleaner - Simplified and More Reliable
Focuses on core cleaning tasks with better error handling
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import shutil
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import hashlib
from PIL import Image
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustDatasetCleaner:
    """Robust medical dataset cleaner with simplified operations"""
    
    def __init__(self, source_path: str, target_path: str = "data/chest_xray_final"):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.target_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'original_counts': {},
            'processed_counts': {},
            'corrupted_removed': 0,
            'duplicates_removed': 0,
            'blurry_removed': 0,
            'standardized': 0,
            'augmented': 0,
            'total_processed': 0
        }
        
        # Quality thresholds
        self.blur_threshold = 100
        self.target_size = (224, 224)
    
    def is_image_valid(self, img_path: Path) -> bool:
        """Check if image is valid and readable"""
        try:
            # Check file size
            if img_path.stat().st_size < 1000:  # Less than 1KB
                return False
            
            # Try to load with OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Check dimensions
            if len(img.shape) < 2 or img.shape[0] < 50 or img.shape[1] < 50:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating {img_path}: {e}")
            return False
    
    def calculate_simple_hash(self, img_path: Path) -> str:
        """Calculate simple hash for duplicate detection"""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize to small size for hashing
            small = cv2.resize(img, (8, 8))
            
            # Calculate hash
            hash_str = hashlib.md5(small.tobytes()).hexdigest()
            return hash_str
            
        except Exception:
            return None
    
    def assess_blur(self, img_path: Path) -> float:
        """Assess image blur using Laplacian variance"""
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0
            
            return cv2.Laplacian(img, cv2.CV_64F).var()
            
        except Exception:
            return 0
    
    def standardize_image(self, img_path: Path, output_path: Path) -> bool:
        """Standardize image to target format and size"""
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JPEG
            img_pil = Image.fromarray(img_resized)
            img_pil.save(output_path.with_suffix('.jpeg'), 'JPEG', quality=95)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error standardizing {img_path}: {e}")
            return False
    
    def create_augmented_images(self, img_path: Path, output_dir: Path, count: int) -> int:
        """Create augmented versions of an image"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return 0
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented_count = 0
            
            # Simple augmentations
            augmentations = [
                ('flip_h', lambda x: cv2.flip(x, 1)),
                ('flip_v', lambda x: cv2.flip(x, 0)),
                ('bright_up', lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=10)),
                ('bright_down', lambda x: cv2.convertScaleAbs(x, alpha=0.8, beta=-10)),
            ]
            
            for i, (name, func) in enumerate(augmentations):
                if augmented_count >= count:
                    break
                
                try:
                    aug_img = func(img_rgb.copy())
                    aug_img_resized = cv2.resize(aug_img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
                    
                    # Save augmented image
                    aug_filename = f"{img_path.stem}_{name}.jpeg"
                    aug_path = output_dir / aug_filename
                    
                    img_pil = Image.fromarray(aug_img_resized)
                    img_pil.save(aug_path, 'JPEG', quality=95)
                    
                    augmented_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error creating augmentation {name}: {e}")
            
            return augmented_count
            
        except Exception as e:
            logger.warning(f"Error augmenting {img_path}: {e}")
            return 0
    
    def process_split(self, split_name: str) -> Dict:
        """Process a complete split"""
        logger.info(f"\\n🔄 Processing {split_name} split...")
        
        split_source = self.source_path / split_name
        split_target = self.target_path / split_name
        
        if not split_source.exists():
            logger.warning(f"Split {split_name} not found")
            return {}
        
        split_stats = {'original': {}, 'processed': {}, 'removed': {}}
        
        # Process each class
        for class_dir in split_source.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            logger.info(f"  Processing {class_name}...")
            
            # Get all image files
            image_files = [f for f in class_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            split_stats['original'][class_name] = len(image_files)
            
            logger.info(f"    Found {len(image_files)} images")
            
            # Step 1: Validate images
            valid_images = []
            for img_path in image_files:
                if self.is_image_valid(img_path):
                    valid_images.append(img_path)
                else:
                    self.stats['corrupted_removed'] += 1
            
            logger.info(f"    Valid images: {len(valid_images)}")
            
            # Step 2: Remove duplicates using simple hashing
            unique_images = []
            seen_hashes = set()
            
            for img_path in valid_images:
                img_hash = self.calculate_simple_hash(img_path)
                if img_hash and img_hash not in seen_hashes:
                    unique_images.append(img_path)
                    seen_hashes.add(img_hash)
                else:
                    self.stats['duplicates_removed'] += 1
            
            logger.info(f"    Unique images: {len(unique_images)}")
            
            # Step 3: Remove very blurry images
            sharp_images = []
            for img_path in unique_images:
                blur_score = self.assess_blur(img_path)
                if blur_score >= self.blur_threshold:
                    sharp_images.append(img_path)
                else:
                    self.stats['blurry_removed'] += 1
            
            logger.info(f"    Sharp images: {len(sharp_images)}")
            
            # Step 4: Standardize and save images
            processed_count = 0
            target_class_dir = split_target / class_name
            
            for img_path in sharp_images:
                output_path = target_class_dir / f"{img_path.stem}.jpeg"
                if self.standardize_image(img_path, output_path):
                    processed_count += 1
                    self.stats['standardized'] += 1
            
            split_stats['processed'][class_name] = processed_count
            logger.info(f"    Processed: {processed_count} images")
            
            # Step 5: Augment NORMAL class for training split
            if split_name == 'train' and class_name == 'NORMAL':
                # Calculate how many augmented images we need
                pneumonia_count = split_stats['processed'].get('PNEUMONIA', 0)
                normal_count = processed_count
                
                if pneumonia_count > normal_count:
                    needed_augmentations = min(pneumonia_count - normal_count, normal_count * 2)  # Don't over-augment
                    
                    if needed_augmentations > 0:
                        logger.info(f"    Creating {needed_augmentations} augmented NORMAL images...")
                        
                        aug_dir = split_target / "NORMAL_augmented"
                        aug_dir.mkdir(parents=True, exist_ok=True)
                        
                        augmented_total = 0
                        augs_per_image = max(1, needed_augmentations // len(sharp_images))
                        
                        for img_path in sharp_images:
                            if augmented_total >= needed_augmentations:
                                break
                            
                            aug_count = self.create_augmented_images(img_path, aug_dir, augs_per_image)
                            augmented_total += aug_count
                            self.stats['augmented'] += aug_count
                        
                        split_stats['processed']['NORMAL_augmented'] = augmented_total
                        logger.info(f"    Created {augmented_total} augmented images")
        
        return split_stats
    
    def create_summary_report(self) -> Dict:
        """Create final summary report"""
        # Count final images
        final_counts = {}
        total_final = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = self.target_path / split
            if split_dir.exists():
                final_counts[split] = {}
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        count = len([f for f in class_dir.glob('*.jpeg')])
                        final_counts[split][class_dir.name] = count
                        total_final += count
        
        report = {
            'processing_summary': {
                'source_path': str(self.source_path),
                'target_path': str(self.target_path),
                'processing_timestamp': datetime.now().isoformat(),
                'total_images_processed': self.stats['standardized'] + self.stats['augmented']
            },
            'cleaning_statistics': {
                'corrupted_removed': self.stats['corrupted_removed'],
                'duplicates_removed': self.stats['duplicates_removed'],
                'blurry_removed': self.stats['blurry_removed'],
                'images_standardized': self.stats['standardized'],
                'images_augmented': self.stats['augmented']
            },
            'final_dataset_structure': final_counts,
            'dataset_ready_for_training': True
        }
        
        return report
    
    def clean_dataset(self) -> Dict:
        """Main cleaning method"""
        logger.info("🚀 Starting robust dataset cleaning...")
        start_time = datetime.now()
        
        # Process each split
        for split in ['train', 'val', 'test']:
            self.process_split(split)
        
        # Generate report
        report = self.create_summary_report()
        
        # Save report
        with open('robust_cleaning_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Cleaning completed in {processing_time:.2f} seconds")
        
        return report


def main():
    """Main function"""
    print("🏥 ROBUST MEDICAL DATASET CLEANER")
    print("=" * 50)
    
    # Use the copied dataset as source
    source_dataset = "data/chest_xray_cleaned"
    final_dataset = "data/chest_xray_final"
    
    # Initialize cleaner
    cleaner = RobustDatasetCleaner(source_dataset, final_dataset)
    
    # Run cleaning
    report = cleaner.clean_dataset()
    
    # Print summary
    print("\\n" + "=" * 50)
    print("🎉 CLEANING COMPLETED!")
    print("=" * 50)
    
    stats = report['cleaning_statistics']
    print(f"📊 CLEANING SUMMARY:")
    print(f"   Corrupted removed: {stats['corrupted_removed']}")
    print(f"   Duplicates removed: {stats['duplicates_removed']}")
    print(f"   Blurry images removed: {stats['blurry_removed']}")
    print(f"   Images standardized: {stats['images_standardized']}")
    print(f"   Images augmented: {stats['images_augmented']}")
    
    print(f"\\n📁 FINAL DATASET:")
    for split, classes in report['final_dataset_structure'].items():
        print(f"   {split.upper()}:")
        for class_name, count in classes.items():
            print(f"     {class_name}: {count:,} images")
    
    print(f"\\n🎯 DATASET READY FOR TRAINING!")
    
    return report

if __name__ == "__main__":
    report = main()