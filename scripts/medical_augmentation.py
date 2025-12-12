#!/usr/bin/env python3
"""
Medical Image Augmentation for Class Balance
Specifically designed for chest X-ray images with medical-appropriate transformations
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import List, Tuple
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalImageAugmenter:
    """Medical-grade image augmentation for chest X-rays"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.normal_path = self.train_path / "NORMAL"
        self.pneumonia_path = self.train_path / "PNEUMONIA"
        self.augmented_path = self.train_path / "NORMAL_augmented"
        
        # Create augmented directory
        self.augmented_path.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'original_normal': 0,
            'original_pneumonia': 0,
            'augmented_created': 0,
            'final_normal_total': 0,
            'augmentation_techniques_used': []
        }
    
    def count_images(self) -> Tuple[int, int]:
        """Count current images in NORMAL and PNEUMONIA folders"""
        normal_count = len([f for f in self.normal_path.glob('*.jpeg') if f.is_file()])
        pneumonia_count = len([f for f in self.pneumonia_path.glob('*.jpeg') if f.is_file()])
        
        self.stats['original_normal'] = normal_count
        self.stats['original_pneumonia'] = pneumonia_count
        
        logger.info(f"Current class distribution:")
        logger.info(f"  NORMAL: {normal_count:,} images")
        logger.info(f"  PNEUMONIA: {pneumonia_count:,} images")
        logger.info(f"  Imbalance ratio: 1:{pneumonia_count/normal_count:.2f}")
        
        return normal_count, pneumonia_count
    
    def medical_horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """Horizontal flip - medically valid for chest X-rays"""
        return cv2.flip(img, 1)
    
    def medical_rotation(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Small rotation - medically appropriate (±5-10 degrees)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def medical_brightness_adjustment(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Brightness adjustment - simulates different X-ray exposure"""
        adjusted = img.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def medical_contrast_adjustment(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Contrast adjustment - simulates different X-ray settings"""
        mean = np.mean(img)
        adjusted = (img - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def medical_zoom(self, img: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Slight zoom - simulates different patient positioning"""
        h, w = img.shape[:2]
        
        if zoom_factor > 1.0:  # Zoom in
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(img, (new_w, new_h))
            
            # Crop to original size from center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h + h, start_w:start_w + w]
        
        else:  # Zoom out
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(img, (new_w, new_h))
            
            # Pad to original size
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return np.pad(resized, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w), (0, 0)), mode='reflect')
    
    def medical_noise_addition(self, img: np.ndarray, noise_factor: float = 0.02) -> np.ndarray:
        """Add subtle noise - simulates X-ray machine variations"""
        noise = np.random.normal(0, noise_factor * 255, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def medical_gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma correction - simulates different X-ray processing"""
        normalized = img.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def apply_augmentation(self, img: np.ndarray, aug_type: str) -> np.ndarray:
        """Apply specific augmentation technique"""
        
        if aug_type == 'horizontal_flip':
            return self.medical_horizontal_flip(img)
        
        elif aug_type == 'rotate_5':
            return self.medical_rotation(img, 5)
        
        elif aug_type == 'rotate_-5':
            return self.medical_rotation(img, -5)
        
        elif aug_type == 'rotate_10':
            return self.medical_rotation(img, 10)
        
        elif aug_type == 'rotate_-10':
            return self.medical_rotation(img, -10)
        
        elif aug_type == 'brightness_up':
            return self.medical_brightness_adjustment(img, 1.15)
        
        elif aug_type == 'brightness_down':
            return self.medical_brightness_adjustment(img, 0.85)
        
        elif aug_type == 'contrast_up':
            return self.medical_contrast_adjustment(img, 1.2)
        
        elif aug_type == 'contrast_down':
            return self.medical_contrast_adjustment(img, 0.8)
        
        elif aug_type == 'zoom_in':
            return self.medical_zoom(img, 1.1)
        
        elif aug_type == 'zoom_out':
            return self.medical_zoom(img, 0.9)
        
        elif aug_type == 'noise':
            return self.medical_noise_addition(img)
        
        elif aug_type == 'gamma_up':
            return self.medical_gamma_correction(img, 1.2)
        
        elif aug_type == 'gamma_down':
            return self.medical_gamma_correction(img, 0.8)
        
        else:
            return img
    
    def create_augmented_images(self, target_count: int) -> int:
        """Create augmented images to reach target count"""
        
        # Get all NORMAL images
        normal_images = [f for f in self.normal_path.glob('*.jpeg') if f.is_file()]
        
        if not normal_images:
            logger.error("No NORMAL images found!")
            return 0
        
        # Calculate how many augmented images we need
        current_normal = len(normal_images)
        needed_augmentations = target_count - current_normal
        
        if needed_augmentations <= 0:
            logger.info("No augmentation needed - NORMAL class already sufficient")
            return 0
        
        logger.info(f"Creating {needed_augmentations:,} augmented NORMAL images...")
        
        # Medical augmentation techniques (order by medical appropriateness)
        augmentation_techniques = [
            'horizontal_flip',      # Most common and medically valid
            'rotate_5',            # Small rotations are medically appropriate
            'rotate_-5',
            'brightness_up',       # Simulates different X-ray exposures
            'brightness_down',
            'contrast_up',         # Simulates different X-ray settings
            'contrast_down',
            'rotate_10',           # Slightly larger rotations
            'rotate_-10',
            'zoom_in',             # Simulates patient positioning
            'zoom_out',
            'gamma_up',            # Different X-ray processing
            'gamma_down',
            'noise'                # X-ray machine variations
        ]
        
        # Calculate augmentations per original image
        augs_per_image = max(1, needed_augmentations // len(normal_images))
        
        augmented_count = 0
        techniques_used = set()
        
        # Create augmented images
        for i, img_path in enumerate(normal_images):
            if augmented_count >= needed_augmentations:
                break
            
            try:
                # Load original image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Apply multiple augmentations to this image
                for j in range(augs_per_image):
                    if augmented_count >= needed_augmentations:
                        break
                    
                    # Select augmentation technique
                    aug_technique = augmentation_techniques[j % len(augmentation_techniques)]
                    techniques_used.add(aug_technique)
                    
                    try:
                        # Apply augmentation
                        aug_img = self.apply_augmentation(img.copy(), aug_technique)
                        
                        # Save augmented image
                        aug_filename = f"{img_path.stem}_aug_{aug_technique}_{j:03d}.jpeg"
                        aug_path = self.augmented_path / aug_filename
                        
                        # Convert BGR to RGB for saving
                        aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(aug_img_rgb)
                        img_pil.save(aug_path, 'JPEG', quality=95)
                        
                        augmented_count += 1
                        
                        if augmented_count % 100 == 0:
                            logger.info(f"  Created {augmented_count:,}/{needed_augmentations:,} augmented images...")
                        
                    except Exception as e:
                        logger.warning(f"Error applying {aug_technique} to {img_path}: {e}")
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
        
        self.stats['augmented_created'] = augmented_count
        self.stats['augmentation_techniques_used'] = list(techniques_used)
        
        logger.info(f"✅ Successfully created {augmented_count:,} augmented NORMAL images")
        logger.info(f"📊 Techniques used: {', '.join(techniques_used)}")
        
        return augmented_count
    
    def balance_dataset(self) -> dict:
        """Main method to balance the dataset"""
        logger.info("🎯 Starting medical dataset balancing...")
        start_time = datetime.now()
        
        # Count current images
        normal_count, pneumonia_count = self.count_images()
        
        # Calculate target count (match pneumonia count)
        target_normal_count = pneumonia_count
        
        logger.info(f"\\n📊 BALANCING STRATEGY:")
        logger.info(f"  Target NORMAL count: {target_normal_count:,}")
        logger.info(f"  Current NORMAL count: {normal_count:,}")
        logger.info(f"  Augmentations needed: {target_normal_count - normal_count:,}")
        
        # Create augmented images
        augmented_created = self.create_augmented_images(target_normal_count)
        
        # Final counts
        final_normal = normal_count + augmented_created
        self.stats['final_normal_total'] = final_normal
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate report
        report = {
            'balancing_summary': {
                'dataset_path': str(self.dataset_path),
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': processing_time
            },
            'class_distribution': {
                'before_balancing': {
                    'NORMAL': normal_count,
                    'PNEUMONIA': pneumonia_count,
                    'ratio': f"1:{pneumonia_count/normal_count:.2f}"
                },
                'after_balancing': {
                    'NORMAL_original': normal_count,
                    'NORMAL_augmented': augmented_created,
                    'NORMAL_total': final_normal,
                    'PNEUMONIA': pneumonia_count,
                    'ratio': f"1:{pneumonia_count/final_normal:.2f}"
                }
            },
            'augmentation_details': {
                'images_created': augmented_created,
                'techniques_used': self.stats['augmentation_techniques_used'],
                'augmented_folder': str(self.augmented_path)
            },
            'balance_achieved': abs(final_normal - pneumonia_count) <= 50  # Within 50 images is considered balanced
        }
        
        return report
    
    def create_visualization(self, report: dict):
        """Create visualization of the balancing results"""
        try:
            import matplotlib.pyplot as plt
            
            # Data for plotting
            before_data = report['class_distribution']['before_balancing']
            after_data = report['class_distribution']['after_balancing']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Before balancing
            classes_before = ['NORMAL', 'PNEUMONIA']
            counts_before = [before_data['NORMAL'], before_data['PNEUMONIA']]
            colors_before = ['#FF6B6B', '#4ECDC4']
            
            bars1 = ax1.bar(classes_before, counts_before, color=colors_before, alpha=0.8)
            ax1.set_title('Before Balancing', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Images')
            ax1.set_ylim(0, max(counts_before) * 1.1)
            
            # Add count labels
            for bar, count in zip(bars1, counts_before):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_before) * 0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            # After balancing
            classes_after = ['NORMAL\\n(Original)', 'NORMAL\\n(Augmented)', 'PNEUMONIA']
            counts_after = [after_data['NORMAL_original'], after_data['NORMAL_augmented'], after_data['PNEUMONIA']]
            colors_after = ['#FF6B6B', '#FFB347', '#4ECDC4']
            
            bars2 = ax2.bar(classes_after, counts_after, color=colors_after, alpha=0.8)
            ax2.set_title('After Balancing', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Images')
            ax2.set_ylim(0, max(counts_after) * 1.1)
            
            # Add count labels
            for bar, count in zip(bars2, counts_after):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_after) * 0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Medical Dataset Class Balancing Results', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('dataset_balancing_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📊 Visualization saved as 'dataset_balancing_results.png'")
            
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")


def main():
    """Main function to balance the dataset"""
    print("⚖️  MEDICAL DATASET CLASS BALANCING")
    print("=" * 50)
    print("Balancing NORMAL vs PNEUMONIA classes in training set...")
    
    # Configuration
    dataset_path = "data/chest_xray_final"
    
    # Initialize augmenter
    augmenter = MedicalImageAugmenter(dataset_path)
    
    # Balance the dataset
    report = augmenter.balance_dataset()
    
    # Create visualization
    augmenter.create_visualization(report)
    
    # Save detailed report
    with open('dataset_balancing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\\n" + "=" * 50)
    print("🎉 DATASET BALANCING COMPLETED!")
    print("=" * 50)
    
    before = report['class_distribution']['before_balancing']
    after = report['class_distribution']['after_balancing']
    
    print(f"📊 BEFORE BALANCING:")
    print(f"   NORMAL: {before['NORMAL']:,} images")
    print(f"   PNEUMONIA: {before['PNEUMONIA']:,} images")
    print(f"   Ratio: {before['ratio']}")
    
    print(f"\\n📊 AFTER BALANCING:")
    print(f"   NORMAL (Original): {after['NORMAL_original']:,} images")
    print(f"   NORMAL (Augmented): {after['NORMAL_augmented']:,} images")
    print(f"   NORMAL (Total): {after['NORMAL_total']:,} images")
    print(f"   PNEUMONIA: {after['PNEUMONIA']:,} images")
    print(f"   New Ratio: {after['ratio']}")
    
    print(f"\\n🎯 AUGMENTATION DETAILS:")
    aug_details = report['augmentation_details']
    print(f"   Images created: {aug_details['images_created']:,}")
    print(f"   Techniques used: {len(aug_details['techniques_used'])}")
    print(f"   Augmented folder: {aug_details['augmented_folder']}")
    
    balance_status = "✅ BALANCED" if report['balance_achieved'] else "⚠️  PARTIALLY BALANCED"
    print(f"\\n{balance_status}")
    print("🚀 Dataset ready for unbiased training!")
    
    return report

if __name__ == "__main__":
    report = main()