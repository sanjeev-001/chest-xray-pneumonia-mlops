#!/usr/bin/env python3
"""
Complete Medical Augmentation - Full Class Balancing
Creates enough augmented images to fully balance NORMAL with PNEUMONIA
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from PIL import Image
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteAugmenter:
    """Complete augmentation to fully balance classes"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.normal_path = self.train_path / "NORMAL"
        self.pneumonia_path = self.train_path / "PNEUMONIA"
        self.augmented_path = self.train_path / "NORMAL_augmented"
        
        # Ensure augmented directory exists
        self.augmented_path.mkdir(exist_ok=True)
    
    def count_current_images(self):
        """Count all current images"""
        normal_original = len([f for f in self.normal_path.glob('*.jpeg')])
        normal_augmented = len([f for f in self.augmented_path.glob('*.jpeg')])
        pneumonia_count = len([f for f in self.pneumonia_path.glob('*.jpeg')])
        
        total_normal = normal_original + normal_augmented
        
        logger.info(f"Current counts:")
        logger.info(f"  NORMAL (Original): {normal_original:,}")
        logger.info(f"  NORMAL (Augmented): {normal_augmented:,}")
        logger.info(f"  NORMAL (Total): {total_normal:,}")
        logger.info(f"  PNEUMONIA: {pneumonia_count:,}")
        
        return normal_original, normal_augmented, total_normal, pneumonia_count
    
    def apply_advanced_augmentation(self, img: np.ndarray, technique: str) -> np.ndarray:
        """Apply advanced augmentation techniques"""
        
        if technique == 'horizontal_flip':
            return cv2.flip(img, 1)
        
        elif technique == 'vertical_flip':
            return cv2.flip(img, 0)
        
        elif technique == 'rotate_3':
            return self.rotate_image(img, 3)
        
        elif technique == 'rotate_-3':
            return self.rotate_image(img, -3)
        
        elif technique == 'rotate_7':
            return self.rotate_image(img, 7)
        
        elif technique == 'rotate_-7':
            return self.rotate_image(img, -7)
        
        elif technique == 'rotate_12':
            return self.rotate_image(img, 12)
        
        elif technique == 'rotate_-12':
            return self.rotate_image(img, -12)
        
        elif technique == 'brightness_110':
            return self.adjust_brightness(img, 1.1)
        
        elif technique == 'brightness_90':
            return self.adjust_brightness(img, 0.9)
        
        elif technique == 'brightness_120':
            return self.adjust_brightness(img, 1.2)
        
        elif technique == 'brightness_80':
            return self.adjust_brightness(img, 0.8)
        
        elif technique == 'contrast_110':
            return self.adjust_contrast(img, 1.1)
        
        elif technique == 'contrast_90':
            return self.adjust_contrast(img, 0.9)
        
        elif technique == 'contrast_125':
            return self.adjust_contrast(img, 1.25)
        
        elif technique == 'contrast_75':
            return self.adjust_contrast(img, 0.75)
        
        elif technique == 'zoom_105':
            return self.zoom_image(img, 1.05)
        
        elif technique == 'zoom_95':
            return self.zoom_image(img, 0.95)
        
        elif technique == 'zoom_115':
            return self.zoom_image(img, 1.15)
        
        elif technique == 'zoom_85':
            return self.zoom_image(img, 0.85)
        
        elif technique == 'gamma_110':
            return self.gamma_correction(img, 1.1)
        
        elif technique == 'gamma_90':
            return self.gamma_correction(img, 0.9)
        
        elif technique == 'gamma_125':
            return self.gamma_correction(img, 1.25)
        
        elif technique == 'gamma_75':
            return self.gamma_correction(img, 0.75)
        
        elif technique == 'noise_light':
            return self.add_noise(img, 0.01)
        
        elif technique == 'noise_medium':
            return self.add_noise(img, 0.02)
        
        elif technique == 'shift_right':
            return self.shift_image(img, 10, 0)
        
        elif technique == 'shift_left':
            return self.shift_image(img, -10, 0)
        
        elif technique == 'shift_up':
            return self.shift_image(img, 0, -10)
        
        elif technique == 'shift_down':
            return self.shift_image(img, 0, 10)
        
        else:
            return img
    
    def rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness"""
        adjusted = img.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast"""
        mean = np.mean(img)
        adjusted = (img - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def zoom_image(self, img: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom image"""
        h, w = img.shape[:2]
        
        if zoom_factor > 1.0:
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(img, (new_w, new_h))
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h + h, start_w:start_w + w]
        else:
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized = cv2.resize(img, (new_w, new_h))
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return np.pad(resized, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w), (0, 0)), mode='reflect')
    
    def gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction"""
        normalized = img.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def add_noise(self, img: np.ndarray, noise_factor: float) -> np.ndarray:
        """Add noise"""
        noise = np.random.normal(0, noise_factor * 255, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def shift_image(self, img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """Shift image"""
        h, w = img.shape[:2]
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def create_additional_augmentations(self, needed_count: int) -> int:
        """Create additional augmented images to reach target"""
        
        # Get original NORMAL images
        normal_images = [f for f in self.normal_path.glob('*.jpeg')]
        
        if not normal_images:
            logger.error("No original NORMAL images found!")
            return 0
        
        logger.info(f"Creating {needed_count:,} additional augmented images...")
        
        # Extended augmentation techniques
        techniques = [
            'horizontal_flip', 'vertical_flip',
            'rotate_3', 'rotate_-3', 'rotate_7', 'rotate_-7', 'rotate_12', 'rotate_-12',
            'brightness_110', 'brightness_90', 'brightness_120', 'brightness_80',
            'contrast_110', 'contrast_90', 'contrast_125', 'contrast_75',
            'zoom_105', 'zoom_95', 'zoom_115', 'zoom_85',
            'gamma_110', 'gamma_90', 'gamma_125', 'gamma_75',
            'noise_light', 'noise_medium',
            'shift_right', 'shift_left', 'shift_up', 'shift_down'
        ]
        
        created_count = 0
        technique_index = 0
        
        while created_count < needed_count:
            for img_path in normal_images:
                if created_count >= needed_count:
                    break
                
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Select technique (cycle through all techniques)
                    technique = techniques[technique_index % len(techniques)]
                    
                    # Apply augmentation
                    aug_img = self.apply_advanced_augmentation(img.copy(), technique)
                    
                    # Save augmented image
                    aug_filename = f"{img_path.stem}_aug_{technique}_{created_count:04d}.jpeg"
                    aug_path = self.augmented_path / aug_filename
                    
                    # Convert and save
                    aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(aug_img_rgb)
                    img_pil.save(aug_path, 'JPEG', quality=95)
                    
                    created_count += 1
                    technique_index += 1
                    
                    if created_count % 100 == 0:
                        logger.info(f"  Created {created_count:,}/{needed_count:,} additional images...")
                
                except Exception as e:
                    logger.warning(f"Error creating augmentation: {e}")
                    technique_index += 1
        
        return created_count
    
    def complete_balancing(self):
        """Complete the class balancing"""
        logger.info("🎯 Completing class balancing...")
        
        # Count current images
        normal_orig, normal_aug, total_normal, pneumonia_count = self.count_current_images()
        
        # Calculate how many more we need
        needed_additional = pneumonia_count - total_normal
        
        if needed_additional <= 0:
            logger.info("✅ Classes are already balanced!")
            return {
                'additional_created': 0,
                'final_balance': True,
                'final_counts': {
                    'NORMAL_total': total_normal,
                    'PNEUMONIA': pneumonia_count
                }
            }
        
        logger.info(f"Need {needed_additional:,} additional augmented images for perfect balance")
        
        # Create additional augmentations
        additional_created = self.create_additional_augmentations(needed_additional)
        
        # Final count
        final_normal_total = total_normal + additional_created
        
        logger.info(f"\\n✅ Balancing completed!")
        logger.info(f"Final counts:")
        logger.info(f"  NORMAL (Total): {final_normal_total:,}")
        logger.info(f"  PNEUMONIA: {pneumonia_count:,}")
        logger.info(f"  Ratio: 1:{pneumonia_count/final_normal_total:.2f}")
        
        return {
            'additional_created': additional_created,
            'final_balance': abs(final_normal_total - pneumonia_count) <= 10,
            'final_counts': {
                'NORMAL_original': normal_orig,
                'NORMAL_augmented_total': normal_aug + additional_created,
                'NORMAL_total': final_normal_total,
                'PNEUMONIA': pneumonia_count
            }
        }


def main():
    """Main function"""
    print("⚖️  COMPLETE CLASS BALANCING")
    print("=" * 40)
    
    dataset_path = "data/chest_xray_final"
    
    augmenter = CompleteAugmenter(dataset_path)
    result = augmenter.complete_balancing()
    
    print("\\n" + "=" * 40)
    print("🎉 COMPLETE BALANCING FINISHED!")
    print("=" * 40)
    
    final_counts = result['final_counts']
    print(f"📊 FINAL DATASET:")
    print(f"   NORMAL (Original): {final_counts.get('NORMAL_original', 0):,}")
    print(f"   NORMAL (Augmented): {final_counts.get('NORMAL_augmented_total', 0):,}")
    print(f"   NORMAL (Total): {final_counts['NORMAL_total']:,}")
    print(f"   PNEUMONIA: {final_counts['PNEUMONIA']:,}")
    
    ratio = final_counts['PNEUMONIA'] / final_counts['NORMAL_total']
    print(f"   Final Ratio: 1:{ratio:.2f}")
    
    if result['final_balance']:
        print("\\n✅ PERFECTLY BALANCED!")
    else:
        print("\\n⚖️  WELL BALANCED!")
    
    print("🚀 Dataset ready for unbiased training!")
    
    return result

if __name__ == "__main__":
    result = main()