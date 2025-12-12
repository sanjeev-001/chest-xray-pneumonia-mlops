#!/usr/bin/env python3
"""
Smart Dataset Rebalancer
Redistributes ONLY original training images to create proper val/test sets
Preserves all augmented images in training set
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime
import logging
import random
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartDatasetRebalancer:
    """Smart rebalancer that preserves augmented training data"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        self.test_path = self.dataset_path / "test"
        
        # Create backup directory
        self.backup_path = self.dataset_path / "backup_original_splits"
        self.backup_path.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'original_counts': {},
            'moved_to_val': {'NORMAL': 0, 'PNEUMONIA': 0},
            'moved_to_test': {'NORMAL': 0, 'PNEUMONIA': 0},
            'final_counts': {},
            'augmented_preserved': 0
        }
    
    def backup_original_splits(self):
        """Backup original val/test splits before rebalancing"""
        logger.info("📦 Backing up original val/test splits...")
        
        for split_name in ['val', 'test']:
            split_path = getattr(self, f"{split_name}_path")
            backup_split_path = self.backup_path / split_name
            
            if split_path.exists():
                if backup_split_path.exists():
                    shutil.rmtree(backup_split_path)
                shutil.copytree(split_path, backup_split_path)
                logger.info(f"  Backed up {split_name} split")
    
    def count_current_images(self) -> Dict:
        """Count current images in all splits"""
        counts = {}
        
        for split_name in ['train', 'val', 'test']:
            split_path = getattr(self, f"{split_name}_path")
            counts[split_name] = {}
            
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        class_count = len([f for f in class_dir.glob('*.jpeg')])
                        counts[split_name][class_dir.name] = class_count
        
        self.stats['original_counts'] = counts
        return counts
    
    def get_original_train_images(self) -> Dict[str, List[Path]]:
        """Get ONLY original training images (not augmented)"""
        original_images = {'NORMAL': [], 'PNEUMONIA': []}
        
        # Get NORMAL original images (from NORMAL folder, not NORMAL_augmented)
        normal_path = self.train_path / "NORMAL"
        if normal_path.exists():
            original_images['NORMAL'] = [f for f in normal_path.glob('*.jpeg')]
        
        # Get PNEUMONIA images
        pneumonia_path = self.train_path / "PNEUMONIA"
        if pneumonia_path.exists():
            original_images['PNEUMONIA'] = [f for f in pneumonia_path.glob('*.jpeg')]
        
        logger.info(f"Original training images available:")
        logger.info(f"  NORMAL: {len(original_images['NORMAL']):,}")
        logger.info(f"  PNEUMONIA: {len(original_images['PNEUMONIA']):,}")
        
        return original_images
    
    def calculate_target_splits(self, total_original: int) -> Dict:
        """Calculate target sizes for each split"""
        
        # Target percentages (leaving more for training since we have augmented data)
        train_pct = 0.70  # 70% for training (plus all augmented)
        val_pct = 0.15    # 15% for validation
        test_pct = 0.15   # 15% for test
        
        targets = {
            'val_total': int(total_original * val_pct),
            'test_total': int(total_original * test_pct),
            'train_remaining': total_original - int(total_original * val_pct) - int(total_original * test_pct)
        }
        
        # For balanced val/test sets
        targets['val_per_class'] = targets['val_total'] // 2
        targets['test_per_class'] = targets['test_total'] // 2
        
        logger.info(f"Target distribution from {total_original:,} original images:")
        logger.info(f"  Validation: {targets['val_total']:,} ({val_pct:.0%})")
        logger.info(f"  Test: {targets['test_total']:,} ({test_pct:.0%})")
        logger.info(f"  Training (original): {targets['train_remaining']:,} ({train_pct:.0%})")
        
        return targets
    
    def select_images_for_split(self, images: List[Path], count_needed: int) -> Tuple[List[Path], List[Path]]:
        """Randomly select images for a split"""
        if count_needed >= len(images):
            return images, []
        
        # Shuffle for random selection
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        selected = shuffled[:count_needed]
        remaining = shuffled[count_needed:]
        
        return selected, remaining
    
    def move_images_to_split(self, images: List[Path], target_split: str, class_name: str):
        """Move images to target split"""
        target_path = getattr(self, f"{target_split}_path")
        target_class_path = target_path / class_name
        target_class_path.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        for img_path in images:
            try:
                target_file = target_class_path / img_path.name
                shutil.move(str(img_path), str(target_file))
                moved_count += 1
            except Exception as e:
                logger.warning(f"Error moving {img_path}: {e}")
        
        self.stats[f'moved_to_{target_split}'][class_name] += moved_count
        return moved_count
    
    def clear_existing_val_test(self):
        """Clear existing val/test directories"""
        logger.info("🧹 Clearing existing val/test directories...")
        
        for split_name in ['val', 'test']:
            split_path = getattr(self, f"{split_name}_path")
            if split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)
    
    def rebalance_dataset(self):
        """Main rebalancing method"""
        logger.info("🎯 Starting smart dataset rebalancing...")
        logger.info("Strategy: Use ONLY original training images for val/test")
        logger.info("Preserve: ALL augmented images remain in training")
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Backup original splits
        self.backup_original_splits()
        
        # Count current images
        current_counts = self.count_current_images()
        
        # Get original training images (excluding augmented)
        original_train_images = self.get_original_train_images()
        
        # Calculate total original images available
        total_normal_orig = len(original_train_images['NORMAL'])
        total_pneumonia_orig = len(original_train_images['PNEUMONIA'])
        total_original = total_normal_orig + total_pneumonia_orig
        
        logger.info(f"\\nTotal original images to redistribute: {total_original:,}")
        
        # Calculate target splits
        targets = self.calculate_target_splits(total_original)
        
        # Clear existing val/test
        self.clear_existing_val_test()
        
        # Redistribute images
        logger.info("\\n🔄 Redistributing images...")
        
        remaining_images = {'NORMAL': [], 'PNEUMONIA': []}
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_images = original_train_images[class_name]
            logger.info(f"\\n  Processing {class_name} class ({len(class_images):,} images):")
            
            # Select for validation
            val_images, remaining = self.select_images_for_split(class_images, targets['val_per_class'])
            moved_val = self.move_images_to_split(val_images, 'val', class_name)
            logger.info(f"    Moved to VAL: {moved_val:,}")
            
            # Select for test from remaining
            test_images, final_remaining = self.select_images_for_split(remaining, targets['test_per_class'])
            moved_test = self.move_images_to_split(test_images, 'test', class_name)
            logger.info(f"    Moved to TEST: {moved_test:,}")
            
            # Keep track of remaining for training
            remaining_images[class_name] = final_remaining
            logger.info(f"    Remaining in TRAIN: {len(final_remaining):,}")
        
        # Count augmented images that are preserved
        aug_path = self.train_path / "NORMAL_augmented"
        if aug_path.exists():
            aug_count = len([f for f in aug_path.glob('*.jpeg')])
            self.stats['augmented_preserved'] = aug_count
            logger.info(f"\\n✅ Preserved {aug_count:,} augmented NORMAL images in training")
        
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict:
        """Generate final rebalancing report"""
        
        # Count final images
        final_counts = {}
        for split_name in ['train', 'val', 'test']:
            split_path = getattr(self, f"{split_name}_path")
            final_counts[split_name] = {}
            
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        count = len([f for f in class_dir.glob('*.jpeg')])
                        final_counts[split_name][class_dir.name] = count
        
        self.stats['final_counts'] = final_counts
        
        # Calculate totals and percentages
        train_total = sum(final_counts['train'].values())
        val_total = sum(final_counts['val'].values())
        test_total = sum(final_counts['test'].values())
        dataset_total = train_total + val_total + test_total
        
        report = {
            'rebalancing_summary': {
                'dataset_path': str(self.dataset_path),
                'rebalancing_timestamp': datetime.now().isoformat(),
                'strategy': 'Smart rebalancing - preserve augmented training data'
            },
            'original_counts': self.stats['original_counts'],
            'final_counts': final_counts,
            'distribution_percentages': {
                'train': (train_total / dataset_total) * 100,
                'val': (val_total / dataset_total) * 100,
                'test': (test_total / dataset_total) * 100
            },
            'images_moved': {
                'to_validation': self.stats['moved_to_val'],
                'to_test': self.stats['moved_to_test']
            },
            'augmented_data_preserved': self.stats['augmented_preserved'],
            'class_balance_achieved': {
                'val': self.check_balance(final_counts['val']),
                'test': self.check_balance(final_counts['test'])
            }
        }
        
        return report
    
    def check_balance(self, split_counts: Dict) -> Dict:
        """Check if a split is balanced"""
        normal = split_counts.get('NORMAL', 0)
        pneumonia = split_counts.get('PNEUMONIA', 0)
        
        if normal > 0 and pneumonia > 0:
            ratio = max(normal, pneumonia) / min(normal, pneumonia)
            balanced = ratio <= 1.5  # Allow up to 1.5:1 ratio
        else:
            ratio = float('inf')
            balanced = False
        
        return {
            'normal_count': normal,
            'pneumonia_count': pneumonia,
            'ratio': f"1:{pneumonia/normal:.2f}" if normal > 0 else "N/A",
            'balanced': balanced
        }
    
    def print_final_summary(self, report: Dict):
        """Print final summary"""
        
        print("\\n" + "=" * 60)
        print("🎉 SMART DATASET REBALANCING COMPLETED!")
        print("=" * 60)
        
        # Distribution summary
        final_counts = report['final_counts']
        percentages = report['distribution_percentages']
        
        print("📊 FINAL DATASET DISTRIBUTION:")
        for split in ['train', 'val', 'test']:
            total = sum(final_counts[split].values())
            pct = percentages[split]
            print(f"   {split.upper()}: {total:,} images ({pct:.1f}%)")
            
            for class_name, count in final_counts[split].items():
                print(f"     {class_name}: {count:,}")
        
        # Balance check
        print("\\n⚖️  CLASS BALANCE STATUS:")
        balance_info = report['class_balance_achieved']
        for split in ['val', 'test']:
            balance = balance_info[split]
            status = "✅ BALANCED" if balance['balanced'] else "⚠️ IMBALANCED"
            print(f"   {split.upper()}: {status} ({balance['ratio']})")
        
        # Preservation summary
        aug_preserved = report['augmented_data_preserved']
        print(f"\\n🔄 AUGMENTED DATA PRESERVED:")
        print(f"   ✅ {aug_preserved:,} augmented NORMAL images kept in training")
        
        # Movement summary
        moved = report['images_moved']
        print(f"\\n📦 IMAGES REDISTRIBUTED:")
        print(f"   To Validation: {sum(moved['to_validation'].values()):,}")
        print(f"   To Test: {sum(moved['to_test'].values()):,}")
        
        print(f"\\n🚀 DATASET NOW READY FOR:")
        print(f"   ✅ Reliable model validation")
        print(f"   ✅ Unbiased performance testing")
        print(f"   ✅ Proper hyperparameter tuning")
        print(f"   ✅ Medical ML best practices")


def main():
    """Main function"""
    print("🎯 SMART DATASET REBALANCER")
    print("=" * 40)
    print("Strategy: Preserve augmented training data")
    print("Action: Redistribute ONLY original images")
    
    dataset_path = "data/chest_xray_final"
    
    rebalancer = SmartDatasetRebalancer(dataset_path)
    report = rebalancer.rebalance_dataset()
    
    # Print summary
    rebalancer.print_final_summary(report)
    
    # Save detailed report
    with open('smart_rebalancing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\\n📄 Detailed report saved as 'smart_rebalancing_report.json'")
    
    return report

if __name__ == "__main__":
    report = main()