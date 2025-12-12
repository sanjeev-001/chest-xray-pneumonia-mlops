#!/usr/bin/env python3
"""
Simple script to copy and organize the chest X-ray dataset for MLOps pipeline
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_dataset():
    """Copy dataset from original location to MLOps pipeline structure"""
    
    # Source and target paths
    source_path = Path("data/chest_xray")
    target_path = Path("data/chest_xray_cleaned")
    
    # Check if source exists
    if not source_path.exists():
        logger.error(f"Source dataset not found at {source_path}")
        return False
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy each split while preserving structure
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    total_copied = 0
    
    for split in splits:
        split_source = source_path / split
        split_target = target_path / split
        
        if not split_source.exists():
            logger.warning(f"Split {split} not found in source")
            continue
            
        logger.info(f"Copying {split} split...")
        
        for class_name in classes:
            class_source = split_source / class_name
            class_target = split_target / class_name
            
            if not class_source.exists():
                logger.warning(f"Class {class_name} not found in {split}")
                continue
            
            # Create target class directory
            class_target.mkdir(parents=True, exist_ok=True)
            
            # Copy all files in this class
            files = [f for f in class_source.glob('*') if f.is_file()]
            logger.info(f"Copying {len(files)} files from {split}/{class_name}")
            
            for file_path in files:
                target_file = class_target / file_path.name
                shutil.copy2(file_path, target_file)
                total_copied += 1
    
    logger.info(f"Dataset copy completed! Total files copied: {total_copied}")
    
    # Verify the copy
    verify_copy(target_path)
    
    return True

def verify_copy(dataset_path):
    """Verify the copied dataset structure"""
    logger.info("Verifying copied dataset...")
    
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    total_files = 0
    
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            for class_name in classes:
                class_path = split_path / class_name
                if class_path.exists():
                    file_count = len([f for f in class_path.glob('*') if f.is_file()])
                    logger.info(f"{split}/{class_name}: {file_count} files")
                    total_files += file_count
    
    logger.info(f"Total files in cleaned dataset: {total_files}")

if __name__ == "__main__":
    print("🔄 Copying Chest X-Ray Dataset for MLOps Pipeline")
    print("=" * 50)
    
    success = copy_dataset()
    
    if success:
        print("\n✅ Dataset successfully copied to data/chest_xray_cleaned/")
        print("🚀 Ready for MLOps pipeline!")
    else:
        print("\n❌ Dataset copy failed. Please check the source path.")