#!/usr/bin/env python3
"""
Kaggle GPU Training Script for Chest X-Ray Pneumonia Detection
Optimized for Kaggle's GPU environment with fast training
"""

# Kaggle-specific imports and setup
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from PIL import Image
import cv2
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleChestXrayDataset(Dataset):
    """Optimized dataset for Kaggle environment"""
    
    def __init__(self, data_dir, transform=None, image_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
        
        logger.info(f"Loaded {len(self.image_paths)} images from {data_dir}")
    
    def _load_dataset(self):
        """Load dataset from Kaggle directory structure"""
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                for img_path in images:
                    self.image_paths.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                logger.info(f"Found {len(images)} images in {class_name}")
        
        # Handle augmented folders
        aug_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and 'augmented' in d.name.lower()]
        for aug_dir in aug_dirs:
            if 'NORMAL' in aug_dir.name:
                images = list(aug_dir.glob('*.jpeg')) + list(aug_dir.glob('*.jpg'))
                for img_path in images:
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # NORMAL class
                logger.info(f"Found {len(images)} augmented NORMAL images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
        except:
            # Fallback to PIL
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.image_size)
            image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class KaggleMedicalTrainer:
    """Optimized trainer for Kaggle GPU environment"""
    
    def __init__(self, dataset_path, batch_size=32, learning_rate=1e-4, epochs=25):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def create_model(self):
        """Create EfficientNet-B4 model for medical imaging"""
        logger.info("Creating EfficientNet-B4 model...")
        
        # Load pre-trained EfficientNet-B4
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Modify classifier for binary classification
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 2)  # NORMAL vs PNEUMONIA
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def create_data_loaders(self):
        """Create data loaders for training, validation, and testing"""
        logger.info("Creating data loaders...")
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = KaggleChestXrayDataset(
            data_dir=Path(self.dataset_path) / "train",
            transform=train_transform
        )
        
        val_dataset = KaggleChestXrayDataset(
            data_dir=Path(self.dataset_path) / "val",
            transform=val_transform
        )
        
        test_dataset = KaggleChestXrayDataset(
            data_dir=Path(self.dataset_path) / "test",
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate medical metrics
        metrics = self.calculate_medical_metrics(all_labels, all_preds, all_probs)
        
        return epoch_loss, epoch_acc, metrics
    
    def calculate_medical_metrics(self, y_true, y_pred, y_prob):
        """Calculate medical-specific metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Medical metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # AUC
        try:
            y_prob_positive = np.array(y_prob)[:, 1]
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob_positive)
        except:
            metrics['auc_roc'] = 0.0
        
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def train(self):
        """Main training function optimized for Kaggle"""
        logger.info("🚀 Starting Kaggle GPU training...")
        
        # Create model
        model = self.create_model()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # Training loop
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\\n=== EPOCH {epoch}/{self.epochs} ===")
            
            # Training
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate step
            scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_chest_xray_model.pth')
                logger.info(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
            logger.info(f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"\\n🎉 Training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation
        logger.info("\\n📊 Final evaluation on test set...")
        model.load_state_dict(torch.load('best_chest_xray_model.pth'))
        test_loss, test_acc, test_metrics = self.validate_epoch(model, test_loader, criterion)
        
        logger.info("\\n🏆 FINAL RESULTS:")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
        logger.info(f"Test Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        
        # Save results
        results = {
            'training_time_hours': training_time / 3600,
            'best_val_accuracy': best_val_acc,
            'test_metrics': test_metrics,
            'training_history': self.history,
            'model_architecture': 'efficientnet_b4',
            'dataset_size': len(train_loader.dataset),
            'device': str(self.device)
        }
        
        with open('kaggle_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot training history
        self.plot_training_history()
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('kaggle_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function for Kaggle training"""
    print("🏥 KAGGLE GPU TRAINING - CHEST X-RAY PNEUMONIA DETECTION")
    print("=" * 60)
    
    # Configuration for Kaggle
    config = {
        'dataset_path': '/kaggle/input/chest-xray-pneumonia/chest_xray',  # Kaggle dataset path
        'batch_size': 32,  # Good for GPU
        'learning_rate': 1e-4,
        'epochs': 25  # Reasonable for Kaggle time limits
    }
    
    # Check if running on Kaggle
    if '/kaggle/' in os.getcwd():
        logger.info("🎯 Running on Kaggle environment")
        # Kaggle-specific dataset path
        dataset_path = '/kaggle/input/chest-xray-pneumonia/chest_xray'
    else:
        logger.info("🏠 Running locally")
        dataset_path = 'data/chest_xray_final'
    
    # Initialize trainer
    trainer = KaggleMedicalTrainer(
        dataset_path=dataset_path,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        epochs=config['epochs']
    )
    
    # Start training
    results = trainer.train()
    
    print("\\n🎉 KAGGLE TRAINING COMPLETED!")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Training Time: {results['training_time_hours']:.2f} hours")
    
    return results

if __name__ == "__main__":
    results = main()