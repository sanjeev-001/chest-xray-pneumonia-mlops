"""
Medical Model Trainer
Implements comprehensive training pipeline for chest X-ray pneumonia detection
Optimized for EfficientNet-B4 with medical imaging best practices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

from .config import TrainingConfig
from .models import ModelFactory
from .metrics import MedicalMetrics
from .dataset import ChestXrayDataset
from .experiment_tracker import ExperimentLogger
from .model_registry import ModelRegistry, AutoModelRegistry

# Import optuna for hyperparameter optimization support
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive medical model trainer with MLOps best practices
    Handles training, validation, and evaluation for chest X-ray models
    """
    
    def __init__(self, config: TrainingConfig, use_mlflow: bool = True):
        self.config = config
        self.device = torch.device(config.device)
        self.scaler = GradScaler() if config.mixed_precision and config.device == 'cuda' else None
        self.use_mlflow = use_mlflow
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.epochs_without_improvement = 0
        
        # Medical metrics calculator
        self.medical_metrics = MedicalMetrics(config.class_names)
        
        # MLflow integration
        self.experiment_logger = None
        self.model_registry = None
        self.auto_registry = None
        
        if self.use_mlflow:
            self._setup_mlflow()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"ModelTrainer initialized with {config.model_architecture} on {self.device}")
        if self.use_mlflow:
            logger.info("MLflow experiment tracking enabled")
    
    def _setup_logging(self):
        """Setup training-specific logging"""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.config.experiment_name}_training.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking and model registry"""
        try:
            # Setup experiment logger
            self.experiment_logger = ExperimentLogger(
                experiment_name=f"chest_xray_{self.config.model_architecture}",
                tracking_uri="sqlite:///mlflow.db",
                artifact_location=str(Path(self.config.output_dir) / "mlflow_artifacts")
            )
            
            # Setup model registry
            self.model_registry = ModelRegistry(tracking_uri="sqlite:///mlflow.db")
            
            # Setup auto registry
            model_name = f"chest_xray_{self.config.model_architecture}_model"
            self.auto_registry = AutoModelRegistry(
                model_registry=self.model_registry,
                model_name=model_name,
                promotion_criteria={
                    'accuracy': 0.85,
                    'auc_roc': 0.80,
                    'sensitivity': 0.80,
                    'specificity': 0.80
                }
            )
            
            logger.info("MLflow integration setup completed")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.use_mlflow = False
    
    def setup_model(self) -> nn.Module:
        """Setup the model architecture"""
        logger.info(f"Setting up {self.config.model_architecture} model...")
        
        # Create model
        self.model = ModelFactory.create_model(
            architecture=self.config.model_architecture,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            pretrained=self.config.pretrained
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup for multi-GPU if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model)
        
        return self.model
    
    def setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration"""
        optimizer_config = self.config.get_optimizer_config()
        
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_config)
        elif self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_config)
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_config)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        logger.info(f"Setup {self.config.optimizer} optimizer with lr={self.config.learning_rate}")
        return self.optimizer
    
    def setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config.scheduler == 'none':
            return None
        
        scheduler_config = self.config.get_scheduler_config()
        
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config['eta_min']
            )
        elif self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
        
        logger.info(f"Setup {self.config.scheduler} scheduler")
        return self.scheduler
    
    def setup_criterion(self) -> nn.Module:
        """Setup loss function"""
        if self.config.loss_function == 'cross_entropy':
            if self.config.class_weights is not None:
                weights = torch.tensor(self.config.class_weights, dtype=torch.float32).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        elif self.config.loss_function == 'focal':
            from .losses import FocalLoss
            self.criterion = FocalLoss(
                alpha=self.config.focal_loss_alpha,
                gamma=self.config.focal_loss_gamma,
                num_classes=self.config.num_classes
            )
        
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
        logger.info(f"Setup {self.config.loss_function} loss function")
        return self.criterion
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders for training, validation, and testing"""
        logger.info("Setting up data loaders...")
        
        # Create datasets
        train_dataset = ChestXrayDataset(
            data_dir=Path(self.config.dataset_path) / "train",
            transform=self.config.get_augmentation_transforms(),
            image_size=self.config.image_size
        )
        
        val_dataset = ChestXrayDataset(
            data_dir=Path(self.config.dataset_path) / "val",
            transform=None,  # No augmentation for validation
            image_size=self.config.image_size
        )
        
        test_dataset = ChestXrayDataset(
            data_dir=Path(self.config.dataset_path) / "test",
            transform=None,  # No augmentation for testing
            image_size=self.config.image_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress tracking
        batch_count = len(train_loader)
        log_interval = max(1, batch_count // 10)  # Log 10 times per epoch
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Logging
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{batch_count}, "
                          f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, Any]]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        
        # Calculate comprehensive metrics
        metrics = self.medical_metrics.calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=np.array(all_probabilities)
        )
        
        return epoch_loss, epoch_acc, metrics
    
    def should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if not self.config.early_stopping:
            return False
        
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                return True
        
        return False
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / self.config.model_save_path
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        # Save regular checkpoint
        if epoch % self.config.save_checkpoint_frequency == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_model_path = checkpoint_dir / f"best_model_{self.config.experiment_name}.pth"
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
            logger.info(f"Best model saved: {best_model_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.epochs} epochs...")
        
        # Start MLflow run
        if self.use_mlflow and self.experiment_logger:
            run_tags = {
                'model_architecture': self.config.model_architecture,
                'dataset_path': self.config.dataset_path,
                'training_type': 'medical_imaging',
                'task': 'chest_xray_pneumonia_detection'
            }
            
            self.experiment_logger.start_run(
                run_name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=run_tags
            )
            
            # Log configuration
            self.experiment_logger.log_config(self.config)
        
        # Setup all components
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_criterion()
        
        # Setup data loaders
        train_loader, val_loader, test_loader = self.setup_data_loaders()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            # Handle gradual fine-tuning
            if (self.config.fine_tuning_strategy == 'gradual' and 
                epoch == self.config.freeze_epochs + 1):
                logger.info(f"Unfreezing backbone at epoch {epoch}")
                if hasattr(self.model, 'freeze_backbone'):
                    self.model.freeze_backbone(False)
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'freeze_backbone'):
                    self.model.module.freeze_backbone(False)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            if epoch % self.config.validation_frequency == 0:
                val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                # Check for best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                
                # Save checkpoint
                if self.config.save_best_model:
                    self.save_checkpoint(epoch, val_acc, is_best)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if self.config.scheduler == 'plateau':
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Epoch timing
                epoch_time = time.time() - epoch_start_time
                self.history['epoch_times'].append(epoch_time)
                
                # MLflow logging
                if self.use_mlflow and self.experiment_logger:
                    # Log training metrics
                    train_metrics = {
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch_time': epoch_time
                    }
                    self.experiment_logger.log_metrics(train_metrics, step=epoch)
                    
                    # Log validation metrics
                    val_metrics_mlflow = {
                        'val_loss': val_loss,
                        'val_accuracy': val_acc
                    }
                    self.experiment_logger.log_metrics(val_metrics_mlflow, step=epoch)
                    
                    # Log medical metrics
                    self.experiment_logger.log_medical_metrics(val_metrics, prefix="val_")
                
                # Console logging
                logger.info(f"Epoch {epoch}/{self.config.epochs}:")
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                logger.info(f"  Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
                logger.info(f"  Val F1: {val_metrics['f1_score']:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}")
                logger.info(f"  Epoch Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Early stopping check
                if self.should_stop_early(val_loss):
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break
            
            else:
                # Just update training history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                epoch_time = time.time() - epoch_start_time
                self.history['epoch_times'].append(epoch_time)
                
                logger.info(f"Epoch {epoch}/{self.config.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final evaluation on test set
        test_metrics = self.evaluate(test_loader)
        
        # MLflow final logging
        if self.use_mlflow and self.experiment_logger:
            # Log test metrics
            self.experiment_logger.log_medical_metrics(test_metrics, prefix="test_")
            
            # Log training history plots
            self.experiment_logger.log_training_plots(self.history, save_dir=self.config.output_dir)
            
            # Log and register model if performance is good
            if self.best_model_path and self.best_model_path.exists():
                try:
                    # Create model URI
                    model_uri = f"runs:/{self.experiment_logger.run_id}/model"
                    
                    # Log the model to MLflow
                    self.experiment_logger.log_model(
                        model=self.model,
                        model_name="chest_xray_model",
                        metadata={
                            'architecture': self.config.model_architecture,
                            'best_val_accuracy': self.best_val_acc,
                            'test_accuracy': test_metrics.get('accuracy', 0.0),
                            'training_epochs': len(self.history['train_loss'])
                        }
                    )
                    
                    # Auto-register model if criteria met
                    if self.auto_registry:
                        model_version, promoted = self.auto_registry.register_and_evaluate(
                            model_uri=model_uri,
                            validation_metrics=test_metrics,
                            description=f"Chest X-ray {self.config.model_architecture} model trained on {datetime.now().strftime('%Y-%m-%d')}",
                            auto_promote=True
                        )
                        
                        logger.info(f"Model registered as version {model_version.version}")
                        if promoted:
                            logger.info("Model automatically promoted to Production!")
                        else:
                            logger.info("Model staged for manual review")
                
                except Exception as e:
                    logger.warning(f"Model registration failed: {e}")
            
            # End MLflow run
            self.experiment_logger.end_run()
        
        # Generate training report
        training_report = self.generate_training_report(test_metrics, total_time)
        
        return training_report
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        # Load best model if available
        if self.best_model_path and self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {self.best_model_path}")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_metrics = self.medical_metrics.calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=np.array(all_probabilities)
        )
        
        logger.info("Test Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
        logger.info(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
        
        return test_metrics
    
    def generate_training_report(self, test_metrics: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'experiment_info': {
                'experiment_name': self.config.experiment_name,
                'model_architecture': self.config.model_architecture,
                'dataset_path': self.config.dataset_path,
                'training_time_seconds': total_time,
                'training_time_formatted': f"{total_time/3600:.2f} hours",
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            },
            'training_config': self.config.to_dict(),
            'training_history': self.history,
            'best_validation_accuracy': self.best_val_acc,
            'test_metrics': test_metrics,
            'model_info': {
                'best_model_path': str(self.best_model_path) if self.best_model_path else None,
                'total_epochs_trained': len(self.history['train_loss']),
                'early_stopping_triggered': len(self.history['train_loss']) < self.config.epochs
            }
        }
        
        # Save report
        report_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
        
        # Generate plots
        self.plot_training_history()
        
        return report
    
    def plot_training_history(self):
        """Plot training history"""
        if len(self.history['train_loss']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.config.experiment_name}', fontsize=16)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        if self.history['val_loss']:
            val_epochs = range(self.config.validation_frequency, 
                             len(self.history['val_loss']) * self.config.validation_frequency + 1, 
                             self.config.validation_frequency)
            axes[0, 0].plot(val_epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        if self.history['val_acc']:
            axes[0, 1].plot(val_epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Epoch time plot
        if self.history['epoch_times']:
            axes[1, 1].plot(epochs, self.history['epoch_times'], 'm-')
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")
    
    def train_with_early_stopping(self, max_epochs: int, patience: int, 
                                 trial: Optional['optuna.Trial'] = None) -> Dict[str, Any]:
        """
        Training method optimized for hyperparameter optimization with pruning support
        
        Args:
            max_epochs: Maximum number of epochs to train
            patience: Early stopping patience
            trial: Optuna trial for pruning (optional)
            
        Returns:
            Dictionary with final metrics
        """
        if not OPTUNA_AVAILABLE and trial is not None:
            logger.warning("Optuna not available, ignoring trial parameter")
            trial = None
        
        logger.info(f"Starting optimization training for max {max_epochs} epochs with patience {patience}")
        
        # Setup model and training components
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_criterion()
        
        # Setup data loaders
        train_loader, val_loader, test_loader = self.setup_data_loaders()
        
        # Training state
        best_val_metric = 0.0
        epochs_without_improvement = 0
        
        # Start MLflow run if available
        if self.use_mlflow and self.experiment_logger:
            self.experiment_logger.start_run(run_name=f"optimization_{self.config.experiment_name}")
            self.experiment_logger.log_params(self.config.to_dict())
        
        try:
            for epoch in range(max_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                
                # Validation phase
                val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['epoch_times'].append(epoch_time)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                # Log metrics
                if self.use_mlflow and self.experiment_logger:
                    metrics_to_log = {
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time
                    }
                    metrics_to_log.update(val_metrics)
                    self.experiment_logger.log_metrics(metrics_to_log, step=epoch)
                
                # Determine primary metric for optimization
                primary_metric = val_metrics.get('f1_score', val_acc)
                
                # Check for improvement
                if primary_metric > best_val_metric:
                    best_val_metric = primary_metric
                    epochs_without_improvement = 0
                    
                    # Save best model
                    self.save_checkpoint(epoch, val_acc, is_best=True)
                else:
                    epochs_without_improvement += 1
                
                # Log progress
                logger.info(
                    f"Epoch {epoch+1}/{max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"F1: {val_metrics.get('f1_score', 0):.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Optuna pruning
                if trial is not None:
                    # Report intermediate value for pruning
                    trial.report(primary_metric, epoch)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"Trial pruned at epoch {epoch+1}")
                        raise optuna.TrialPruned()
                
                # Early stopping check
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            # Final evaluation on validation set
            final_val_loss, final_val_acc, final_val_metrics = self.validate_epoch(val_loader)
            
            # Prepare results
            results = {
                'val_accuracy': final_val_acc,
                'val_loss': final_val_loss,
                'val_f1_score': final_val_metrics.get('f1_score', 0.0),
                'val_auc_roc': final_val_metrics.get('auc_roc', 0.0),
                'val_precision': final_val_metrics.get('precision', 0.0),
                'val_recall': final_val_metrics.get('recall', 0.0),
                'val_sensitivity': final_val_metrics.get('sensitivity', 0.0),
                'val_specificity': final_val_metrics.get('specificity', 0.0),
                'epochs_trained': len(self.history['train_loss']),
                'best_val_metric': best_val_metric,
                'total_time': sum(self.history['epoch_times'])
            }
            
            # Log final results
            if self.use_mlflow and self.experiment_logger:
                self.experiment_logger.log_metrics(results)
                self.experiment_logger.end_run()
            
            logger.info(f"Optimization training completed. Best metric: {best_val_metric:.4f}")
            
            return results
            
        except Exception as e:
            if self.use_mlflow and self.experiment_logger:
                self.experiment_logger.end_run(status='FAILED')
            
            if isinstance(e, optuna.TrialPruned):
                raise  # Re-raise pruning exception
            else:
                logger.error(f"Training failed: {str(e)}")
                raise


# Alias for backward compatibility
MedicalImageTrainer = ModelTrainer


if __name__ == "__main__":
    # Test trainer setup
    from .config import get_config_for_scenario
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ModelTrainer...")
    
    # Create test configuration
    config = get_config_for_scenario('quick_test')
    config.dataset_path = 'data/chest_xray_final'  # Update path
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    print(f"Trainer initialized with {config.model_architecture}")
    print(f"Device: {trainer.device}")
    print(f"Mixed precision: {trainer.scaler is not None}")
    
    print("✅ ModelTrainer setup successful!")