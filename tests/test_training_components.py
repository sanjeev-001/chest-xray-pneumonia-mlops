"""
Unit Tests for Training Components
Tests for model training, evaluation, and related functionality
"""

import pytest
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os
from PIL import Image

from training.config import TrainingConfig, MedicalTrainingPresets
from training.trainer import ModelTrainer
from training.models import ModelFactory, MedicalCNNArchitecture
from training.metrics import MedicalMetrics
from training.losses import FocalLoss, WeightedCrossEntropyLoss
from training.dataset import ChestXrayDataset


class TestTrainingConfig:
    """Test TrainingConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.model_architecture == 'efficientnet_b4'
        assert config.num_classes == 2
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.optimizer == 'adamw'
        assert config.class_names == ['NORMAL', 'PNEUMONIA']
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            TrainingConfig(learning_rate=-1.0)
        
        # Test invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            TrainingConfig(batch_size=0)
        
        # Test invalid epochs
        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            TrainingConfig(epochs=-1)
        
        # Test invalid dropout rate
        with pytest.raises(ValueError, match="Dropout rate must be between 0 and 1"):
            TrainingConfig(dropout_rate=1.5)
    
    def test_config_serialization(self):
        """Test configuration saving and loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Create and save config
            original_config = TrainingConfig(
                learning_rate=5e-4,
                batch_size=16,
                epochs=25
            )
            original_config.save_config(config_path)
            
            # Load config
            loaded_config = TrainingConfig.load_config(config_path)
            
            assert loaded_config.learning_rate == 5e-4
            assert loaded_config.batch_size == 16
            assert loaded_config.epochs == 25
            
        finally:
            os.unlink(config_path)
    
    def test_preset_configs(self):
        """Test preset configurations"""
        quick_config = MedicalTrainingPresets.get_quick_test_config()
        assert quick_config.epochs == 5
        assert quick_config.batch_size == 8
        
        dev_config = MedicalTrainingPresets.get_development_config()
        assert dev_config.epochs == 25
        assert dev_config.batch_size == 16
        
        prod_config = MedicalTrainingPresets.get_production_config()
        assert prod_config.epochs == 100
        assert prod_config.mixed_precision is True


class TestModelFactory:
    """Test ModelFactory class"""
    
    def test_supported_architectures(self):
        """Test that all supported architectures can be created"""
        architectures = ['efficientnet_b4', 'resnet50', 'densenet121']
        
        for arch in architectures:
            model = ModelFactory.create_model(
                architecture=arch,
                num_classes=2,
                pretrained=False  # Avoid downloading weights in tests
            )
            assert isinstance(model, nn.Module)
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (1, 2)  # batch_size=1, num_classes=2
    
    def test_invalid_architecture(self):
        """Test error handling for invalid architecture"""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ModelFactory.create_model(
                architecture='invalid_arch',
                num_classes=2
            )
    
    def test_model_customization(self):
        """Test model customization options"""
        model = ModelFactory.create_model(
            architecture='efficientnet_b4',
            num_classes=5,  # Different number of classes
            dropout_rate=0.5,
            pretrained=False
        )
        
        # Test output shape
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (2, 5)  # batch_size=2, num_classes=5


class TestMedicalMetrics:
    """Test MedicalMetrics class"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing"""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_prob = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9],
            [0.85, 0.15], [0.3, 0.7], [0.25, 0.75], [0.9, 0.1]
        ])
        return y_true, y_pred, y_prob
    
    def test_metrics_calculation(self, sample_predictions):
        """Test medical metrics calculation"""
        y_true, y_pred, y_prob = sample_predictions
        
        metrics_calculator = MedicalMetrics(['NORMAL', 'PNEUMONIA'])
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        # Perfect predictions should give perfect metrics
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['sensitivity'] == 1.0
        assert metrics['specificity'] == 1.0
        assert metrics['auc_roc'] == 1.0
    
    def test_imperfect_predictions(self):
        """Test metrics with imperfect predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])  # One false positive, one false negative
        y_prob = np.array([
            [0.8, 0.2], [0.4, 0.6], [0.3, 0.7], 
            [0.2, 0.8], [0.9, 0.1], [0.6, 0.4]
        ])
        
        metrics_calculator = MedicalMetrics(['NORMAL', 'PNEUMONIA'])
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        # Should have accuracy of 4/6 = 0.667
        assert abs(metrics['accuracy'] - 0.6667) < 0.001
        assert metrics['accuracy'] < 1.0
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics
    
    def test_confusion_matrix_format(self, sample_predictions):
        """Test confusion matrix formatting"""
        y_true, y_pred, y_prob = sample_predictions
        
        metrics_calculator = MedicalMetrics(['NORMAL', 'PNEUMONIA'])
        metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        cm = metrics['confusion_matrix']
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)
        
        # For perfect predictions, diagonal should be non-zero
        assert cm[0, 0] > 0  # True negatives
        assert cm[1, 1] > 0  # True positives
        assert cm[0, 1] == 0  # False positives
        assert cm[1, 0] == 0  # False negatives


class TestLossFunctions:
    """Test custom loss functions"""
    
    def test_focal_loss(self):
        """Test FocalLoss implementation"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, num_classes=2)
        
        # Create sample predictions and targets
        predictions = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        
        loss = focal_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0  # Loss should be non-negative
        assert loss.requires_grad  # Should be differentiable
    
    def test_weighted_cross_entropy(self):
        """Test WeightedCrossEntropyLoss implementation"""
        class_weights = [1.0, 2.0]  # Weight pneumonia class more heavily
        weighted_loss = WeightedCrossEntropyLoss(class_weights)
        
        predictions = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        
        loss = weighted_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
        assert loss.requires_grad
    
    def test_loss_comparison(self):
        """Test that different losses give different values"""
        predictions = torch.tensor([[1.5, 0.5], [0.5, 1.5]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        
        # Standard cross entropy
        ce_loss = nn.CrossEntropyLoss()
        ce_value = ce_loss(predictions, targets)
        
        # Focal loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, num_classes=2)
        focal_value = focal_loss(predictions, targets)
        
        # Weighted cross entropy
        weighted_loss = WeightedCrossEntropyLoss([1.0, 2.0])
        weighted_value = weighted_loss(predictions, targets)
        
        # All should be different
        assert not torch.allclose(ce_value, focal_value)
        assert not torch.allclose(ce_value, weighted_value)
        assert not torch.allclose(focal_value, weighted_value)


class TestModelTrainer:
    """Test ModelTrainer class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock training configuration"""
        return TrainingConfig(
            dataset_path=temp_dir,
            epochs=2,
            batch_size=4,
            model_architecture='efficientnet_b4',
            experiment_name='test_training',
            output_dir=temp_dir,
            device='cpu',  # Force CPU for testing
            use_augmentation=False,  # Disable augmentation for testing
            early_stopping=False  # Disable early stopping for testing
        )
    
    def test_trainer_initialization(self, mock_config):
        """Test trainer initialization"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        
        assert trainer.config == mock_config
        assert trainer.device.type == 'cpu'
        assert trainer.model is None  # Not set up yet
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.criterion is None
        assert len(trainer.history['train_loss']) == 0
    
    def test_model_setup(self, mock_config):
        """Test model setup"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        model = trainer.setup_model()
        
        assert isinstance(model, nn.Module)
        assert trainer.model is not None
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 2)
    
    def test_optimizer_setup(self, mock_config):
        """Test optimizer setup"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        trainer.setup_model()
        optimizer = trainer.setup_optimizer()
        
        assert trainer.optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
        
        # Test different optimizers
        for opt_name in ['adam', 'adamw', 'sgd']:
            mock_config.optimizer = opt_name
            trainer = ModelTrainer(mock_config, use_mlflow=False)
            trainer.setup_model()
            trainer.setup_optimizer()
            assert trainer.optimizer is not None
    
    def test_scheduler_setup(self, mock_config):
        """Test learning rate scheduler setup"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        trainer.setup_model()
        trainer.setup_optimizer()
        
        # Test different schedulers
        for scheduler_name in ['cosine', 'step', 'plateau', 'none']:
            mock_config.scheduler = scheduler_name
            trainer = ModelTrainer(mock_config, use_mlflow=False)
            trainer.setup_model()
            trainer.setup_optimizer()
            scheduler = trainer.setup_scheduler()
            
            if scheduler_name == 'none':
                assert scheduler is None
            else:
                assert scheduler is not None
    
    def test_criterion_setup(self, mock_config):
        """Test loss function setup"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        
        # Test different loss functions
        for loss_name in ['cross_entropy', 'focal']:
            mock_config.loss_function = loss_name
            trainer = ModelTrainer(mock_config, use_mlflow=False)
            criterion = trainer.setup_criterion()
            
            assert criterion is not None
            assert isinstance(criterion, nn.Module)
    
    def test_early_stopping_logic(self, mock_config):
        """Test early stopping logic"""
        mock_config.early_stopping = True
        mock_config.early_stopping_patience = 3
        mock_config.early_stopping_min_delta = 0.01
        
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        
        # Test no early stopping initially
        assert not trainer.should_stop_early(1.0)
        
        # Simulate improving validation loss
        trainer.best_val_loss = 1.0
        assert not trainer.should_stop_early(0.8)  # Improvement
        
        # Simulate no improvement
        trainer.best_val_loss = 0.8
        trainer.epochs_without_improvement = 0
        
        assert not trainer.should_stop_early(0.81)  # Small increase, increment counter
        assert trainer.epochs_without_improvement == 1
        
        assert not trainer.should_stop_early(0.82)  # Another increase
        assert trainer.epochs_without_improvement == 2
        
        assert not trainer.should_stop_early(0.83)  # Another increase
        assert trainer.epochs_without_improvement == 3
        
        assert trainer.should_stop_early(0.84)  # Should trigger early stopping
    
    @patch('training.trainer.ChestXrayDataset')
    def test_data_loader_setup(self, mock_dataset_class, mock_config, temp_dir):
        """Test data loader setup with mocked dataset"""
        # Create mock dataset directories
        for split in ['train', 'val', 'test']:
            split_dir = Path(temp_dir) / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock dataset class
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset_class.return_value = mock_dataset
        
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        train_loader, val_loader, test_loader = trainer.setup_data_loaders()
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Verify dataset was called correctly
        assert mock_dataset_class.call_count == 3  # train, val, test
    
    def test_checkpoint_saving(self, mock_config, temp_dir):
        """Test model checkpoint saving"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.setup_scheduler()
        
        # Save checkpoint
        trainer.save_checkpoint(epoch=5, val_acc=0.85, is_best=True)
        
        # Check that checkpoint file was created
        checkpoint_dir = Path(temp_dir) / mock_config.model_save_path
        assert checkpoint_dir.exists()
        
        # Check for best model file
        best_model_files = list(checkpoint_dir.glob("best_model_*.pth"))
        assert len(best_model_files) > 0
    
    def test_training_history_tracking(self, mock_config):
        """Test training history tracking"""
        trainer = ModelTrainer(mock_config, use_mlflow=False)
        
        # Simulate adding training history
        trainer.history['train_loss'].extend([1.0, 0.8, 0.6])
        trainer.history['train_acc'].extend([0.6, 0.7, 0.8])
        trainer.history['val_loss'].extend([1.2, 0.9, 0.7])
        trainer.history['val_acc'].extend([0.5, 0.65, 0.75])
        
        assert len(trainer.history['train_loss']) == 3
        assert len(trainer.history['train_acc']) == 3
        assert len(trainer.history['val_loss']) == 3
        assert len(trainer.history['val_acc']) == 3
        
        # Test that values are tracked correctly
        assert trainer.history['train_loss'][-1] == 0.6  # Latest loss
        assert trainer.history['val_acc'][-1] == 0.75   # Latest accuracy


class TestDatasetIntegration:
    """Test dataset integration with training"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory with sample images"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = Path(temp_dir) / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                for i in range(2):  # 2 images per class
                    img = Image.new('RGB', (224, 224), color='white')
                    img_path = class_dir / f"image_{i}.jpg"
                    img.save(img_path)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataset_creation(self, temp_dataset_dir):
        """Test dataset creation with real directory structure"""
        train_dir = Path(temp_dataset_dir) / 'train'
        
        dataset = ChestXrayDataset(
            data_dir=train_dir,
            transform=None,
            image_size=(224, 224)
        )
        
        assert len(dataset) == 4  # 2 classes × 2 images each
        
        # Test getting an item
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert label in [0, 1]
    
    def test_training_with_real_data(self, temp_dataset_dir):
        """Test training setup with real data structure"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=1,
            batch_size=2,
            model_architecture='efficientnet_b4',
            device='cpu',
            use_augmentation=False,
            early_stopping=False
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        
        # Setup all components
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.setup_criterion()
        
        # Setup data loaders
        train_loader, val_loader, test_loader = trainer.setup_data_loaders()
        
        # Verify data loaders work
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Test getting a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            assert images.shape[0] <= config.batch_size
            assert images.shape[1:] == (3, 224, 224)
            assert len(labels) == images.shape[0]
            break  # Just test first batch


class TestTrainingIntegration:
    """Integration tests for training components"""
    
    def test_full_training_setup(self):
        """Test that all training components can be set up together"""
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            model_architecture='efficientnet_b4',
            device='cpu',
            use_augmentation=False
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        
        # Test that all components can be set up without errors
        model = trainer.setup_model()
        optimizer = trainer.setup_optimizer()
        scheduler = trainer.setup_scheduler()
        criterion = trainer.setup_criterion()
        
        assert model is not None
        assert optimizer is not None
        assert criterion is not None
        # scheduler can be None if scheduler='none'
    
    def test_model_forward_pass(self):
        """Test model forward pass with different configurations"""
        for arch in ['efficientnet_b4', 'resnet50']:
            config = TrainingConfig(
                model_architecture=arch,
                device='cpu'
            )
            
            trainer = ModelTrainer(config, use_mlflow=False)
            model = trainer.setup_model()
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (2, 2)  # batch_size=2, num_classes=2
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])