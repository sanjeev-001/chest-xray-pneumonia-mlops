"""
Integration Tests for Training Pipeline
Tests the complete training workflow integration
"""

import pytest
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

from training.config import TrainingConfig
from training.trainer import ModelTrainer
from training.dataset import ChestXrayDataset
from training.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig


class TestTrainingIntegration:
    """Integration tests for the complete training pipeline"""
    
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
                for i in range(3):  # 3 images per class per split
                    img = Image.new('RGB', (224, 224), color='white')
                    img_path = class_dir / f"image_{i}.jpg"
                    img.save(img_path)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_complete_training_setup(self, temp_dataset_dir):
        """Test complete training pipeline setup"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=2,
            batch_size=4,
            model_architecture='efficientnet_b4',
            device='cpu',
            use_augmentation=False,
            early_stopping=False,
            experiment_name='integration_test'
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        
        # Setup all components
        model = trainer.setup_model()
        optimizer = trainer.setup_optimizer()
        scheduler = trainer.setup_scheduler()
        criterion = trainer.setup_criterion()
        
        # Setup data loaders
        train_loader, val_loader, test_loader = trainer.setup_data_loaders()
        
        # Verify everything is set up correctly
        assert model is not None
        assert optimizer is not None
        assert criterion is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Test forward pass with real data
        model.eval()
        for batch_idx, (images, labels) in enumerate(train_loader):
            with torch.no_grad():
                outputs = model(images)
            
            assert outputs.shape[0] == images.shape[0]
            assert outputs.shape[1] == config.num_classes
            assert not torch.isnan(outputs).any()
            break  # Just test first batch
    
    def test_training_epoch_simulation(self, temp_dataset_dir):
        """Test simulated training epoch"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=1,
            batch_size=2,
            model_architecture='efficientnet_b4',
            device='cpu',
            use_augmentation=False,
            learning_rate=1e-3  # Higher LR for faster convergence in test
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        
        # Setup components
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.setup_criterion()
        train_loader, val_loader, _ = trainer.setup_data_loaders()
        
        # Simulate one training epoch
        initial_loss = None
        for epoch in range(1):
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # Verify training metrics
            assert isinstance(train_loss, float)
            assert isinstance(train_acc, float)
            assert train_loss >= 0.0
            assert 0.0 <= train_acc <= 1.0
            
            if initial_loss is None:
                initial_loss = train_loss
        
        # Test validation epoch
        val_loss, val_acc, val_metrics = trainer.validate_epoch(val_loader)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0.0
        assert 0.0 <= val_acc <= 1.0
        assert 'precision' in val_metrics
        assert 'recall' in val_metrics
        assert 'f1_score' in val_metrics
    
    def test_checkpoint_and_history(self, temp_dataset_dir):
        """Test checkpoint saving and history tracking"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=1,
            batch_size=2,
            device='cpu',
            save_best_model=True,
            save_checkpoint_frequency=1
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.setup_scheduler()
        
        # Simulate training history
        trainer.history['train_loss'] = [1.0, 0.8, 0.6]
        trainer.history['val_loss'] = [1.2, 0.9, 0.7]
        trainer.history['train_acc'] = [0.6, 0.7, 0.8]
        trainer.history['val_acc'] = [0.5, 0.65, 0.75]
        
        # Test checkpoint saving
        trainer.save_checkpoint(epoch=1, val_acc=0.75, is_best=True)
        
        # Verify checkpoint was saved
        checkpoint_dir = Path(config.output_dir) / config.model_save_path
        assert checkpoint_dir.exists()
        
        best_model_files = list(checkpoint_dir.glob("best_model_*.pth"))
        assert len(best_model_files) > 0
        
        # Test history tracking
        assert len(trainer.history['train_loss']) == 3
        assert trainer.history['val_acc'][-1] == 0.75
    
    def test_early_stopping_integration(self, temp_dataset_dir):
        """Test early stopping integration"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=10,
            batch_size=2,
            device='cpu',
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_min_delta=0.01
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        
        # Test early stopping logic
        trainer.best_val_loss = 1.0
        trainer.epochs_without_improvement = 0
        
        # Simulate improving loss
        assert not trainer.should_stop_early(0.8)
        assert trainer.epochs_without_improvement == 0
        
        # Simulate no improvement
        trainer.best_val_loss = 0.8
        assert not trainer.should_stop_early(0.81)  # Small increase
        assert trainer.epochs_without_improvement == 1
        
        # Continue without improvement
        assert not trainer.should_stop_early(0.82)
        assert trainer.epochs_without_improvement == 2
        
        assert not trainer.should_stop_early(0.83)
        assert trainer.epochs_without_improvement == 3
        
        # Should trigger early stopping
        assert trainer.should_stop_early(0.84)
    
    def test_hyperparameter_optimization_integration(self, temp_dataset_dir):
        """Test hyperparameter optimization integration"""
        base_config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=1,
            batch_size=2,
            device='cpu'
        )
        
        opt_config = OptimizationConfig(
            n_trials=2,  # Very small for testing
            max_epochs_per_trial=1,
            study_name='test_integration'
        )
        
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # Test search space definition
        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = 'efficientnet_b4'
        mock_trial.suggest_float.return_value = 1e-4
        mock_trial.suggest_int.return_value = 10
        
        search_space = optimizer.define_search_space(mock_trial)
        
        # Verify search space contains expected parameters
        expected_params = [
            'model_architecture', 'learning_rate', 'batch_size',
            'optimizer', 'weight_decay', 'dropout_rate'
        ]
        
        for param in expected_params:
            assert param in search_space
        
        # Test trial config creation
        trial_config = optimizer._create_trial_config(search_space)
        
        assert isinstance(trial_config, TrainingConfig)
        assert trial_config.dataset_path == base_config.dataset_path
        assert trial_config.epochs == opt_config.max_epochs_per_trial
    
    def test_model_evaluation_integration(self, temp_dataset_dir):
        """Test model evaluation integration"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            epochs=1,
            batch_size=2,
            device='cpu',
            use_augmentation=False
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        trainer.setup_model()
        trainer.setup_criterion()
        
        # Setup data loader
        _, _, test_loader = trainer.setup_data_loaders()
        
        # Test evaluation
        test_metrics = trainer.evaluate(test_loader)
        
        # Verify evaluation metrics
        assert 'accuracy' in test_metrics
        assert 'precision' in test_metrics
        assert 'recall' in test_metrics
        assert 'f1_score' in test_metrics
        assert 'auc_roc' in test_metrics
        assert 'sensitivity' in test_metrics
        assert 'specificity' in test_metrics
        
        # Verify metric ranges
        for metric_name, metric_value in test_metrics.items():
            if metric_name not in ['confusion_matrix', 'classification_report']:
                assert 0.0 <= metric_value <= 1.0, f"{metric_name} = {metric_value} is out of range"
    
    def test_different_model_architectures(self, temp_dataset_dir):
        """Test training with different model architectures"""
        architectures = ['efficientnet_b4', 'resnet50', 'densenet121']
        
        for arch in architectures:
            config = TrainingConfig(
                dataset_path=temp_dataset_dir,
                epochs=1,
                batch_size=2,
                model_architecture=arch,
                device='cpu',
                use_augmentation=False
            )
            
            trainer = ModelTrainer(config, use_mlflow=False)
            
            # Test that model can be set up and used
            model = trainer.setup_model()
            assert model is not None
            
            # Test forward pass
            model.eval()
            dummy_input = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (2, 2)
            assert not torch.isnan(output).any()
    
    def test_different_optimizers_and_schedulers(self, temp_dataset_dir):
        """Test training with different optimizers and schedulers"""
        optimizers = ['adam', 'adamw', 'sgd']
        schedulers = ['cosine', 'step', 'plateau', 'none']
        
        for opt_name in optimizers:
            for sched_name in schedulers:
                config = TrainingConfig(
                    dataset_path=temp_dataset_dir,
                    epochs=1,
                    batch_size=2,
                    optimizer=opt_name,
                    scheduler=sched_name,
                    device='cpu'
                )
                
                trainer = ModelTrainer(config, use_mlflow=False)
                trainer.setup_model()
                
                # Test optimizer setup
                optimizer = trainer.setup_optimizer()
                assert optimizer is not None
                
                # Test scheduler setup
                scheduler = trainer.setup_scheduler()
                if sched_name == 'none':
                    assert scheduler is None
                else:
                    assert scheduler is not None
    
    def test_loss_functions_integration(self, temp_dataset_dir):
        """Test training with different loss functions"""
        loss_functions = ['cross_entropy', 'focal']
        
        for loss_name in loss_functions:
            config = TrainingConfig(
                dataset_path=temp_dataset_dir,
                epochs=1,
                batch_size=2,
                loss_function=loss_name,
                device='cpu'
            )
            
            trainer = ModelTrainer(config, use_mlflow=False)
            
            # Test criterion setup
            criterion = trainer.setup_criterion()
            assert criterion is not None
            
            # Test loss computation
            dummy_predictions = torch.randn(2, 2)
            dummy_targets = torch.tensor([0, 1], dtype=torch.long)
            
            loss = criterion(dummy_predictions, dummy_targets)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0.0
            assert loss.requires_grad


class TestDatasetIntegration:
    """Test dataset integration with training pipeline"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure with more images for better testing
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = Path(temp_dir) / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images with different colors for variety
                colors = ['white', 'lightgray', 'gray']
                for i in range(5):  # 5 images per class
                    color = colors[i % len(colors)]
                    img = Image.new('RGB', (224, 224), color=color)
                    img_path = class_dir / f"image_{i}.jpg"
                    img.save(img_path)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataset_loading_and_preprocessing(self, temp_dataset_dir):
        """Test dataset loading and preprocessing"""
        train_dir = Path(temp_dataset_dir) / 'train'
        
        # Test without transforms
        dataset = ChestXrayDataset(
            data_dir=train_dir,
            transform=None,
            image_size=(224, 224)
        )
        
        assert len(dataset) == 10  # 2 classes × 5 images each
        
        # Test data loading
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert label in [0, 1]
        
        # Test with transforms
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset_with_transforms = ChestXrayDataset(
            data_dir=train_dir,
            transform=transform,
            image_size=(224, 224)
        )
        
        image_transformed, label_transformed = dataset_with_transforms[0]
        assert isinstance(image_transformed, torch.Tensor)
        assert image_transformed.shape == (3, 224, 224)
        assert label_transformed in [0, 1]
        
        # Verify normalization was applied (values should be roughly in [-2, 2] range)
        assert image_transformed.min() >= -3.0
        assert image_transformed.max() <= 3.0
    
    def test_data_loader_integration(self, temp_dataset_dir):
        """Test data loader integration with training"""
        config = TrainingConfig(
            dataset_path=temp_dataset_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            device='cpu'
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        train_loader, val_loader, test_loader = trainer.setup_data_loaders()
        
        # Test train loader
        for batch_idx, (images, labels) in enumerate(train_loader):
            assert images.shape[0] <= config.batch_size
            assert images.shape[1:] == (3, 224, 224)
            assert len(labels) == images.shape[0]
            assert all(label in [0, 1] for label in labels.tolist())
            break
        
        # Test val loader
        for batch_idx, (images, labels) in enumerate(val_loader):
            assert images.shape[0] <= config.batch_size
            assert images.shape[1:] == (3, 224, 224)
            break
        
        # Test test loader
        for batch_idx, (images, labels) in enumerate(test_loader):
            assert images.shape[0] <= config.batch_size
            assert images.shape[1:] == (3, 224, 224)
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])