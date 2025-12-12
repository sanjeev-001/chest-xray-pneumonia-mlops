"""
Tests for Hyperparameter Optimization
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import torch
import numpy as np
from unittest.mock import Mock, patch

from training.config import TrainingConfig
from training.hyperparameter_optimizer import (
    HyperparameterOptimizer, 
    OptimizationConfig, 
    trigger_optimization_on_poor_performance
)


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass"""
    
    def test_default_config(self):
        """Test default optimization configuration"""
        config = OptimizationConfig()
        
        assert config.n_trials == 50
        assert config.primary_metric == 'val_f1_score'
        assert config.direction == 'maximize'
        assert config.enable_pruning is True
        assert config.max_epochs_per_trial == 25
        assert config.min_accuracy_threshold == 0.7


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def base_config(self, temp_dir):
        """Create base training configuration"""
        return TrainingConfig(
            dataset_path=temp_dir,  # Use temp dir as dummy dataset path
            epochs=5,
            batch_size=8,
            model_architecture='efficientnet_b4',
            experiment_name='test_optimization'
        )
    
    @pytest.fixture
    def opt_config(self, temp_dir):
        """Create optimization configuration"""
        return OptimizationConfig(
            n_trials=3,  # Very few trials for testing
            max_epochs_per_trial=2,
            output_dir=temp_dir,
            study_name='test_study'
        )
    
    def test_optimizer_initialization(self, base_config, opt_config):
        """Test optimizer initialization"""
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        assert optimizer.base_config == base_config
        assert optimizer.opt_config == opt_config
        assert optimizer.best_trial is None
        assert optimizer.best_score == float('-inf')  # maximize direction
    
    def test_search_space_definition(self, base_config, opt_config):
        """Test hyperparameter search space definition"""
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # Mock optuna trial
        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = 'efficientnet_b4'
        mock_trial.suggest_float.return_value = 1e-4
        mock_trial.suggest_int.return_value = 10
        
        search_space = optimizer.define_search_space(mock_trial)
        
        # Check that all expected parameters are defined
        expected_params = [
            'model_architecture', 'learning_rate', 'batch_size', 'optimizer',
            'weight_decay', 'dropout_rate', 'scheduler', 'loss_function',
            'fine_tuning_strategy', 'augmentation_config'
        ]
        
        for param in expected_params:
            assert param in search_space
        
        # Check augmentation config structure
        assert isinstance(search_space['augmentation_config'], dict)
        assert 'rotation_range' in search_space['augmentation_config']
    
    def test_trial_config_creation(self, base_config, opt_config):
        """Test creation of trial configuration"""
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # Mock parameters
        params = {
            'model_architecture': 'efficientnet_b4',
            'learning_rate': 1e-4,
            'batch_size': 16,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'scheduler': 'cosine',
            'loss_function': 'cross_entropy',
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'fine_tuning_strategy': 'gradual',
            'freeze_epochs': 10,
            'augmentation_config': {
                'rotation_range': 10,
                'brightness_range': 0.1,
                'contrast_range': 0.1,
                'horizontal_flip': True,
                'vertical_flip': False,
                'zoom_range': 0.1,
                'shear_range': 5,
                'fill_mode': 'reflect'
            }
        }
        
        trial_config = optimizer._create_trial_config(params)
        
        # Check that parameters are correctly applied
        assert trial_config.model_architecture == 'efficientnet_b4'
        assert trial_config.learning_rate == 1e-4
        assert trial_config.batch_size == 16
        assert trial_config.optimizer == 'adamw'
        assert trial_config.epochs == opt_config.max_epochs_per_trial
        
        # Check that base config values are preserved
        assert trial_config.dataset_path == base_config.dataset_path
        assert trial_config.num_classes == base_config.num_classes
    
    def test_best_trial_update(self, base_config, opt_config):
        """Test best trial tracking"""
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        
        params = {'learning_rate': 1e-4}
        results = {'val_accuracy': 0.85, 'val_f1_score': 0.82}
        score = 0.82
        
        # Update best trial
        optimizer._update_best_trial(mock_trial, score, params, results)
        
        assert optimizer.best_score == score
        assert optimizer.best_trial is not None
        assert optimizer.best_trial['trial_number'] == 1
        assert optimizer.best_trial['score'] == score
        assert optimizer.best_trial['params'] == params
        assert optimizer.best_trial['results'] == results
    
    def test_get_best_config(self, base_config, opt_config):
        """Test getting best configuration"""
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # No best trial yet
        assert optimizer.get_best_config() is None
        
        # Set a best trial
        optimizer.best_trial = {
            'params': {
                'model_architecture': 'efficientnet_b4',
                'learning_rate': 5e-4,
                'batch_size': 32,
                'optimizer': 'adamw',
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'scheduler': 'cosine',
                'loss_function': 'cross_entropy',
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'fine_tuning_strategy': 'gradual',
                'freeze_epochs': 10,
                'augmentation_config': {}
            }
        }
        
        best_config = optimizer.get_best_config()
        assert best_config is not None
        assert best_config.learning_rate == 5e-4
        assert best_config.batch_size == 32
    
    def test_production_config_creation(self, base_config, opt_config):
        """Test production configuration creation"""
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # No best trial
        assert optimizer.create_production_config() is None
        
        # Set best trial
        optimizer.best_trial = {
            'params': {
                'model_architecture': 'efficientnet_b4',
                'learning_rate': 1e-4,
                'batch_size': 16,
                'optimizer': 'adamw',
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'scheduler': 'cosine',
                'loss_function': 'cross_entropy',
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'fine_tuning_strategy': 'gradual',
                'freeze_epochs': 10,
                'augmentation_config': {}
            }
        }
        
        prod_config = optimizer.create_production_config(extended_epochs=150)
        
        assert prod_config is not None
        assert prod_config.epochs == 150
        assert prod_config.early_stopping_patience == 30  # 150 // 5
        assert 'production' in prod_config.experiment_name


class TestAutoOptimization:
    """Test automatic optimization triggering"""
    
    def test_no_optimization_needed(self):
        """Test when current accuracy is above threshold"""
        base_config = TrainingConfig(epochs=5, batch_size=8)
        
        result = trigger_optimization_on_poor_performance(
            current_accuracy=0.90,
            threshold=0.85,
            base_config=base_config
        )
        
        assert result is None
    
    @patch('training.hyperparameter_optimizer.HyperparameterOptimizer')
    def test_optimization_triggered(self, mock_optimizer_class):
        """Test when optimization is triggered"""
        # Mock optimizer and study
        mock_optimizer = Mock()
        mock_study = Mock()
        mock_optimizer.optimize.return_value = mock_study
        mock_optimizer.create_production_config.return_value = TrainingConfig()
        mock_optimizer_class.return_value = mock_optimizer
        
        base_config = TrainingConfig(epochs=5, batch_size=8)
        
        result = trigger_optimization_on_poor_performance(
            current_accuracy=0.80,  # Below threshold
            threshold=0.85,
            base_config=base_config
        )
        
        # Check that optimizer was created and used
        mock_optimizer_class.assert_called_once()
        mock_optimizer.optimize.assert_called_once()
        mock_optimizer.create_production_config.assert_called_once()
        
        assert result is not None


class TestOptimizationIntegration:
    """Integration tests for optimization system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_config_serialization(self, temp_dir):
        """Test configuration saving and loading"""
        config = OptimizationConfig(
            n_trials=10,
            study_name='test_study',
            output_dir=temp_dir
        )
        
        # Test that output directory is created
        optimizer = HyperparameterOptimizer(
            base_config=TrainingConfig(),
            optimization_config=config
        )
        
        assert Path(temp_dir).exists()
    
    def test_best_params_saving(self, temp_dir):
        """Test saving best parameters"""
        base_config = TrainingConfig()
        opt_config = OptimizationConfig(
            study_name='test_study',
            output_dir=temp_dir,
            save_best_params=True
        )
        
        optimizer = HyperparameterOptimizer(
            base_config=base_config,
            optimization_config=opt_config
        )
        
        # Set best trial
        optimizer.best_trial = {
            'trial_number': 5,
            'score': 0.85,
            'params': {'learning_rate': 1e-4},
            'results': {'val_accuracy': 0.85},
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Save best params
        optimizer._save_best_params()
        
        # Check file was created
        expected_file = Path(temp_dir) / f"{opt_config.study_name}_best_params.json"
        assert expected_file.exists()
        
        # Check content
        with open(expected_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['trial_number'] == 5
        assert saved_data['score'] == 0.85
        assert saved_data['params']['learning_rate'] == 1e-4


if __name__ == "__main__":
    pytest.main([__file__])