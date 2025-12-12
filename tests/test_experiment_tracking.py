"""
Unit Tests for Experiment Tracking and Model Registry
Tests for MLflow integration and model management
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

from training.experiment_tracker import ExperimentLogger
from training.model_registry import ModelRegistry, AutoModelRegistry
from training.config import TrainingConfig
from training.models import ModelFactory


class TestExperimentLogger:
    """Test ExperimentLogger class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow for testing"""
        with patch('training.experiment_tracker.mlflow') as mock_mlflow:
            mock_mlflow.set_tracking_uri = Mock()
            mock_mlflow.set_experiment = Mock()
            mock_mlflow.start_run = Mock()
            mock_mlflow.end_run = Mock()
            mock_mlflow.log_param = Mock()
            mock_mlflow.log_metric = Mock()
            mock_mlflow.log_params = Mock()
            mock_mlflow.log_metrics = Mock()
            mock_mlflow.log_artifact = Mock()
            mock_mlflow.active_run = Mock()
            mock_mlflow.active_run.return_value = Mock(info=Mock(run_id='test_run_id'))
            yield mock_mlflow
    
    def test_logger_initialization(self, temp_dir, mock_mlflow):
        """Test experiment logger initialization"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db',
            artifact_location=temp_dir
        )
        
        assert logger.experiment_name == 'test_experiment'
        assert logger.tracking_uri == f'sqlite:///{temp_dir}/test.db'
        assert logger.artifact_location == temp_dir
        
        # Verify MLflow was configured
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()
    
    def test_start_end_run(self, temp_dir, mock_mlflow):
        """Test starting and ending runs"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        # Test start run
        logger.start_run(run_name='test_run')
        mock_mlflow.start_run.assert_called_once()
        
        # Test end run
        logger.end_run()
        mock_mlflow.end_run.assert_called_once()
    
    def test_context_manager(self, temp_dir, mock_mlflow):
        """Test using logger as context manager"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        with logger.start_run(run_name='test_run'):
            # Should start run
            mock_mlflow.start_run.assert_called_once()
        
        # Should end run when exiting context
        mock_mlflow.end_run.assert_called_once()
    
    def test_log_params(self, temp_dir, mock_mlflow):
        """Test logging parameters"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        # Test single parameter
        logger.log_param('learning_rate', 0.001)
        mock_mlflow.log_param.assert_called_with('learning_rate', 0.001)
        
        # Test multiple parameters
        params = {'batch_size': 32, 'epochs': 50}
        logger.log_params(params)
        mock_mlflow.log_params.assert_called_with(params)
    
    def test_log_metrics(self, temp_dir, mock_mlflow):
        """Test logging metrics"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        # Test single metric
        logger.log_metric('accuracy', 0.95)
        mock_mlflow.log_metric.assert_called_with('accuracy', 0.95, step=None)
        
        # Test metric with step
        logger.log_metric('loss', 0.1, step=10)
        mock_mlflow.log_metric.assert_called_with('loss', 0.1, step=10)
        
        # Test multiple metrics
        metrics = {'precision': 0.92, 'recall': 0.88}
        logger.log_metrics(metrics, step=5)
        mock_mlflow.log_metrics.assert_called_with(metrics, step=5)
    
    def test_log_config(self, temp_dir, mock_mlflow):
        """Test logging training configuration"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=50
        )
        
        logger.log_config(config)
        
        # Should log parameters from config
        mock_mlflow.log_params.assert_called_once()
        call_args = mock_mlflow.log_params.call_args[0][0]
        
        assert 'learning_rate' in call_args
        assert 'batch_size' in call_args
        assert 'epochs' in call_args
    
    def test_log_medical_metrics(self, temp_dir, mock_mlflow):
        """Test logging medical-specific metrics"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        medical_metrics = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'auc_roc': 0.94,
            'sensitivity': 0.88,
            'specificity': 0.96,
            'confusion_matrix': np.array([[50, 5], [3, 42]]),
            'classification_report': {'0': {'precision': 0.94}, '1': {'precision': 0.89}}
        }
        
        logger.log_medical_metrics(medical_metrics, prefix='val_')
        
        # Should log all metrics with prefix
        mock_mlflow.log_metrics.assert_called()
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        
        assert 'val_accuracy' in call_args
        assert 'val_precision' in call_args
        assert 'val_sensitivity' in call_args
    
    @patch('training.experiment_tracker.plt')
    def test_log_training_plots(self, mock_plt, temp_dir, mock_mlflow):
        """Test logging training plots"""
        logger = ExperimentLogger(
            experiment_name='test_experiment',
            tracking_uri=f'sqlite:///{temp_dir}/test.db'
        )
        
        history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.2, 0.9, 0.7],
            'train_acc': [0.6, 0.7, 0.8],
            'val_acc': [0.5, 0.65, 0.75]
        }
        
        logger.log_training_plots(history, save_dir=temp_dir)
        
        # Should create and save plots
        mock_plt.figure.assert_called()
        mock_plt.savefig.assert_called()
        mock_mlflow.log_artifact.assert_called()


class TestModelRegistry:
    """Test ModelRegistry class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_mlflow_client(self):
        """Mock MLflow client for testing"""
        with patch('training.model_registry.MlflowClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock client methods
            mock_client.create_registered_model = Mock()
            mock_client.create_model_version = Mock()
            mock_client.get_registered_model = Mock()
            mock_client.search_registered_models = Mock()
            mock_client.transition_model_version_stage = Mock()
            mock_client.get_latest_versions = Mock()
            
            yield mock_client
    
    def test_registry_initialization(self, temp_dir, mock_mlflow_client):
        """Test model registry initialization"""
        registry = ModelRegistry(tracking_uri=f'sqlite:///{temp_dir}/test.db')
        
        assert registry.tracking_uri == f'sqlite:///{temp_dir}/test.db'
        assert registry.client is not None
    
    def test_register_model(self, temp_dir, mock_mlflow_client):
        """Test model registration"""
        registry = ModelRegistry(tracking_uri=f'sqlite:///{temp_dir}/test.db')
        
        # Mock successful registration
        mock_version = Mock()
        mock_version.version = '1'
        mock_mlflow_client.create_model_version.return_value = mock_version
        
        model_uri = 'runs:/test_run_id/model'
        metadata = {
            'architecture': 'efficientnet_b4',
            'accuracy': 0.95,
            'dataset': 'chest_xray'
        }
        
        version = registry.register_model(
            model_name='test_model',
            model_uri=model_uri,
            description='Test model',
            metadata=metadata
        )
        
        assert version.version == '1'
        mock_mlflow_client.create_model_version.assert_called_once()
    
    def test_get_model(self, temp_dir, mock_mlflow_client):
        """Test getting model from registry"""
        registry = ModelRegistry(tracking_uri=f'sqlite:///{temp_dir}/test.db')
        
        # Mock model version
        mock_version = Mock()
        mock_version.source = 'runs:/test_run_id/model'
        mock_mlflow_client.get_latest_versions.return_value = [mock_version]
        
        with patch('training.model_registry.mlflow.pytorch.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            model = registry.get_model('test_model', stage='Production')
            
            assert model == mock_model
            mock_load.assert_called_once_with(mock_version.source)
    
    def test_list_models(self, temp_dir, mock_mlflow_client):
        """Test listing models"""
        registry = ModelRegistry(tracking_uri=f'sqlite:///{temp_dir}/test.db')
        
        # Mock registered models
        mock_model1 = Mock()
        mock_model1.name = 'model1'
        mock_model1.description = 'First model'
        
        mock_model2 = Mock()
        mock_model2.name = 'model2'
        mock_model2.description = 'Second model'
        
        mock_mlflow_client.search_registered_models.return_value = [mock_model1, mock_model2]
        
        models = registry.list_models()
        
        assert len(models) == 2
        assert models[0]['name'] == 'model1'
        assert models[1]['name'] == 'model2'
    
    def test_promote_model(self, temp_dir, mock_mlflow_client):
        """Test model promotion"""
        registry = ModelRegistry(tracking_uri=f'sqlite:///{temp_dir}/test.db')
        
        success = registry.promote_model('test_model', version='1', stage='Production')
        
        assert success is True
        mock_mlflow_client.transition_model_version_stage.assert_called_once_with(
            name='test_model',
            version='1',
            stage='Production'
        )


class TestAutoModelRegistry:
    """Test AutoModelRegistry class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_registry(self):
        """Mock ModelRegistry for testing"""
        mock_registry = Mock()
        mock_registry.register_model = Mock()
        mock_registry.promote_model = Mock()
        mock_registry.get_model = Mock()
        return mock_registry
    
    def test_auto_registry_initialization(self, mock_model_registry):
        """Test auto registry initialization"""
        promotion_criteria = {
            'accuracy': 0.85,
            'auc_roc': 0.80,
            'sensitivity': 0.80
        }
        
        auto_registry = AutoModelRegistry(
            model_registry=mock_model_registry,
            model_name='test_model',
            promotion_criteria=promotion_criteria
        )
        
        assert auto_registry.model_registry == mock_model_registry
        assert auto_registry.model_name == 'test_model'
        assert auto_registry.promotion_criteria == promotion_criteria
    
    def test_meets_promotion_criteria(self, mock_model_registry):
        """Test promotion criteria evaluation"""
        promotion_criteria = {
            'accuracy': 0.85,
            'auc_roc': 0.80,
            'sensitivity': 0.80
        }
        
        auto_registry = AutoModelRegistry(
            model_registry=mock_model_registry,
            model_name='test_model',
            promotion_criteria=promotion_criteria
        )
        
        # Test metrics that meet criteria
        good_metrics = {
            'accuracy': 0.90,
            'auc_roc': 0.85,
            'sensitivity': 0.82,
            'specificity': 0.88
        }
        
        assert auto_registry._meets_promotion_criteria(good_metrics) is True
        
        # Test metrics that don't meet criteria
        poor_metrics = {
            'accuracy': 0.80,  # Below threshold
            'auc_roc': 0.85,
            'sensitivity': 0.82
        }
        
        assert auto_registry._meets_promotion_criteria(poor_metrics) is False
    
    def test_register_and_evaluate(self, mock_model_registry):
        """Test automatic registration and evaluation"""
        promotion_criteria = {
            'accuracy': 0.85,
            'auc_roc': 0.80
        }
        
        auto_registry = AutoModelRegistry(
            model_registry=mock_model_registry,
            model_name='test_model',
            promotion_criteria=promotion_criteria
        )
        
        # Mock successful registration
        mock_version = Mock()
        mock_version.version = '1'
        mock_model_registry.register_model.return_value = mock_version
        mock_model_registry.promote_model.return_value = True
        
        # Test with good metrics
        good_metrics = {
            'accuracy': 0.90,
            'auc_roc': 0.85
        }
        
        version, promoted = auto_registry.register_and_evaluate(
            model_uri='runs:/test_run_id/model',
            validation_metrics=good_metrics,
            description='Test model',
            auto_promote=True
        )
        
        assert version.version == '1'
        assert promoted is True
        
        # Verify registration and promotion were called
        mock_model_registry.register_model.assert_called_once()
        mock_model_registry.promote_model.assert_called_once()
    
    def test_no_auto_promotion(self, mock_model_registry):
        """Test registration without auto-promotion"""
        auto_registry = AutoModelRegistry(
            model_registry=mock_model_registry,
            model_name='test_model',
            promotion_criteria={'accuracy': 0.85}
        )
        
        mock_version = Mock()
        mock_version.version = '1'
        mock_model_registry.register_model.return_value = mock_version
        
        # Test with auto_promote=False
        version, promoted = auto_registry.register_and_evaluate(
            model_uri='runs:/test_run_id/model',
            validation_metrics={'accuracy': 0.90},
            auto_promote=False
        )
        
        assert version.version == '1'
        assert promoted is False
        
        # Should register but not promote
        mock_model_registry.register_model.assert_called_once()
        mock_model_registry.promote_model.assert_not_called()


class TestIntegration:
    """Integration tests for experiment tracking and model registry"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow from experiment to model registry"""
        # This test would require actual MLflow setup, so we'll mock it
        with patch('training.experiment_tracker.mlflow') as mock_mlflow, \
             patch('training.model_registry.MlflowClient') as mock_client_class:
            
            # Setup mocks
            mock_mlflow.active_run.return_value = Mock(info=Mock(run_id='test_run_id'))
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Create experiment logger
            logger = ExperimentLogger(
                experiment_name='integration_test',
                tracking_uri=f'sqlite:///{temp_dir}/test.db'
            )
            
            # Create model registry
            registry = ModelRegistry(tracking_uri=f'sqlite:///{temp_dir}/test.db')
            
            # Simulate experiment workflow
            with logger.start_run(run_name='test_run'):
                # Log configuration
                config = TrainingConfig(learning_rate=0.001, batch_size=32)
                logger.log_config(config)
                
                # Log metrics
                metrics = {
                    'accuracy': 0.95,
                    'precision': 0.92,
                    'recall': 0.88,
                    'f1_score': 0.90
                }
                logger.log_metrics(metrics)
                
                # Register model
                mock_version = Mock()
                mock_version.version = '1'
                mock_client.create_model_version.return_value = mock_version
                
                version = registry.register_model(
                    model_name='integration_test_model',
                    model_uri='runs:/test_run_id/model',
                    description='Integration test model',
                    metadata={'accuracy': 0.95}
                )
                
                assert version.version == '1'
            
            # Verify all components were called
            mock_mlflow.start_run.assert_called()
            mock_mlflow.end_run.assert_called()
            mock_client.create_model_version.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])