#!/usr/bin/env python3
"""
Training Components Test Runner
Quick validation of key training components
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_config():
    """Test training configuration"""
    print("Testing TrainingConfig...")
    try:
        from training.config import TrainingConfig, MedicalTrainingPresets
        
        # Test default config
        config = TrainingConfig()
        assert config.model_architecture == 'efficientnet_b4'
        assert config.num_classes == 2
        assert config.class_names == ['NORMAL', 'PNEUMONIA']
        
        # Test preset configs
        quick_config = MedicalTrainingPresets.get_quick_test_config()
        assert quick_config.epochs == 5
        
        print("✅ TrainingConfig tests passed")
        return True
    except Exception as e:
        print(f"❌ TrainingConfig tests failed: {e}")
        traceback.print_exc()
        return False

def test_models():
    """Test model factory"""
    print("Testing ModelFactory...")
    try:
        from training.models import ModelFactory
        import torch
        
        # Test model creation
        model = ModelFactory.create_model(
            architecture='efficientnet_b4',
            num_classes=2,
            pretrained=False  # Avoid downloading weights
        )
        
        # Test forward pass (use batch size > 1 for batch norm)
        model.eval()  # Set to eval mode to avoid batch norm issues
        dummy_input = torch.randn(2, 3, 224, 224)  # batch_size=2
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (2, 2)
        print("✅ ModelFactory tests passed")
        return True
    except Exception as e:
        print(f"❌ ModelFactory tests failed: {e}")
        traceback.print_exc()
        return False

def test_metrics():
    """Test medical metrics"""
    print("Testing MedicalMetrics...")
    try:
        from training.metrics import MedicalMetrics
        import numpy as np
        
        # Test perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        
        metrics_calc = MedicalMetrics(['NORMAL', 'PNEUMONIA'])
        metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_prob)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert 'confusion_matrix' in metrics
        
        print("✅ MedicalMetrics tests passed")
        return True
    except Exception as e:
        print(f"❌ MedicalMetrics tests failed: {e}")
        traceback.print_exc()
        return False

def test_losses():
    """Test loss functions"""
    print("Testing Loss Functions...")
    try:
        from training.losses import FocalLoss, WeightedCrossEntropyLoss
        import torch
        
        # Test focal loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, num_classes=2)
        predictions = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        
        loss = focal_loss(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
        
        # Test weighted cross entropy
        import torch
        weights = torch.tensor([1.0, 2.0])
        weighted_loss = WeightedCrossEntropyLoss(weights)
        loss2 = weighted_loss(predictions, targets)
        assert isinstance(loss2, torch.Tensor)
        assert loss2.item() >= 0.0
        
        print("✅ Loss Functions tests passed")
        return True
    except Exception as e:
        print(f"❌ Loss Functions tests failed: {e}")
        traceback.print_exc()
        return False

def test_trainer():
    """Test model trainer initialization"""
    print("Testing ModelTrainer...")
    try:
        from training.trainer import ModelTrainer
        from training.config import TrainingConfig
        
        # Test trainer initialization
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            device='cpu',
            use_augmentation=False
        )
        
        trainer = ModelTrainer(config, use_mlflow=False)
        assert trainer.config == config
        assert trainer.device.type == 'cpu'
        
        # Test model setup
        model = trainer.setup_model()
        assert model is not None
        
        # Test optimizer setup
        optimizer = trainer.setup_optimizer()
        assert optimizer is not None
        
        # Test criterion setup
        criterion = trainer.setup_criterion()
        assert criterion is not None
        
        print("✅ ModelTrainer tests passed")
        return True
    except Exception as e:
        print(f"❌ ModelTrainer tests failed: {e}")
        traceback.print_exc()
        return False

def test_hyperparameter_optimizer():
    """Test hyperparameter optimizer"""
    print("Testing HyperparameterOptimizer...")
    try:
        from training.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
        from training.config import TrainingConfig
        
        # Test optimizer initialization
        base_config = TrainingConfig(epochs=2, batch_size=4)
        opt_config = OptimizationConfig(n_trials=1, max_epochs_per_trial=1)
        
        optimizer = HyperparameterOptimizer(base_config, opt_config)
        assert optimizer.base_config == base_config
        assert optimizer.opt_config == opt_config
        
        print("✅ HyperparameterOptimizer tests passed")
        return True
    except Exception as e:
        print(f"❌ HyperparameterOptimizer tests failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TRAINING COMPONENTS TEST SUITE")
    print("="*60)
    
    tests = [
        test_config,
        test_models,
        test_metrics,
        test_losses,
        test_trainer,
        test_hyperparameter_optimizer
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Training components are working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)