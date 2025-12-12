"""
Training Pipeline Package
Medical-grade training infrastructure for chest X-ray pneumonia detection
"""

from .config import TrainingConfig, MedicalTrainingPresets, get_config_for_scenario
from .models import (
    MedicalEfficientNet, MedicalResNet, ModelFactory, 
    MedicalCNNArchitecture, count_parameters, get_model_summary
)
from .trainer import ModelTrainer
from .dataset import ChestXrayDataset, create_data_loaders, analyze_dataset
from .metrics import MedicalMetrics
from .losses import (
    FocalLoss, WeightedCrossEntropyLoss, DiceLoss, 
    CombinedLoss, LabelSmoothingLoss, get_loss_function, calculate_class_weights
)
from .experiment_tracker import ExperimentLogger, setup_mlflow_tracking
from .model_registry import ModelRegistry, AutoModelRegistry, ModelStage

__version__ = "1.0.0"
__author__ = "MLOps Team"

# Package metadata
__all__ = [
    # Configuration
    'TrainingConfig',
    'MedicalTrainingPresets', 
    'get_config_for_scenario',
    
    # Models
    'MedicalEfficientNet',
    'MedicalResNet', 
    'ModelFactory',
    'MedicalCNNArchitecture',
    'count_parameters',
    'get_model_summary',
    
    # Training
    'ModelTrainer',
    
    # Dataset
    'ChestXrayDataset',
    'create_data_loaders',
    'analyze_dataset',
    
    # Metrics
    'MedicalMetrics',
    
    # Losses
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'DiceLoss',
    'CombinedLoss', 
    'LabelSmoothingLoss',
    'get_loss_function',
    'calculate_class_weights',
    
    # MLflow Integration
    'ExperimentLogger',
    'setup_mlflow_tracking',
    'ModelRegistry',
    'AutoModelRegistry',
    'ModelStage'
]

# Package information
PACKAGE_INFO = {
    'name': 'training',
    'version': __version__,
    'description': 'Medical-grade training infrastructure for chest X-ray pneumonia detection',
    'author': __author__,
    'components': {
        'models': 'EfficientNet-B4 and ResNet-50 architectures optimized for medical imaging',
        'trainer': 'Comprehensive training pipeline with medical metrics and best practices',
        'dataset': 'Chest X-ray dataset loader with medical-specific preprocessing',
        'metrics': 'Medical evaluation metrics including sensitivity, specificity, and AUC',
        'losses': 'Specialized loss functions for medical imaging and class imbalance',
        'config': 'Flexible configuration system with medical imaging presets'
    },
    'features': [
        'Transfer learning with pre-trained CNNs',
        'Medical-specific evaluation metrics',
        'Class imbalance handling',
        'Mixed precision training',
        'Comprehensive experiment tracking',
        'Medical imaging best practices',
        'Configurable training strategies'
    ]
}

def get_package_info():
    """Get package information"""
    return PACKAGE_INFO

def print_package_info():
    """Print package information"""
    info = get_package_info()
    
    print(f"\n{info['name'].upper()} PACKAGE v{info['version']}")
    print("=" * 50)
    print(f"Description: {info['description']}")
    print(f"Author: {info['author']}")
    
    print("\nCOMPONENTS:")
    for component, description in info['components'].items():
        print(f"  {component}: {description}")
    
    print("\nFEATURES:")
    for feature in info['features']:
        print(f"  • {feature}")
    
    print("\nUSAGE EXAMPLE:")
    print("""
    from training import TrainingConfig, ModelTrainer, get_config_for_scenario
    
    # Create configuration
    config = get_config_for_scenario('production')
    config.dataset_path = 'data/chest_xray_final'
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Start training
    results = trainer.train()
    """)

if __name__ == "__main__":
    print_package_info()