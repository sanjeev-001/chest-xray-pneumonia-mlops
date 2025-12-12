"""
Training Configuration Management
Handles all training hyperparameters and configurations for medical imaging
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration for medical chest X-ray models
    Optimized for EfficientNet-B4 with medical imaging best practices
    """
    
    # Model Configuration
    model_architecture: str = 'efficientnet_b4'
    num_classes: int = 2
    dropout_rate: float = 0.3
    pretrained: bool = True
    
    # Training Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    
    # Learning Rate Scheduling
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 50,  # For cosine annealing
        'eta_min': 1e-6,
        'step_size': 15,  # For step scheduler
        'gamma': 0.1,
        'patience': 5,  # For plateau scheduler
        'factor': 0.5
    })
    
    # Loss Function Configuration
    loss_function: str = 'cross_entropy'  # 'cross_entropy', 'focal', 'weighted_ce'
    class_weights: Optional[List[float]] = None  # Auto-calculated if None
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Data Configuration
    dataset_path: str = 'data/chest_xray_final'
    image_size: Tuple[int, int] = (380, 380)  # EfficientNet-B4 optimal size
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data Augmentation
    use_augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        'rotation_range': 10,
        'brightness_range': 0.1,
        'contrast_range': 0.1,
        'horizontal_flip': True,
        'vertical_flip': False,  # Not recommended for chest X-rays
        'zoom_range': 0.1,
        'shear_range': 5,
        'fill_mode': 'reflect'
    })
    
    # Training Strategy
    fine_tuning_strategy: str = 'gradual'  # 'gradual', 'full', 'frozen'
    freeze_epochs: int = 10  # Epochs to keep backbone frozen
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Validation and Evaluation
    validation_frequency: int = 1  # Validate every N epochs
    save_best_model: bool = True
    save_checkpoint_frequency: int = 5
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 
        'sensitivity', 'specificity', 'confusion_matrix'
    ])
    
    # Medical-Specific Configuration
    medical_metrics: bool = True
    class_names: List[str] = field(default_factory=lambda: ['NORMAL', 'PNEUMONIA'])
    confidence_threshold: float = 0.5
    
    # Output Configuration
    output_dir: str = 'outputs/training'
    experiment_name: Optional[str] = None
    model_save_path: str = 'models'
    log_level: str = 'INFO'
    
    # Hardware Configuration
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_config()
        self._setup_paths()
        self._auto_configure()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model architecture
        from .models import MedicalCNNArchitecture
        available_archs = list(MedicalCNNArchitecture.ARCHITECTURES.keys())
        if self.model_architecture not in available_archs:
            raise ValueError(f"Invalid architecture '{self.model_architecture}'. Available: {available_archs}")
        
        # Validate hyperparameters
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        # Validate paths
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
        
        # Validate class configuration
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"Number of class names ({len(self.class_names)}) must match num_classes ({self.num_classes})")
    
    def _setup_paths(self):
        """Setup output directories"""
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, self.model_save_path)).mkdir(parents=True, exist_ok=True)
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.model_architecture}_{timestamp}"
    
    def _auto_configure(self):
        """Auto-configure based on system and data"""
        # Auto-detect device
        if self.device == 'auto':
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("Auto-detected CUDA device")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("Auto-detected MPS device")
            else:
                self.device = 'cpu'
                logger.info("Using CPU device")
        
        # Adjust batch size for device memory
        if self.device == 'cuda':
            import torch
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory < 8 and self.batch_size > 16:
                self.batch_size = 16
                logger.info(f"Reduced batch size to {self.batch_size} for GPU memory constraints")
        
        # Set scheduler T_max to match epochs
        if self.scheduler == 'cosine':
            self.scheduler_params['T_max'] = self.epochs
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        config = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer.lower() == 'sgd':
            config['momentum'] = self.momentum
        
        return config
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        return self.scheduler_params.copy()
    
    def get_augmentation_transforms(self):
        """Get data augmentation transforms"""
        if not self.use_augmentation:
            return None
        
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            transforms = []
            
            # Rotation
            if self.augmentation_config.get('rotation_range', 0) > 0:
                transforms.append(A.Rotate(
                    limit=self.augmentation_config['rotation_range'],
                    border_mode=0,
                    p=0.5
                ))
            
            # Brightness and contrast
            if self.augmentation_config.get('brightness_range', 0) > 0 or self.augmentation_config.get('contrast_range', 0) > 0:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=self.augmentation_config.get('brightness_range', 0),
                    contrast_limit=self.augmentation_config.get('contrast_range', 0),
                    p=0.5
                ))
            
            # Horizontal flip
            if self.augmentation_config.get('horizontal_flip', False):
                transforms.append(A.HorizontalFlip(p=0.5))
            
            # Zoom (ShiftScaleRotate for zoom effect)
            if self.augmentation_config.get('zoom_range', 0) > 0:
                transforms.append(A.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=self.augmentation_config['zoom_range'],
                    rotate_limit=0,
                    border_mode=0,
                    p=0.5
                ))
            
            # Normalization and tensor conversion
            transforms.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            return A.Compose(transforms)
            
        except ImportError:
            logger.warning("Albumentations not available. Using basic transforms.")
            return None
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, type(None))):
                config_dict[key] = value
            elif isinstance(value, dict):
                config_dict[key] = value
            elif isinstance(value, tuple):
                config_dict[key] = list(value)
            else:
                config_dict[key] = str(value)
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class MedicalTrainingPresets:
    """Pre-defined training configurations for different scenarios"""
    
    @staticmethod
    def get_quick_test_config() -> TrainingConfig:
        """Configuration for quick testing and debugging"""
        return TrainingConfig(
            epochs=5,
            batch_size=8,
            learning_rate=1e-3,
            early_stopping_patience=3,
            validation_frequency=1,
            experiment_name="quick_test"
        )
    
    @staticmethod
    def get_development_config() -> TrainingConfig:
        """Configuration for development and experimentation"""
        return TrainingConfig(
            epochs=25,
            batch_size=16,
            learning_rate=5e-4,
            early_stopping_patience=7,
            freeze_epochs=5,
            experiment_name="development"
        )
    
    @staticmethod
    def get_production_config() -> TrainingConfig:
        """Configuration for production-quality training"""
        return TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=1e-4,
            early_stopping_patience=15,
            freeze_epochs=15,
            gradient_clip_norm=1.0,
            mixed_precision=True,
            experiment_name="production"
        )
    
    @staticmethod
    def get_fine_tuning_config() -> TrainingConfig:
        """Configuration for fine-tuning pre-trained models"""
        return TrainingConfig(
            epochs=30,
            batch_size=24,
            learning_rate=5e-5,
            fine_tuning_strategy='gradual',
            freeze_epochs=10,
            scheduler='cosine',
            experiment_name="fine_tuning"
        )
    
    @staticmethod
    def get_high_accuracy_config() -> TrainingConfig:
        """Configuration optimized for maximum accuracy"""
        return TrainingConfig(
            epochs=150,
            batch_size=16,  # Smaller batch for better gradients
            learning_rate=5e-5,
            dropout_rate=0.4,  # Higher dropout for regularization
            early_stopping_patience=20,
            freeze_epochs=20,
            mixed_precision=True,
            gradient_clip_norm=0.5,
            experiment_name="high_accuracy"
        )


def get_config_for_scenario(scenario: str) -> TrainingConfig:
    """Get configuration for a specific training scenario"""
    scenarios = {
        'quick_test': MedicalTrainingPresets.get_quick_test_config,
        'development': MedicalTrainingPresets.get_development_config,
        'production': MedicalTrainingPresets.get_production_config,
        'fine_tuning': MedicalTrainingPresets.get_fine_tuning_config,
        'high_accuracy': MedicalTrainingPresets.get_high_accuracy_config
    }
    
    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {available}")
    
    return scenarios[scenario]()


if __name__ == "__main__":
    # Test configuration creation
    print("Testing Training Configuration...")
    
    # Test default configuration
    config = TrainingConfig()
    print(f"Default config created: {config.experiment_name}")
    
    # Test preset configurations
    for scenario in ['quick_test', 'development', 'production']:
        preset_config = get_config_for_scenario(scenario)
        print(f"{scenario} config: {preset_config.epochs} epochs, batch_size={preset_config.batch_size}")
    
    # Test configuration saving/loading
    config.save_config("test_config.json")
    loaded_config = TrainingConfig.load_config("test_config.json")
    print(f"Config saved and loaded successfully: {loaded_config.experiment_name}")
    
    # Clean up
    os.remove("test_config.json")
    
    print("✅ Configuration system working correctly!")