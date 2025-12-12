"""
Medical CNN Models for Chest X-Ray Pneumonia Detection
Implements EfficientNet-B4 with ResNet-50 backup for medical imaging
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MedicalCNNArchitecture:
    """Medical-grade CNN architectures for chest X-ray classification"""
    
    ARCHITECTURES = {
        'efficientnet_b4': {
            'model_class': efficientnet_b4,
            'weights': EfficientNet_B4_Weights.IMAGENET1K_V1,
            'input_size': (224, 224),
            'features_dim': 1792,
            'description': 'EfficientNet-B4 - Optimal for medical imaging',
            'recommended': True
        },
        'resnet50': {
            'model_class': resnet50,
            'weights': ResNet50_Weights.IMAGENET1K_V2,
            'input_size': (224, 224),
            'features_dim': 2048,
            'description': 'ResNet-50 - Reliable backup architecture',
            'recommended': False
        }
    }
    
    @classmethod
    def get_available_architectures(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available architectures"""
        return cls.ARCHITECTURES
    
    @classmethod
    def get_recommended_architecture(cls) -> str:
        """Get the recommended architecture for medical imaging"""
        for name, config in cls.ARCHITECTURES.items():
            if config.get('recommended', False):
                return name
        return 'efficientnet_b4'  # Default fallback


class MedicalEfficientNet(nn.Module):
    """
    EfficientNet-B4 adapted for medical chest X-ray classification
    Optimized for NORMAL vs PNEUMONIA binary classification
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3, pretrained: bool = True):
        super(MedicalEfficientNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained EfficientNet-B4
        if pretrained:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b4(weights=weights)
            logger.info("Loaded pre-trained EfficientNet-B4 weights from ImageNet")
        else:
            self.backbone = efficientnet_b4(weights=None)
            logger.info("Initialized EfficientNet-B4 without pre-trained weights")
        
        # Get the number of features from the backbone
        backbone_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with medical-specific head
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Medical-specific classification head - simplified to match checkpoint
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)  # Direct connection: 512 -> 2 classes
        )
        
        # Initialize the new layers
        self._initialize_classifier_weights()
        
        logger.info(f"Created MedicalEfficientNet with {num_classes} classes and {dropout_rate} dropout")
    
    def _initialize_classifier_weights(self):
        """Initialize the weights of the new classifier layers"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Apply medical-specific classifier
        output = self.classifier(features)
        
        return output
    
    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction part of the model"""
        return self.backbone
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            logger.info("Frozen EfficientNet-B4 backbone for fine-tuning")
        else:
            logger.info("Unfrozen EfficientNet-B4 backbone for full training")


class MedicalResNet(nn.Module):
    """
    ResNet-50 adapted for medical chest X-ray classification
    Backup architecture for comparison and reliability
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3, pretrained: bool = True):
        super(MedicalResNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
            logger.info("Loaded pre-trained ResNet-50 weights from ImageNet")
        else:
            self.backbone = resnet50(weights=None)
            logger.info("Initialized ResNet-50 without pre-trained weights")
        
        # Get the number of features from the backbone
        backbone_features = self.backbone.fc.in_features
        
        # Replace the final layer with medical-specific head
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Medical-specific classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize the new layers
        self._initialize_classifier_weights()
        
        logger.info(f"Created MedicalResNet with {num_classes} classes and {dropout_rate} dropout")
    
    def _initialize_classifier_weights(self):
        """Initialize the weights of the new classifier layers"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Apply medical-specific classifier
        output = self.classifier(features)
        
        return output
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone for fine-tuning"""
        # Freeze all layers except the final classifier
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze the final layer
                param.requires_grad = not freeze
        
        if freeze:
            logger.info("Frozen ResNet-50 backbone for fine-tuning")
        else:
            logger.info("Unfrozen ResNet-50 backbone for full training")


class ModelFactory:
    """Factory class for creating medical CNN models"""
    
    @staticmethod
    def create_model(
        architecture: str = 'efficientnet_b4',
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True
    ) -> nn.Module:
        """
        Create a medical CNN model
        
        Args:
            architecture: Model architecture ('efficientnet_b4' or 'resnet50')
            num_classes: Number of output classes (default: 2 for NORMAL/PNEUMONIA)
            dropout_rate: Dropout rate for regularization
            pretrained: Whether to use pre-trained weights
            
        Returns:
            PyTorch model ready for training
        """
        
        if architecture not in MedicalCNNArchitecture.ARCHITECTURES:
            available = list(MedicalCNNArchitecture.ARCHITECTURES.keys())
            raise ValueError(f"Architecture '{architecture}' not supported. Available: {available}")
        
        if architecture == 'efficientnet_b4':
            model = MedicalEfficientNet(
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                pretrained=pretrained
            )
        elif architecture == 'resnet50':
            model = MedicalResNet(
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                pretrained=pretrained
            )
        else:
            raise ValueError(f"Architecture '{architecture}' not implemented")
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Created {architecture} model:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model
    
    @staticmethod
    def get_model_info(architecture: str) -> Dict[str, Any]:
        """Get information about a specific architecture"""
        if architecture not in MedicalCNNArchitecture.ARCHITECTURES:
            available = list(MedicalCNNArchitecture.ARCHITECTURES.keys())
            raise ValueError(f"Architecture '{architecture}' not supported. Available: {available}")
        
        return MedicalCNNArchitecture.ARCHITECTURES[architecture]


# Utility functions for model management
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }


def get_model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)) -> str:
    """Get a summary of the model architecture"""
    try:
        from torchsummary import summary
        import io
        import sys
        
        # Capture the summary output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        summary(model, input_size)
        
        sys.stdout = old_stdout
        summary_str = buffer.getvalue()
        
        return summary_str
    except ImportError:
        logger.warning("torchsummary not available. Install with: pip install torchsummary")
        param_info = count_parameters(model)
        return f"Model Parameters: {param_info}"


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Medical CNN Models...")
    
    # Test EfficientNet-B4 (primary)
    print("\\n1. Testing EfficientNet-B4:")
    model_eff = ModelFactory.create_model('efficientnet_b4')
    print(f"   Parameters: {count_parameters(model_eff)}")
    
    # Test ResNet-50 (backup)
    print("\\n2. Testing ResNet-50:")
    model_res = ModelFactory.create_model('resnet50')
    print(f"   Parameters: {count_parameters(model_res)}")
    
    # Test forward pass
    print("\\n3. Testing forward pass:")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output_eff = model_eff(dummy_input)
        output_res = model_res(dummy_input)
        
        print(f"   EfficientNet output shape: {output_eff.shape}")
        print(f"   ResNet output shape: {output_res.shape}")
    
    print("\\n✅ All models created successfully!")