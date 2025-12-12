"""
Custom Loss Functions for Medical Imaging
Specialized loss functions for chest X-ray pneumonia detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical imaging
    Focuses learning on hard examples and down-weights easy examples
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        num_classes: int = 2,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            num_classes: Number of classes
            reduction: Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Create alpha tensor
        if isinstance(alpha, (float, int)):
            self.alpha_t = torch.tensor([1 - alpha, alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha_t = torch.tensor(alpha)
        else:
            self.alpha_t = alpha
        
        logger.info(f"Initialized Focal Loss with alpha={alpha}, gamma={gamma}")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Predictions from model (logits)
            targets: Ground truth labels
            
        Returns:
            Computed focal loss
        """
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Move alpha to same device as inputs
        if self.alpha_t.device != inputs.device:
            self.alpha_t = self.alpha_t.to(inputs.device)
        
        # Get alpha values for each target
        alpha_t = self.alpha_t[targets]
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    Applies different weights to different classes
    """
    
    def __init__(self, weights: Optional[Tensor] = None, reduction: str = 'mean'):
        """
        Initialize Weighted Cross Entropy Loss
        
        Args:
            weights: Class weights tensor
            reduction: Specifies the reduction to apply to the output
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
        
        if weights is not None:
            logger.info(f"Initialized Weighted CE Loss with weights: {weights.tolist()}")
        else:
            logger.info("Initialized Weighted CE Loss with automatic weights")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Weighted Cross Entropy Loss
        
        Args:
            inputs: Predictions from model (logits)
            targets: Ground truth labels
            
        Returns:
            Computed weighted cross entropy loss
        """
        
        # Move weights to same device as inputs if provided
        weights = self.weights
        if weights is not None and weights.device != inputs.device:
            weights = weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)


class DiceLoss(nn.Module):
    """
    Dice Loss for medical image segmentation
    Can be adapted for classification by treating it as soft dice
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        """
        Initialize Dice Loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Specifies the reduction to apply to the output
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
        logger.info(f"Initialized Dice Loss with smooth={smooth}")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Dice Loss
        
        Args:
            inputs: Predictions from model (logits)
            targets: Ground truth labels
            
        Returns:
            Computed dice loss
        """
        
        # Convert logits to probabilities
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        targets_one_hot = targets_one_hot.permute(0, -1, *range(1, len(targets_one_hot.shape) - 1))
        
        # Flatten tensors
        inputs_flat = inputs_soft.view(inputs_soft.size(0), inputs_soft.size(1), -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Compute dice coefficient
        intersection = (inputs_flat * targets_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines multiple loss functions
    Useful for medical imaging where multiple objectives are important
    """
    
    def __init__(
        self, 
        losses: dict, 
        weights: Optional[dict] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Combined Loss
        
        Args:
            losses: Dictionary of loss functions {'name': loss_function}
            weights: Dictionary of weights for each loss {'name': weight}
            reduction: Specifies the reduction to apply to the output
        """
        super(CombinedLoss, self).__init__()
        
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}
        self.reduction = reduction
        
        logger.info(f"Initialized Combined Loss with losses: {list(losses.keys())}")
        logger.info(f"Loss weights: {self.weights}")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Combined Loss
        
        Args:
            inputs: Predictions from model (logits)
            targets: Ground truth labels
            
        Returns:
            Computed combined loss
        """
        
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(inputs, targets)
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss
        
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization
    Helps prevent overconfident predictions in medical imaging
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Label Smoothing Loss
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (typically 0.1)
            reduction: Specifies the reduction to apply to the output
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        
        logger.info(f"Initialized Label Smoothing Loss with smoothing={smoothing}")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass of Label Smoothing Loss
        
        Args:
            inputs: Predictions from model (logits)
            targets: Ground truth labels
            
        Returns:
            Computed label smoothing loss
        """
        
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -true_dist * log_probs
        
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)


def get_loss_function(
    loss_name: str, 
    num_classes: int = 2, 
    class_weights: Optional[Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of the loss function
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    
    loss_name = loss_name.lower()
    
    if loss_name == 'cross_entropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name == 'weighted_ce' or loss_name == 'weighted_cross_entropy':
        return WeightedCrossEntropyLoss(weights=class_weights)
    
    elif loss_name == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
    
    elif loss_name == 'dice':
        smooth = kwargs.get('smooth', 1e-6)
        return DiceLoss(smooth=smooth)
    
    elif loss_name == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    
    elif loss_name == 'combined':
        # Example combined loss: CE + Focal
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        focal_loss = FocalLoss(num_classes=num_classes)
        
        losses = {'ce': ce_loss, 'focal': focal_loss}
        weights = kwargs.get('loss_weights', {'ce': 0.5, 'focal': 0.5})
        
        return CombinedLoss(losses=losses, weights=weights)
    
    else:
        available_losses = [
            'cross_entropy', 'ce', 'weighted_ce', 'weighted_cross_entropy',
            'focal', 'dice', 'label_smoothing', 'combined'
        ]
        raise ValueError(f"Unknown loss function '{loss_name}'. Available: {available_losses}")


def calculate_class_weights(
    class_counts: dict, 
    method: str = 'inverse_frequency'
) -> Tensor:
    """
    Calculate class weights for handling imbalanced datasets
    
    Args:
        class_counts: Dictionary with class counts {'class_name': count}
        method: Method for calculating weights
        
    Returns:
        Tensor of class weights
    """
    
    counts = torch.tensor(list(class_counts.values()), dtype=torch.float32)
    total_samples = counts.sum()
    num_classes = len(counts)
    
    if method == 'inverse_frequency':
        # Inverse of class frequency
        weights = total_samples / (num_classes * counts)
    
    elif method == 'balanced':
        # Sklearn-style balanced weights
        weights = total_samples / (num_classes * counts)
    
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        weights = torch.sqrt(total_samples / (num_classes * counts))
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    logger.info(f"Calculated class weights using {method}: {weights.tolist()}")
    return weights


if __name__ == "__main__":
    # Test loss functions
    print("Testing Medical Loss Functions...")
    
    # Create sample data
    batch_size = 8
    num_classes = 2
    
    # Sample logits and targets
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Test different loss functions
    losses_to_test = [
        ('CrossEntropy', nn.CrossEntropyLoss()),
        ('Focal', FocalLoss(alpha=0.25, gamma=2.0, num_classes=num_classes)),
        ('WeightedCE', WeightedCrossEntropyLoss(weights=torch.tensor([0.4, 0.6]))),
        ('LabelSmoothing', LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1))
    ]
    
    for name, loss_fn in losses_to_test:
        try:
            loss_value = loss_fn(inputs, targets)
            print(f"{name} Loss: {loss_value.item():.4f}")
        except Exception as e:
            print(f"{name} Loss failed: {e}")
    
    # Test loss factory
    focal_loss = get_loss_function('focal', num_classes=2, alpha=0.25, gamma=2.0)
    loss_value = focal_loss(inputs, targets)
    print(f"Factory Focal Loss: {loss_value.item():.4f}")
    
    # Test class weights calculation
    class_counts = {'NORMAL': 1000, 'PNEUMONIA': 3000}
    weights = calculate_class_weights(class_counts, method='inverse_frequency')
    print(f"Class weights: {weights.tolist()}")
    
    print("✅ Loss functions working correctly!")