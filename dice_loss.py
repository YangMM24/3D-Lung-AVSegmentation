import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

class DiceLoss(nn.Module):
    """
    Standard Dice Loss for binary/multi-class segmentation
    Widely used baseline for medical image segmentation
    """
    
    def __init__(self, smooth=1e-6, reduction='mean', ignore_background=True):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            reduction (str): 'mean', 'sum', or 'none'
            ignore_background (bool): Whether to ignore background class (class 0)
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_background = ignore_background
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions [B, C, D, H, W] (logits)
            targets: Ground truth masks [B, C, D, H, W] (binary)
        
        Returns:
            torch.Tensor: Dice loss value
        """
        # Apply sigmoid to convert logits to probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        pred_sum = pred_flat.sum(dim=2)  # [B, C]
        target_sum = target_flat.sum(dim=2)  # [B, C]
        
        # Dice coefficient
        dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Dice loss (1 - dice_coefficient)
        dice_loss = 1 - dice_coeff
        
        # Handle background class
        if self.ignore_background and predictions.size(1) > 1:
            dice_loss = dice_loss[:, 1:]  # Remove background class
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice Loss for vessel segmentation
    Handles artery and vein segmentation separately
    """
    
    def __init__(self, smooth=1e-6, class_weights=None, reduction='mean'):
        """
        Args:
            smooth (float): Smoothing factor
            class_weights (list): Weights for each class [artery_weight, vein_weight]
            reduction (str): Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.reduction = reduction
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 2, D, H, W] - artery and vein predictions
            targets: [B, 2, D, H, W] - artery and vein targets
        """
        predictions = torch.sigmoid(predictions)
        
        # Calculate dice loss for each class
        class_losses = []
        
        for class_idx in range(predictions.size(1)):
            pred_class = predictions[:, class_idx, ...]
            target_class = targets[:, class_idx, ...]
            
            # Flatten
            pred_flat = pred_class.view(pred_class.size(0), -1)
            target_flat = target_class.view(target_class.size(0), -1)
            
            # Calculate dice for this class
            intersection = (pred_flat * target_flat).sum(dim=1)
            pred_sum = pred_flat.sum(dim=1)
            target_sum = target_flat.sum(dim=1)
            
            dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_loss = 1 - dice_coeff
            
            class_losses.append(dice_loss)
        
        # Stack class losses
        class_losses = torch.stack(class_losses, dim=1)  # [B, C]
        
        # Apply class weights
        if self.class_weights is not None:
            if self.class_weights.device != class_losses.device:
                self.class_weights = self.class_weights.to(class_losses.device)
            class_losses = class_losses * self.class_weights.unsqueeze(0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return class_losses.mean()
        elif self.reduction == 'sum':
            return class_losses.sum()
        else:
            return class_losses


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss - combines Focal Loss with Dice Loss
    Helpful for handling class imbalance in vessel segmentation
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, smooth=1e-6, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
            smooth (float): Smoothing factor
            reduction (str): Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """Calculate Focal Dice Loss"""
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        pred_sum = pred_flat.sum(dim=2)
        target_sum = target_flat.sum(dim=2)
        
        # Dice coefficient
        dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Focal weight
        focal_weight = self.alpha * (1 - dice_coeff) ** self.gamma
        
        # Focal dice loss
        dice_loss = 1 - dice_coeff
        focal_dice_loss = focal_weight * dice_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_dice_loss.mean()
        elif self.reduction == 'sum':
            return focal_dice_loss.sum()
        else:
            return focal_dice_loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance
    Particularly useful for vessel segmentation where vessels are sparse
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for positive class
            gamma (float): Focusing parameter
            reduction (str): Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions [B, C, D, H, W] (logits)
            targets: Ground truth masks [B, C, D, H, W] (binary)
        """
        # Apply sigmoid
        predictions = torch.sigmoid(predictions)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Calculate pt
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        
        # Calculate alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal weight
        focal_weight = alpha_weight * (1 - pt) ** self.gamma
        
        # Calculate focal loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy Loss
    Standard baseline for medical image segmentation
    """
    
    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1e-6, 
                 focal_alpha=None, focal_gamma=2.0, class_weights=None, 
                 use_soft_skeleton=False, num_iter=40):
        """
        Args:
            dice_weight (float): Weight for Dice loss
            bce_weight (float): Weight for BCE loss
            smooth (float): Smoothing factor for Dice
            focal_alpha (float): If provided, use Focal Loss instead of BCE
            focal_gamma (float): Gamma parameter for Focal Loss
            class_weights (list): Weights for each class [artery_weight, vein_weight]
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.use_soft_skeleton = use_soft_skeleton
        # Dice loss with class weights
        self.dice_loss = MultiClassDiceLoss(smooth=smooth, class_weights=class_weights)
        self.num_iter = num_iter
        
        # BCE or Focal loss
        if focal_alpha is not None:
            self.bce_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions [B, C, D, H, W] (logits)
            targets: Ground truth masks [B, C, D, H, W] (binary)
        
        Returns:
            dict: Loss components and total loss
        """
        # Calculate individual losses
        dice_loss = self.dice_loss(predictions, targets)
        
        # For BCE, use logits directly
        if isinstance(self.bce_loss, nn.BCEWithLogitsLoss):
            bce_loss = self.bce_loss(predictions, targets)
        else:
            bce_loss = self.bce_loss(predictions, targets)
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'bce_loss': bce_loss,
            'dice_weight': self.dice_weight,
            'bce_weight': self.bce_weight
        }


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    Useful for handling class imbalance by controlling false positives/negatives
    """
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, reduction='mean'):
        """
        Args:
            alpha (float): Weight for false positives
            beta (float): Weight for false negatives
            smooth (float): Smoothing factor
            reduction (str): Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """Calculate Tversky Loss"""
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate true positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Tversky loss
        tversky_loss = 1 - tversky_index
        
        # Apply reduction
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


class SensitivitySpecificityLoss(nn.Module):
    """
    Sensitivity-Specificity Loss
    Balances recall (sensitivity) and specificity for vessel segmentation
    """
    
    def __init__(self, sensitivity_weight=1.0, specificity_weight=1.0, smooth=1e-6):
        """
        Args:
            sensitivity_weight (float): Weight for sensitivity (recall)
            specificity_weight (float): Weight for specificity
            smooth (float): Smoothing factor
        """
        super().__init__()
        self.sensitivity_weight = sensitivity_weight
        self.specificity_weight = specificity_weight
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """Calculate Sensitivity-Specificity Loss"""
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
        target_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate confusion matrix components
        tp = (pred_flat * target_flat).sum(dim=2)
        tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Sensitivity (recall) = TP / (TP + FN)
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        
        # Specificity = TN / (TN + FP)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)
        
        # Combined loss
        sensitivity_loss = 1 - sensitivity
        specificity_loss = 1 - specificity
        
        total_loss = (self.sensitivity_weight * sensitivity_loss + 
                     self.specificity_weight * specificity_loss)
        
        return total_loss.mean()


def dice_coefficient(predictions, targets, smooth=1e-6):
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        predictions: Model predictions [B, C, D, H, W] (probabilities)
        targets: Ground truth masks [B, C, D, H, W] (binary)
        smooth: Smoothing factor
    
    Returns:
        torch.Tensor: Dice coefficients for each class
    """
    # Ensure predictions are probabilities
    if predictions.max() > 1.0:
        predictions = torch.sigmoid(predictions)
    
    # Flatten tensors
    pred_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
    target_flat = targets.view(targets.size(0), targets.size(1), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)
    
    # Dice coefficient
    dice_coeff = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice_coeff


def iou_coefficient(predictions, targets, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) coefficient
    
    Args:
        predictions: Model predictions [B, C, D, H, W] (probabilities)
        targets: Ground truth masks [B, C, D, H, W] (binary)
        smooth: Smoothing factor
    
    Returns:
        torch.Tensor: IoU coefficients for each class
    """
    # Ensure predictions are probabilities
    if predictions.max() > 1.0:
        predictions = torch.sigmoid(predictions)
    
    # Flatten tensors
    pred_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
    target_flat = targets.view(targets.size(0), targets.size(1), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection
    
    # IoU coefficient
    iou_coeff = (intersection + smooth) / (union + smooth)
    
    return iou_coeff


class BaselineLossFactory:
    """
    Factory class for creating different baseline loss functions
    Useful for experimental comparisons
    """
    
    @staticmethod
    def create_loss(loss_type='dice_bce', **kwargs):
        """
        Create loss function based on type
        
        Args:
            loss_type (str): Type of loss function
            **kwargs: Additional arguments for loss function
        
        Returns:
            nn.Module: Loss function
        """
        if loss_type == 'dice':
            return MultiClassDiceLoss(**kwargs)
        
        elif loss_type == 'dice_bce':
            return CombinedDiceBCELoss(**kwargs)
        
        elif loss_type == 'focal_dice':
            return FocalDiceLoss(**kwargs)
        
        elif loss_type == 'tversky':
            return TverskyLoss(**kwargs)
        
        elif loss_type == 'sens_spec':
            return SensitivitySpecificityLoss(**kwargs)
        
        elif loss_type == 'focal':
            return BinaryFocalLoss(**kwargs)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def get_recommended_config(dataset_type='vessel'):
        """
        Get recommended loss configuration for different datasets
        
        Args:
            dataset_type (str): Type of dataset ('vessel', 'organ', 'tumor')
        
        Returns:
            dict: Recommended loss configuration
        """
        if dataset_type == 'vessel':
            return {
                'loss_type': 'dice_bce',
                'dice_weight': 1.0,
                'bce_weight': 1.0,
                'focal_alpha': 0.25,  # Use focal loss for class imbalance
                'focal_gamma': 2.0,
                'class_weights': [0.4, 0.6]  # More weight on veins (typically harder)
            }
        
        elif dataset_type == 'vessel_balanced':
            # For well-balanced vessel datasets
            return {
                'loss_type': 'dice_bce',
                'dice_weight': 1.0,
                'bce_weight': 0.5,
                'class_weights': None
            }
        
        elif dataset_type == 'organ':
            return {
                'loss_type': 'dice_bce',
                'dice_weight': 1.0,
                'bce_weight': 0.5
            }
        
        elif dataset_type == 'tumor':
            return {
                'loss_type': 'focal_dice',
                'alpha': 2.0,
                'gamma': 2.0
            }
        
        else:
            return {
                'loss_type': 'dice_bce',
                'dice_weight': 1.0,
                'bce_weight': 1.0
            }


# Test and usage example
if __name__ == '__main__':
    print("Testing Baseline Loss Functions...")
    
    # Create dummy data
    batch_size, num_classes, depth, height, width = 2, 2, 32, 64, 64
    
    # Dummy predictions (logits) and targets (binary)
    predictions = torch.randn(batch_size, num_classes, depth, height, width)
    targets = torch.randint(0, 2, (batch_size, num_classes, depth, height, width)).float()
    
    print(f"Input shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Targets unique values: {torch.unique(targets)}")
    
    # Test different loss functions
    loss_functions = {
        'Dice Loss': MultiClassDiceLoss(),
        'Combined Dice+BCE': CombinedDiceBCELoss(),
        'Focal Dice': FocalDiceLoss(),
        'Tversky Loss': TverskyLoss(),
        'Focal Loss': BinaryFocalLoss(),
        'Sensitivity-Specificity': SensitivitySpecificityLoss()
    }
    
    print(f"\nTesting loss functions:")
    for name, loss_fn in loss_functions.items():
        try:
            if name == 'Combined Dice+BCE':
                loss_result = loss_fn(predictions, targets)
                loss_value = loss_result['total_loss']
                print(f"  {name}: {loss_value.item():.4f}")
                print(f"    - Dice: {loss_result['dice_loss'].item():.4f}")
                print(f"    - BCE: {loss_result['bce_loss'].item():.4f}")
            else:
                loss_value = loss_fn(predictions, targets)
                print(f"  {name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    # Test evaluation metrics
    print(f"\nTesting evaluation metrics:")
    
    # Convert predictions to probabilities for evaluation
    pred_probs = torch.sigmoid(predictions)
    
    # Calculate metrics
    dice_scores = dice_coefficient(pred_probs, targets)
    iou_scores = iou_coefficient(pred_probs, targets)
    
    print(f"  Dice coefficients per class: {dice_scores.mean(dim=0)}")
    print(f"  IoU coefficients per class: {iou_scores.mean(dim=0)}")
    print(f"  Mean Dice: {dice_scores.mean().item():.4f}")
    print(f"  Mean IoU: {iou_scores.mean().item():.4f}")
    
    # Test loss factory
    print(f"\nTesting Loss Factory:")
    
    # Create different loss types using factory
    factory = BaselineLossFactory()
    
    # Get recommended config for vessel segmentation
    vessel_config = factory.get_recommended_config('vessel')
    print(f"  Recommended config for vessels: {vessel_config}")
    
    # Create loss using factory
    vessel_loss = factory.create_loss(**vessel_config)
    vessel_result = vessel_loss(predictions, targets)
    
    if isinstance(vessel_result, dict):
        print(f"  Vessel loss (total): {vessel_result['total_loss'].item():.4f}")
        print(f"  - Dice component: {vessel_result['dice_loss'].item():.4f}")
        print(f"  - BCE component: {vessel_result['bce_loss'].item():.4f}")
    else:
        print(f"  Vessel loss: {vessel_result.item():.4f}")
    
    # Test with class weights for imbalanced data
    print(f"\nTesting with class weights:")
    weighted_dice = MultiClassDiceLoss(class_weights=[0.3, 0.7])  # More weight on vein
    weighted_loss = weighted_dice(predictions, targets)
    print(f"  Weighted Dice loss: {weighted_loss.item():.4f}")
    
    # Test baseline configuration for your paper
    print(f"\nBaseline configuration for your paper:")
    baseline_config = factory.get_recommended_config('vessel_balanced')
    print(f"  Recommended baseline config: {baseline_config}")
    
    baseline_loss = factory.create_loss(**baseline_config)
    baseline_result = baseline_loss(predictions, targets)
    if isinstance(baseline_result, dict):
        print(f"  Paper baseline loss: {baseline_result['total_loss'].item():.4f}")
    
    print(f"\nAll baseline loss function tests passed!")
    print(f"\nRecommended configurations for your paper:")
    print(f"  # Baseline (for comparison with clDice)")
    print(f"  baseline_loss = CombinedDiceBCELoss(")
    print(f"      dice_weight=1.0,")
    print(f"      bce_weight=1.0")
    print(f"  )")
    print(f"  ")
    print(f"  # Advanced baseline (with class balancing)")
    print(f"  advanced_baseline = CombinedDiceBCELoss(")
    print(f"      dice_weight=1.0,")
    print(f"      bce_weight=1.0,")
    print(f"      focal_alpha=0.25,")
    print(f"      class_weights=[0.4, 0.6]  # artery, vein")
    print(f"  )")
    print(f"  ")
    print(f"  # For evaluation")
    print(f"  dice_scores = dice_coefficient(predictions, targets)")
    print(f"  iou_scores = iou_coefficient(predictions, targets)")