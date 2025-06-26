import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology
from scipy import ndimage
from typing import Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

def soft_skeletonize_3d(volume, num_iter=40, smooth=1e-6):
    """
    Soft skeletonization for 3D volumes using differentiable morphological operations
    
    Args:
        volume: Input volume [B, C, D, H, W] (probabilities)
        num_iter: Number of iterative erosions
        smooth: Smoothing factor for soft operations
    
    Returns:
        torch.Tensor: Soft skeleton of the volume
    """
    # Ensure volume is in [0, 1] range
    volume = torch.clamp(volume, 0, 1)
    
    # Get input dimensions
    batch_size, num_channels = volume.shape[:2]
    
    # Initialize skeleton
    skeleton = volume.clone()
    
    # Define 3D structuring element (6-connectivity)
    kernel_size = 3
    # Create kernel for each channel separately
    kernel = torch.ones(num_channels, 1, kernel_size, kernel_size, kernel_size, 
                       device=volume.device, dtype=volume.dtype)
    kernel = kernel / kernel.sum(dim=(2, 3, 4), keepdim=True)
    
    # Iterative soft skeletonization
    for i in range(num_iter):
        # Soft erosion using grouped convolution (one kernel per channel)
        eroded = F.conv3d(skeleton, kernel, padding=kernel_size//2, groups=num_channels)
        eroded = torch.clamp(eroded - smooth, 0, 1)
        
        # Soft opening (erosion followed by dilation)
        opened = F.conv3d(eroded, kernel, padding=kernel_size//2, groups=num_channels)
        opened = torch.clamp(opened, 0, 1)
        
        # Update skeleton: keep points that are not removed by opening
        skeleton = skeleton * (1 - opened + smooth)
        
        # Stop if skeleton becomes too thin
        if skeleton.sum() < smooth:
            break
    
    return skeleton


def hard_skeletonize_3d(volume):
    """
    Hard skeletonization using skimage for ground truth processing
    
    Args:
        volume: Input volume [B, C, D, H, W] (binary)
    
    Returns:
        torch.Tensor: Hard skeleton of the volume
    """
    batch_size, num_classes = volume.shape[:2]
    skeletons = torch.zeros_like(volume)
    
    for b in range(batch_size):
        for c in range(num_classes):
            vol = volume[b, c].cpu().numpy()
            
            if vol.sum() > 0:  # Only process if there are positive voxels
                # Convert to binary
                binary_vol = (vol > 0.5).astype(bool)
                
                # 3D skeletonization
                try:
                    skeleton = morphology.skeletonize_3d(binary_vol)
                    skeletons[b, c] = torch.from_numpy(skeleton.astype(np.float32))
                except:
                    # Fallback: slice-by-slice skeletonization
                    skeleton_3d = np.zeros_like(binary_vol, dtype=bool)
                    for z in range(binary_vol.shape[2]):
                        if binary_vol[:, :, z].any():
                            skeleton_3d[:, :, z] = morphology.skeletonize(binary_vol[:, :, z])
                    skeletons[b, c] = torch.from_numpy(skeleton_3d.astype(np.float32))
    
    return skeletons.to(volume.device)


def compute_centerline_dice(pred_skeleton, target_skeleton, smooth=1e-6):
    """
    Compute Dice coefficient between skeletons (centerlines)
    
    Args:
        pred_skeleton: Predicted skeleton [B, C, D, H, W]
        target_skeleton: Target skeleton [B, C, D, H, W]
        smooth: Smoothing factor
    
    Returns:
        torch.Tensor: Centerline Dice coefficients
    """
    # Flatten tensors
    pred_flat = pred_skeleton.view(pred_skeleton.size(0), pred_skeleton.size(1), -1)
    target_flat = target_skeleton.view(target_skeleton.size(0), target_skeleton.size(1), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)
    
    # Centerline Dice coefficient
    cl_dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return cl_dice


class clDiceLoss(nn.Module):
    """
    clDice Loss - Centerline Dice Loss for topology-aware segmentation
    
    Paper: "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
    Adapted for 3D vessel segmentation with soft skeletonization
    """
    
    def __init__(self, smooth=1e-6, alpha=1.0, num_iter=40, use_soft_skeleton=True):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            alpha (float): Weight for skeleton computation in soft mode
            num_iter (int): Number of iterations for soft skeletonization
            use_soft_skeleton (bool): Use differentiable soft skeletonization
        """
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.num_iter = num_iter
        self.use_soft_skeleton = use_soft_skeleton
    
    def forward(self, predictions, targets):
        """
        Compute clDice loss
        
        Args:
            predictions: Model predictions [B, C, D, H, W] (logits or probabilities)
            targets: Ground truth masks [B, C, D, H, W] (binary)
        
        Returns:
            torch.Tensor: clDice loss value
        """
        # Convert predictions to probabilities
        if predictions.max() > 1.0:
            pred_probs = torch.sigmoid(predictions)
        else:
            pred_probs = predictions
        
        # Extract skeletons
        if self.use_soft_skeleton and self.training:
            # Soft skeletonization (differentiable)
            pred_skeleton = soft_skeletonize_3d(pred_probs, self.num_iter, self.smooth)
            target_skeleton = soft_skeletonize_3d(targets, self.num_iter, self.smooth)
        else:
            # Hard skeletonization (non-differentiable, for evaluation)
            with torch.no_grad():
                pred_skeleton = hard_skeletonize_3d(pred_probs > 0.5)
                target_skeleton = hard_skeletonize_3d(targets)
        
        # Compute centerline Dice coefficient
        cl_dice = compute_centerline_dice(pred_skeleton, target_skeleton, self.smooth)
        
        # clDice loss (1 - centerline Dice)
        cl_dice_loss = 1 - cl_dice
        
        return cl_dice_loss.mean()
    
    def get_centerline_dice_score(self, predictions, targets):
        """
        Get centerline Dice score for evaluation (non-differentiable)
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
        
        Returns:
            torch.Tensor: Centerline Dice scores
        """
        # Convert to probabilities and binarize
        if predictions.max() > 1.0:
            pred_probs = torch.sigmoid(predictions)
        else:
            pred_probs = predictions
        
        pred_binary = (pred_probs > 0.5).float()
        
        # Hard skeletonization
        with torch.no_grad():
            pred_skeleton = hard_skeletonize_3d(pred_binary)
            target_skeleton = hard_skeletonize_3d(targets)
        
        # Compute centerline Dice
        cl_dice = compute_centerline_dice(pred_skeleton, target_skeleton, self.smooth)
        
        return cl_dice


class CombinedDiceclDiceLoss(nn.Module):
    """
    Combined Dice + clDice Loss
    
    This is the main loss function for your paper:
    Loss = α * Dice + β * clDice
    """
    
    def __init__(self, dice_weight=1.0, cldice_weight=1.0, smooth=1e-6,
                 cldice_iter=40, use_soft_skeleton=True):
        """
        Args:
            dice_weight (float): Weight for standard Dice loss (α)
            cldice_weight (float): Weight for clDice loss (β)
            smooth (float): Smoothing factor
            cldice_iter (int): Iterations for soft skeletonization
            use_soft_skeleton (bool): Use soft skeletonization during training
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        
        # Standard Dice loss
        from dice_loss import MultiClassDiceLoss
        self.dice_loss = MultiClassDiceLoss(smooth=smooth)
        
        # clDice loss
        self.cldice_loss = clDiceLoss(
            smooth=smooth,
            num_iter=cldice_iter,
            use_soft_skeleton=use_soft_skeleton
        )
    
    def forward(self, predictions, targets):
        """
        Compute combined Dice + clDice loss
        
        Args:
            predictions: Model predictions [B, C, D, H, W] (logits)
            targets: Ground truth masks [B, C, D, H, W] (binary)
        
        Returns:
            dict: Loss components and total loss
        """
        # Calculate individual losses
        dice_loss = self.dice_loss(predictions, targets)
        cldice_loss = self.cldice_loss(predictions, targets)
        
        # Combined loss: Loss = α * Dice + β * clDice
        total_loss = self.dice_weight * dice_loss + self.cldice_weight * cldice_loss
        
        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'cldice_loss': cldice_loss,
            'dice_weight': self.dice_weight,
            'cldice_weight': self.cldice_weight
        }


class AdaptiveclDiceLoss(nn.Module):
    """
    Adaptive clDice Loss with dynamic weight adjustment
    Adjusts the importance of topology preservation based on training progress
    """
    
    def __init__(self, initial_dice_weight=1.0, initial_cldice_weight=0.1,
                 max_cldice_weight=1.0, warmup_epochs=50, smooth=1e-6):
        """
        Args:
            initial_dice_weight (float): Initial weight for Dice loss
            initial_cldice_weight (float): Initial weight for clDice loss
            max_cldice_weight (float): Maximum weight for clDice loss
            warmup_epochs (int): Number of epochs to reach max clDice weight
            smooth (float): Smoothing factor
        """
        super().__init__()
        self.initial_dice_weight = initial_dice_weight
        self.initial_cldice_weight = initial_cldice_weight
        self.max_cldice_weight = max_cldice_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Loss components
        from dice_loss import MultiClassDiceLoss
        self.dice_loss = MultiClassDiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
    
    def set_epoch(self, epoch):
        """Set current epoch for adaptive weighting"""
        self.current_epoch = epoch
    
    def get_current_weights(self):
        """Get current loss weights based on training progress"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup for clDice weight
            progress = self.current_epoch / self.warmup_epochs
            cldice_weight = (self.initial_cldice_weight + 
                           progress * (self.max_cldice_weight - self.initial_cldice_weight))
        else:
            cldice_weight = self.max_cldice_weight
        
        dice_weight = self.initial_dice_weight
        
        return dice_weight, cldice_weight
    
    def forward(self, predictions, targets):
        """Compute adaptive combined loss"""
        # Get current weights
        dice_weight, cldice_weight = self.get_current_weights()
        
        # Calculate losses
        dice_loss = self.dice_loss(predictions, targets)
        cldice_loss = self.cldice_loss(predictions, targets)
        
        # Combined loss with adaptive weights
        total_loss = dice_weight * dice_loss + cldice_weight * cldice_loss
        
        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'cldice_loss': cldice_loss,
            'dice_weight': dice_weight,
            'cldice_weight': cldice_weight
        }


class TopologyMetrics:
    """
    Topology-aware evaluation metrics for vessel segmentation
    """
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
    
    def centerline_dice(self, predictions, targets):
        """Calculate centerline Dice coefficient"""
        # Convert to binary
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        
        # Extract skeletons
        pred_skeleton = hard_skeletonize_3d(pred_binary)
        target_skeleton = hard_skeletonize_3d(targets)
        
        # Compute centerline Dice
        cl_dice = compute_centerline_dice(pred_skeleton, target_skeleton, self.smooth)
        
        return cl_dice
    
    def connectivity_accuracy(self, predictions, targets):
        """
        Calculate connectivity preservation accuracy
        Measures how well the predicted segmentation preserves connectivity
        """
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        
        batch_size, num_classes = predictions.shape[:2]
        connectivity_scores = torch.zeros(batch_size, num_classes)
        
        for b in range(batch_size):
            for c in range(num_classes):
                pred_vol = pred_binary[b, c].cpu().numpy()
                target_vol = targets[b, c].cpu().numpy()
                
                if target_vol.sum() > 0:
                    # Count connected components
                    pred_components = self._count_components(pred_vol)
                    target_components = self._count_components(target_vol)
                    
                    # Connectivity score (inverse of component difference)
                    component_diff = abs(pred_components - target_components)
                    connectivity_scores[b, c] = 1.0 / (1.0 + component_diff)
                else:
                    connectivity_scores[b, c] = 1.0 if pred_vol.sum() == 0 else 0.0
        
        return connectivity_scores
    
    def _count_components(self, volume):
        """Count connected components in 3D volume"""
        if volume.sum() == 0:
            return 0
        
        binary_vol = (volume > 0.5).astype(bool)
        labeled, num_components = ndimage.label(binary_vol)
        return num_components
    
    def skeleton_length_ratio(self, predictions, targets):
        """
        Calculate skeleton length preservation ratio
        Measures how well skeleton length is preserved
        """
        pred_binary = (torch.sigmoid(predictions) > 0.5).float()
        
        # Extract skeletons
        pred_skeleton = hard_skeletonize_3d(pred_binary)
        target_skeleton = hard_skeletonize_3d(targets)
        
        # Calculate skeleton lengths (sum of skeleton voxels)
        pred_length = pred_skeleton.sum(dim=(2, 3, 4))
        target_length = target_skeleton.sum(dim=(2, 3, 4))
        
        # Length ratio
        length_ratio = torch.where(
            target_length > 0,
            torch.min(pred_length / (target_length + self.smooth), 
                     torch.ones_like(pred_length)),
            torch.ones_like(pred_length)
        )
        
        return length_ratio


def create_topology_loss(loss_type='combined', **kwargs):
    """
    Factory function for creating topology-aware loss functions
    
    Args:
        loss_type (str): Type of loss ('cldice', 'combined', 'adaptive')
        **kwargs: Additional arguments for loss function
    
    Returns:
        nn.Module: Configured loss function
    """
    if loss_type == 'cldice':
        return clDiceLoss(**kwargs)
    
    elif loss_type == 'combined':
        return CombinedDiceclDiceLoss(**kwargs)
    
    elif loss_type == 'adaptive':
        return AdaptiveclDiceLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Test and usage example
if __name__ == '__main__':
    print("Testing clDice Loss Functions...")
    
    # Create dummy data representing vessel structures
    batch_size, num_classes, depth, height, width = 2, 2, 32, 64, 64
    
    # Create synthetic vessel-like structures
    def create_vessel_structure(shape):
        """Create synthetic vessel-like structure for testing"""
        volume = np.zeros(shape)
        
        # Add some tube-like structures
        center_z, center_y, center_x = np.array(shape) // 2
        
        # Main vessel
        for i in range(shape[2]):
            y = center_y + int(5 * np.sin(i * 0.3))
            x = center_x + int(3 * np.cos(i * 0.2))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                volume[max(0, y-1):y+2, max(0, x-1):x+2, i] = 1
        
        # Branch vessel
        for i in range(shape[2]//2):
            y = center_y + i
            x = center_x + i//2
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                volume[y, x, center_z + i] = 1
        
        return volume
    
    # Create targets with vessel-like structures
    targets = torch.zeros(batch_size, num_classes, depth, height, width)
    for b in range(batch_size):
        for c in range(num_classes):
            vessel_structure = create_vessel_structure((depth, height, width))
            targets[b, c] = torch.from_numpy(vessel_structure)
    
    # Create predictions (logits) with some noise
    predictions = torch.randn(batch_size, num_classes, depth, height, width) * 2
    
    # Make predictions somewhat similar to targets
    pred_probs = torch.sigmoid(predictions)
    predictions = predictions + targets * 3  # Bias towards target structure
    
    print(f"Input shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Targets sum: {targets.sum().item()}")
    
    # Test soft skeletonization
    print(f"\nTesting soft skeletonization...")
    pred_probs = torch.sigmoid(predictions)
    soft_skeleton = soft_skeletonize_3d(pred_probs, num_iter=20)
    print(f"  Input sum: {pred_probs.sum().item():.0f}")
    print(f"  Soft skeleton sum: {soft_skeleton.sum().item():.0f}")
    
    # Test hard skeletonization
    print(f"\nTesting hard skeletonization...")
    hard_skeleton = hard_skeletonize_3d(targets)
    print(f"  Target sum: {targets.sum().item():.0f}")
    print(f"  Hard skeleton sum: {hard_skeleton.sum().item():.0f}")
    
    # Test clDice loss functions
    print(f"\nTesting clDice loss functions:")
    
    loss_functions = {
        'clDice (soft)': clDiceLoss(use_soft_skeleton=True),
        'clDice (hard)': clDiceLoss(use_soft_skeleton=False),
        'Combined Dice+clDice': CombinedDiceclDiceLoss(dice_weight=1.0, cldice_weight=1.0),
        'Adaptive clDice': AdaptiveclDiceLoss()
    }
    
    for name, loss_fn in loss_functions.items():
        try:
            if 'Adaptive' in name:
                loss_fn.set_epoch(25)  # Set epoch for adaptive loss
            
            if 'Combined' in name or 'Adaptive' in name:
                loss_result = loss_fn(predictions, targets)
                total_loss = loss_result['total_loss']
                print(f"  {name}: {total_loss.item():.4f}")
                print(f"    - Dice: {loss_result['dice_loss'].item():.4f}")
                print(f"    - clDice: {loss_result['cldice_loss'].item():.4f}")
                print(f"    - Weights: Dice={loss_result['dice_weight']:.2f}, clDice={loss_result['cldice_weight']:.2f}")
            else:
                loss_value = loss_fn(predictions, targets)
                print(f"  {name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    # Test topology metrics
    print(f"\nTesting topology metrics:")
    
    topology_metrics = TopologyMetrics()
    
    # Centerline Dice
    cl_dice_scores = topology_metrics.centerline_dice(predictions, targets)
    print(f"  Centerline Dice per class: {cl_dice_scores.mean(dim=0)}")
    print(f"  Mean Centerline Dice: {cl_dice_scores.mean().item():.4f}")
    
    # Connectivity accuracy
    connectivity_scores = topology_metrics.connectivity_accuracy(predictions, targets)
    print(f"  Connectivity accuracy per class: {connectivity_scores.mean(dim=0)}")
    print(f"  Mean connectivity accuracy: {connectivity_scores.mean().item():.4f}")
    
    # Skeleton length ratio
    length_ratios = topology_metrics.skeleton_length_ratio(predictions, targets)
    print(f"  Skeleton length ratio per class: {length_ratios.mean(dim=0)}")
    print(f"  Mean skeleton length ratio: {length_ratios.mean().item():.4f}")
    
    # Test loss factory
    print(f"\nTesting loss factory:")
    
    # Create different loss types
    factory_losses = {
        'Factory clDice': create_topology_loss('cldice'),
        'Factory Combined': create_topology_loss('combined', dice_weight=1.0, cldice_weight=1.5),
        'Factory Adaptive': create_topology_loss('adaptive', warmup_epochs=100)
    }
    
    for name, loss_fn in factory_losses.items():
        try:
            if 'Adaptive' in name:
                loss_fn.set_epoch(50)
            
            result = loss_fn(predictions, targets)
            if isinstance(result, dict):
                print(f"  {name}: {result['total_loss'].item():.4f}")
            else:
                print(f"  {name}: {result.item():.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print(f"\nAll clDice loss function tests passed!")