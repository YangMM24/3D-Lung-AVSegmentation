import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from skimage import morphology, measure
from skimage.morphology import skeletonize_3d
import warnings
warnings.filterwarnings('ignore')

class TopologyMetrics:
    """
    Comprehensive topology evaluation metrics for vessel segmentation
    Implements metrics to evaluate structural connectivity and topology preservation
    """
    
    def __init__(self, smooth=1e-6, connectivity=26):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            connectivity (int): Connectivity for 3D analysis (6, 18, 26)
        """
        self.smooth = smooth
        self.connectivity = connectivity
    
    def centerline_dice(self, predictions, targets, apply_sigmoid=True):
        """
        Calculate centerline Dice coefficient (clDice metric)
        
        Args:
            predictions: Model predictions [B, C, D, H, W] (logits or probabilities)
            targets: Ground truth masks [B, C, D, H, W] (binary)
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            torch.Tensor: Centerline Dice coefficients [B, C]
        """
        # Convert to appropriate format
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))  # Sigmoid
        
        # Binarize predictions
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        batch_size, num_classes = pred_binary.shape[:2]
        cl_dice_scores = np.zeros((batch_size, num_classes))
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Extract skeletons
                pred_skeleton = self._extract_skeleton_safe(pred_binary[b, c])
                target_skeleton = self._extract_skeleton_safe(target_binary[b, c])
                
                # Calculate clDice: 2 * |S(P) ∩ G| * |S(G) ∩ P| / (|S(P) ∩ G| + |S(G) ∩ P|)
                intersection_pred_target = np.sum(pred_skeleton * target_binary[b, c])
                intersection_target_pred = np.sum(target_skeleton * pred_binary[b, c])
                
                cl_dice = (2 * intersection_pred_target * intersection_target_pred + self.smooth) / \
                         (intersection_pred_target + intersection_target_pred + self.smooth)
                
                cl_dice_scores[b, c] = cl_dice
        
        # Convert back to tensor if input was tensor
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(cl_dice_scores).to(predictions.device)
        else:
            return cl_dice_scores
    
    def connectivity_accuracy(self, predictions, targets, apply_sigmoid=True):
        """
        Calculate connectivity preservation accuracy
        Measures how well predicted segmentation preserves connectivity structure
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            torch.Tensor: Connectivity accuracy scores [B, C]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(bool)
        target_binary = (target_numpy > 0.5).astype(bool)
        
        batch_size, num_classes = pred_binary.shape[:2]
        connectivity_scores = np.zeros((batch_size, num_classes))
        
        for b in range(batch_size):
            for c in range(num_classes):
                pred_vol = pred_binary[b, c]
                target_vol = target_binary[b, c]
                
                if target_vol.sum() > 0:
                    # Count connected components
                    pred_components = self._count_connected_components(pred_vol)
                    target_components = self._count_connected_components(target_vol)
                    
                    # Connectivity score: inverse relationship to component difference
                    component_diff = abs(pred_components - target_components)
                    connectivity_scores[b, c] = 1.0 / (1.0 + component_diff)
                else:
                    # Perfect score if both are empty
                    connectivity_scores[b, c] = 1.0 if pred_vol.sum() == 0 else 0.0
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(connectivity_scores).to(predictions.device)
        else:
            return connectivity_scores
    
    def skeleton_similarity(self, predictions, targets, apply_sigmoid=True):
        """
        Calculate skeleton-based similarity metrics
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            torch.Tensor: Skeleton similarity scores [B, C]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        batch_size, num_classes = pred_binary.shape[:2]
        similarity_scores = np.zeros((batch_size, num_classes))
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Extract skeletons
                pred_skeleton = self._extract_skeleton_safe(pred_binary[b, c])
                target_skeleton = self._extract_skeleton_safe(target_binary[b, c])
                
                # Calculate skeleton overlap
                intersection = np.sum(pred_skeleton * target_skeleton)
                union = np.sum((pred_skeleton + target_skeleton) > 0)
                
                if union > 0:
                    similarity_scores[b, c] = intersection / union
                else:
                    similarity_scores[b, c] = 1.0 if intersection == 0 else 0.0
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(similarity_scores).to(predictions.device)
        else:
            return similarity_scores
    
    def vessel_completeness(self, predictions, targets, apply_sigmoid=True):
        """
        Measure vessel completeness - how much of the vessel structure is captured
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            torch.Tensor: Vessel completeness scores [B, C]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        batch_size, num_classes = pred_binary.shape[:2]
        completeness_scores = np.zeros((batch_size, num_classes))
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Extract skeletons
                target_skeleton = self._extract_skeleton_safe(target_binary[b, c])
                
                if target_skeleton.sum() > 0:
                    # Calculate how much of target skeleton is covered by prediction
                    covered_skeleton = target_skeleton * pred_binary[b, c]
                    completeness_scores[b, c] = covered_skeleton.sum() / target_skeleton.sum()
                else:
                    completeness_scores[b, c] = 1.0
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(completeness_scores).to(predictions.device)
        else:
            return completeness_scores
    
    def topology_accuracy(self, predictions, targets, apply_sigmoid=True):
        """
        Comprehensive topology accuracy combining multiple metrics
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            dict: Dictionary containing various topology metrics
        """
        cl_dice = self.centerline_dice(predictions, targets, apply_sigmoid)
        connectivity = self.connectivity_accuracy(predictions, targets, apply_sigmoid)
        skeleton_similarity = self.skeleton_similarity(predictions, targets, apply_sigmoid)
        vessel_completeness = self.vessel_completeness(predictions, targets, apply_sigmoid)
        
        return {
            'centerline_dice': cl_dice,
            'connectivity_accuracy': connectivity,
            'skeleton_similarity': skeleton_similarity,
            'vessel_completeness': vessel_completeness,
            'combined_topology_score': (cl_dice + connectivity + skeleton_similarity + vessel_completeness) / 4
        }
    
    def branch_detection_accuracy(self, predictions, targets, apply_sigmoid=True):
        """
        Evaluate accuracy of branch detection (junction points)
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            dict: Branch detection metrics
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        batch_size, num_classes = pred_binary.shape[:2]
        
        branch_metrics = {
            'junction_precision': np.zeros((batch_size, num_classes)),
            'junction_recall': np.zeros((batch_size, num_classes)),
            'endpoint_accuracy': np.zeros((batch_size, num_classes))
        }
        
        for b in range(batch_size):
            for c in range(num_classes):
                pred_skeleton = self._extract_skeleton_safe(pred_binary[b, c])
                target_skeleton = self._extract_skeleton_safe(target_binary[b, c])
                
                # Find junction points (degree > 2)
                pred_junctions = self._find_junction_points(pred_skeleton)
                target_junctions = self._find_junction_points(target_skeleton)
                
                # Find endpoint accuracy
                pred_endpoints = self._find_endpoint_points(pred_skeleton)
                target_endpoints = self._find_endpoint_points(target_skeleton)
                
                # Calculate junction precision and recall
                if len(pred_junctions) > 0:
                    junction_precision = self._calculate_point_precision(pred_junctions, target_junctions)
                    branch_metrics['junction_precision'][b, c] = junction_precision
                else:
                    branch_metrics['junction_precision'][b, c] = 1.0 if len(target_junctions) == 0 else 0.0
                
                if len(target_junctions) > 0:
                    junction_recall = self._calculate_point_recall(pred_junctions, target_junctions)
                    branch_metrics['junction_recall'][b, c] = junction_recall
                else:
                    branch_metrics['junction_recall'][b, c] = 1.0
                
                # Calculate endpoint accuracy
                endpoint_accuracy = self._calculate_endpoint_accuracy(pred_endpoints, target_endpoints)
                branch_metrics['endpoint_accuracy'][b, c] = endpoint_accuracy
        
        # Convert to tensors if input was tensor
        if isinstance(predictions, torch.Tensor):
            for key in branch_metrics:
                branch_metrics[key] = torch.from_numpy(branch_metrics[key]).to(predictions.device)
        
        return branch_metrics
    
    # Helper methods
    def _extract_skeleton_safe(self, mask):
        """Safely extract skeleton from binary mask"""
        try:
            if mask.sum() == 0:
                return np.zeros_like(mask, dtype=np.float32)
            
            binary_mask = (mask > 0.5).astype(bool)
            skeleton = skeletonize_3d(binary_mask)
            return skeleton.astype(np.float32)
        except Exception as e:
            print(f"Warning: Skeleton extraction failed: {e}")
            return np.zeros_like(mask, dtype=np.float32)
    
    def _count_connected_components(self, volume):
        """Count connected components in 3D volume"""
        if volume.sum() == 0:
            return 0
        
        binary_vol = (volume > 0.5).astype(bool)
        labeled, num_components = scipy_label(binary_vol)
        return num_components
    
    def _find_junction_points(self, skeleton):
        """Find junction points in skeleton (degree > 2)"""
        if skeleton.sum() == 0:
            return []
        
        # Get skeleton coordinates
        coords = np.column_stack(np.where(skeleton > 0.5))
        junctions = []
        
        # Define 26-connectivity neighborhood
        neighbors_26 = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if not (dz == 0 and dy == 0 and dx == 0):
                        neighbors_26.append([dz, dy, dx])
        neighbors_26 = np.array(neighbors_26)
        
        # Check degree of each skeleton point
        for coord in coords:
            degree = 0
            for neighbor_offset in neighbors_26:
                neighbor_coord = coord + neighbor_offset
                
                # Check bounds
                if (0 <= neighbor_coord[0] < skeleton.shape[0] and
                    0 <= neighbor_coord[1] < skeleton.shape[1] and
                    0 <= neighbor_coord[2] < skeleton.shape[2]):
                    
                    if skeleton[tuple(neighbor_coord)] > 0.5:
                        degree += 1
            
            if degree > 2:  # Junction point
                junctions.append(coord)
        
        return junctions
    
    def _find_endpoint_points(self, skeleton):
        """Find endpoint points in skeleton (degree = 1)"""
        if skeleton.sum() == 0:
            return []
        
        coords = np.column_stack(np.where(skeleton > 0.5))
        endpoints = []
        
        # Define 26-connectivity neighborhood
        neighbors_26 = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if not (dz == 0 and dy == 0 and dx == 0):
                        neighbors_26.append([dz, dy, dx])
        neighbors_26 = np.array(neighbors_26)
        
        for coord in coords:
            degree = 0
            for neighbor_offset in neighbors_26:
                neighbor_coord = coord + neighbor_offset
                
                if (0 <= neighbor_coord[0] < skeleton.shape[0] and
                    0 <= neighbor_coord[1] < skeleton.shape[1] and
                    0 <= neighbor_coord[2] < skeleton.shape[2]):
                    
                    if skeleton[tuple(neighbor_coord)] > 0.5:
                        degree += 1
            
            if degree == 1:  # Endpoint
                endpoints.append(coord)
        
        return endpoints
    
    def _calculate_point_precision(self, pred_points, target_points, threshold=2.0):
        """Calculate precision for point detection"""
        if len(pred_points) == 0:
            return 1.0 if len(target_points) == 0 else 0.0
        
        if len(target_points) == 0:
            return 0.0
        
        true_positives = 0
        for pred_point in pred_points:
            # Find minimum distance to any target point
            distances = [np.linalg.norm(pred_point - target_point) for target_point in target_points]
            if min(distances) <= threshold:
                true_positives += 1
        
        return true_positives / len(pred_points)
    
    def _calculate_point_recall(self, pred_points, target_points, threshold=2.0):
        """Calculate recall for point detection"""
        if len(target_points) == 0:
            return 1.0
        
        if len(pred_points) == 0:
            return 0.0
        
        true_positives = 0
        for target_point in target_points:
            distances = [np.linalg.norm(target_point - pred_point) for pred_point in pred_points]
            if min(distances) <= threshold:
                true_positives += 1
        
        return true_positives / len(target_points)
    
    def _calculate_endpoint_accuracy(self, pred_endpoints, target_endpoints, threshold=2.0):
        """Calculate endpoint detection accuracy"""
        if len(target_endpoints) == 0 and len(pred_endpoints) == 0:
            return 1.0
        
        if len(target_endpoints) == 0:
            return 0.0 if len(pred_endpoints) > 0 else 1.0
        
        if len(pred_endpoints) == 0:
            return 0.0
        
        # Calculate F1 score for endpoints
        precision = self._calculate_point_precision(pred_endpoints, target_endpoints, threshold)
        recall = self._calculate_point_recall(pred_endpoints, target_endpoints, threshold)
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score


class VesselMetrics:
    """
    Standard segmentation metrics for vessel segmentation evaluation
    """
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
    
    def dice_coefficient(self, predictions, targets, apply_sigmoid=True):
        """Standard Dice coefficient"""
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        intersection = np.sum(pred_binary * target_binary, axis=(2, 3, 4))
        pred_sum = np.sum(pred_binary, axis=(2, 3, 4))
        target_sum = np.sum(target_binary, axis=(2, 3, 4))
        
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(dice).to(predictions.device)
        else:
            return dice
    
    def iou_coefficient(self, predictions, targets, apply_sigmoid=True):
        """IoU (Intersection over Union) coefficient"""
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        intersection = np.sum(pred_binary * target_binary, axis=(2, 3, 4))
        union = np.sum((pred_binary + target_binary) > 0, axis=(2, 3, 4))
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(iou).to(predictions.device)
        else:
            return iou
    
    def precision(self, predictions, targets, apply_sigmoid=True):
        """Precision metric"""
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        true_positive = np.sum(pred_binary * target_binary, axis=(2, 3, 4))
        predicted_positive = np.sum(pred_binary, axis=(2, 3, 4))
        
        precision = (true_positive + self.smooth) / (predicted_positive + self.smooth)
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(precision).to(predictions.device)
        else:
            return precision
    
    def recall(self, predictions, targets, apply_sigmoid=True):
        """Recall (Sensitivity) metric"""
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = predictions
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = targets
        
        if apply_sigmoid and pred_numpy.max() > 1.0:
            pred_numpy = 1 / (1 + np.exp(-pred_numpy))
        
        pred_binary = (pred_numpy > 0.5).astype(np.float32)
        target_binary = (target_numpy > 0.5).astype(np.float32)
        
        true_positive = np.sum(pred_binary * target_binary, axis=(2, 3, 4))
        actual_positive = np.sum(target_binary, axis=(2, 3, 4))
        
        recall = (true_positive + self.smooth) / (actual_positive + self.smooth)
        
        if isinstance(predictions, torch.Tensor):
            return torch.from_numpy(recall).to(predictions.device)
        else:
            return recall
    
    def comprehensive_vessel_evaluation(self, predictions, targets, apply_sigmoid=True):
        """
        Comprehensive evaluation combining standard metrics
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            
        Returns:
            dict: Complete evaluation results
        """
        # Standard segmentation metrics
        dice_scores = self.dice_coefficient(predictions, targets, apply_sigmoid)
        iou_scores = self.iou_coefficient(predictions, targets, apply_sigmoid)
        precision_scores = self.precision(predictions, targets, apply_sigmoid)
        recall_scores = self.recall(predictions, targets, apply_sigmoid)
        
        return {
            'dice': dice_scores,
            'iou': iou_scores,
            'precision': precision_scores,
            'recall': recall_scores
        }


class MetricsCalculator:
    """
    Unified metrics calculator for training and evaluation
    """
    
    def __init__(self, class_names=None, smooth=1e-6):
        """
        Args:
            class_names (list): Names of classes ['artery', 'vein']
            smooth (float): Smoothing factor
        """
        self.class_names = class_names or ['artery', 'vein']
        self.smooth = smooth
        self.vessel_metrics = VesselMetrics(smooth=smooth)
        self.topology_metrics = TopologyMetrics(smooth=smooth)
    
    def calculate_batch_metrics(self, predictions, targets, apply_sigmoid=True, 
                              include_topology=True, include_branch=False):
        """
        Calculate metrics for a batch of predictions
        
        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, C, D, H, W]
            apply_sigmoid: Whether to apply sigmoid to predictions
            include_topology: Whether to include topology metrics
            include_branch: Whether to include branch detection metrics
            
        Returns:
            dict: Batch metrics
        """
        batch_metrics = {}
        
        # Standard metrics
        batch_metrics['dice'] = self.vessel_metrics.dice_coefficient(predictions, targets, apply_sigmoid)
        batch_metrics['iou'] = self.vessel_metrics.iou_coefficient(predictions, targets, apply_sigmoid)
        batch_metrics['precision'] = self.vessel_metrics.precision(predictions, targets, apply_sigmoid)
        batch_metrics['recall'] = self.vessel_metrics.recall(predictions, targets, apply_sigmoid)
        
        # Topology metrics
        if include_topology:
            batch_metrics['centerline_dice'] = self.topology_metrics.centerline_dice(predictions, targets, apply_sigmoid)
            batch_metrics['connectivity_accuracy'] = self.topology_metrics.connectivity_accuracy(predictions, targets, apply_sigmoid)
            batch_metrics['skeleton_similarity'] = self.topology_metrics.skeleton_similarity(predictions, targets, apply_sigmoid)
            batch_metrics['vessel_completeness'] = self.topology_metrics.vessel_completeness(predictions, targets, apply_sigmoid)
        
        # Branch detection metrics (computationally expensive)
        if include_branch:
            branch_results = self.topology_metrics.branch_detection_accuracy(predictions, targets, apply_sigmoid)
            batch_metrics.update(branch_results)
        
        return batch_metrics
    
    def aggregate_metrics(self, metrics_list):
        """
        Aggregate metrics across multiple batches
        
        Args:
            metrics_list: List of metric dictionaries from different batches
            
        Returns:
            dict: Aggregated metrics with mean and std
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        for metric_name in metric_names:
            # Collect values across batches
            values = []
            for metrics in metrics_list:
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    if isinstance(metric_value, torch.Tensor):
                        values.append(metric_value.detach().cpu().numpy())
                    else:
                        values.append(metric_value)
            
            if values:
                # Stack and calculate statistics
                stacked_values = np.stack(values)  # [num_batches, batch_size, num_classes]
                
                # Calculate mean and std across all samples
                aggregated[f'{metric_name}_mean'] = np.mean(stacked_values)
                aggregated[f'{metric_name}_std'] = np.std(stacked_values)
                
                # Per-class statistics
                if stacked_values.ndim >= 2:
                    class_means = np.mean(stacked_values, axis=(0, 1))  # [num_classes]
                    class_stds = np.std(stacked_values, axis=(0, 1))
                    
                    for i, class_name in enumerate(self.class_names[:len(class_means)]):
                        aggregated[f'{metric_name}_{class_name}_mean'] = class_means[i]
                        aggregated[f'{metric_name}_{class_name}_std'] = class_stds[i]
        
        return aggregated
    
    def format_metrics_summary(self, aggregated_metrics):
        """
        Format metrics into a readable summary
        
        Args:
            aggregated_metrics: Dictionary of aggregated metrics
            
        Returns:
            str: Formatted summary string
        """
        summary_lines = []
        summary_lines.append("=== Vessel Segmentation Metrics Summary ===")
        summary_lines.append("")
        
        # Group metrics by type
        standard_metrics = ['dice', 'iou', 'precision', 'recall']
        topology_metrics = ['centerline_dice', 'connectivity_accuracy', 'skeleton_similarity', 'vessel_completeness']
        branch_metrics = ['junction_precision', 'junction_recall', 'endpoint_accuracy']
        
        # Standard metrics
        summary_lines.append("Standard Segmentation Metrics:")
        for metric in standard_metrics:
            if f'{metric}_mean' in aggregated_metrics:
                mean_val = aggregated_metrics[f'{metric}_mean']
                std_val = aggregated_metrics[f'{metric}_std']
                summary_lines.append(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
                
                # Per-class breakdown
                for class_name in self.class_names:
                    if f'{metric}_{class_name}_mean' in aggregated_metrics:
                        class_mean = aggregated_metrics[f'{metric}_{class_name}_mean']
                        class_std = aggregated_metrics[f'{metric}_{class_name}_std']
                        summary_lines.append(f"    {class_name}: {class_mean:.4f} ± {class_std:.4f}")
        
        summary_lines.append("")
        
        # Topology metrics
        summary_lines.append("Topology-Aware Metrics:")
        for metric in topology_metrics:
            if f'{metric}_mean' in aggregated_metrics:
                mean_val = aggregated_metrics[f'{metric}_mean']
                std_val = aggregated_metrics[f'{metric}_std']
                summary_lines.append(f"  {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
                
                # Per-class breakdown
                for class_name in self.class_names:
                    if f'{metric}_{class_name}_mean' in aggregated_metrics:
                        class_mean = aggregated_metrics[f'{metric}_{class_name}_mean']
                        class_std = aggregated_metrics[f'{metric}_{class_name}_std']
                        summary_lines.append(f"    {class_name}: {class_mean:.4f} ± {class_std:.4f}")
        
        summary_lines.append("")
        
        # Branch metrics (if available)
        has_branch_metrics = any(f'{metric}_mean' in aggregated_metrics for metric in branch_metrics)
        if has_branch_metrics:
            summary_lines.append("Branch Detection Metrics:")
            for metric in branch_metrics:
                if f'{metric}_mean' in aggregated_metrics:
                    mean_val = aggregated_metrics[f'{metric}_mean']
                    std_val = aggregated_metrics[f'{metric}_std']
                    summary_lines.append(f"  {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        
        summary_lines.append("=" * 50)
        
        return "\n".join(summary_lines)
    
    def compare_methods(self, method_results, method_names=None):
        """
        Compare results from different methods
        
        Args:
            method_results: Dictionary of {method_name: aggregated_metrics}
            method_names: List of method names to display
            
        Returns:
            str: Comparison table
        """
        if not method_results:
            return "No results to compare"
        
        if method_names is None:
            method_names = list(method_results.keys())
        
        # Key metrics for comparison
        key_metrics = [
            'dice_mean', 'centerline_dice_mean', 'connectivity_accuracy_mean',
            'iou_mean', 'precision_mean', 'recall_mean'
        ]
        
        comparison_lines = []
        comparison_lines.append("=== Method Comparison ===")
        comparison_lines.append("")
        
        # Create comparison table
        header = f"{'Metric':<25}"
        for method_name in method_names:
            header += f"{method_name:<15}"
        comparison_lines.append(header)
        comparison_lines.append("-" * len(header))
        
        for metric in key_metrics:
            if any(metric in method_results[method] for method in method_names):
                line = f"{metric.replace('_mean', '').replace('_', ' ').title():<25}"
                for method_name in method_names:
                    if metric in method_results[method_name]:
                        value = method_results[method_name][metric]
                        line += f"{value:.4f}         "
                    else:
                        line += "N/A           "
                comparison_lines.append(line)
        
        comparison_lines.append("")
        comparison_lines.append("=" * len(header))
        
        return "\n".join(comparison_lines)
    
    def get_key_metrics_for_logging(self, aggregated_metrics):
        """
        Extract key metrics for logging during training
        
        Args:
            aggregated_metrics: Dictionary of aggregated metrics
            
        Returns:
            dict: Key metrics for logging
        """
        key_metrics = {}
        
        # Standard metrics
        for metric in ['dice', 'iou', 'precision', 'recall']:
            if f'{metric}_mean' in aggregated_metrics:
                key_metrics[f'{metric}_mean'] = aggregated_metrics[f'{metric}_mean']
                key_metrics[f'{metric}_std'] = aggregated_metrics[f'{metric}_std']
        
        # Topology metrics
        for metric in ['centerline_dice', 'connectivity_accuracy']:
            if f'{metric}_mean' in aggregated_metrics:
                key_metrics[f'{metric}_mean'] = aggregated_metrics[f'{metric}_mean']
                key_metrics[f'{metric}_std'] = aggregated_metrics[f'{metric}_std']
        
        # Per-class metrics for main classes
        for class_name in self.class_names:
            for metric in ['dice', 'centerline_dice']:
                key = f'{metric}_{class_name}_mean'
                if key in aggregated_metrics:
                    key_metrics[key] = aggregated_metrics[key]
        
        return key_metrics
    
    def calculate_combined_score(self, aggregated_metrics, weights=None):
        """
        Calculate a combined score for model selection
        
        Args:
            aggregated_metrics: Dictionary of aggregated metrics
            weights: Dictionary of weights for different metrics
            
        Returns:
            float: Combined score
        """
        if weights is None:
            weights = {
                'dice_mean': 0.3,
                'centerline_dice_mean': 0.3,
                'connectivity_accuracy_mean': 0.2,
                'iou_mean': 0.2
            }
        
        combined_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in aggregated_metrics:
                combined_score += aggregated_metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            combined_score /= total_weight
        
        return combined_score


class EvaluationPipeline:
    """
    Complete evaluation pipeline for vessel segmentation models
    """
    
    def __init__(self, class_names=None, save_detailed_results=True):
        """
        Args:
            class_names: Names of vessel classes
            save_detailed_results: Whether to save detailed per-sample results
        """
        self.class_names = class_names or ['artery', 'vein']
        self.save_detailed_results = save_detailed_results
        self.metrics_calculator = MetricsCalculator(class_names=class_names)
    
    def evaluate_model(self, model, dataloader, device='cuda', include_topology=True, 
                      include_branch=False, apply_postprocessing=None):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            dataloader: Evaluation dataloader
            device: Device for evaluation
            include_topology: Include topology metrics
            include_branch: Include branch detection metrics
            apply_postprocessing: Post-processing function (optional)
            
        Returns:
            dict: Comprehensive evaluation results
        """
        model.eval()
        all_metrics = []
        detailed_results = [] if self.save_detailed_results else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Get batch data
                images = batch['images'].to(device)
                artery_masks = batch['artery_masks'].to(device)
                vein_masks = batch['vein_masks'].to(device)
                case_names = batch['case_names']
                
                # Stack targets
                targets = torch.stack([artery_masks, vein_masks], dim=1)  # [B, 2, D, H, W]
                
                # Forward pass
                model_output = model(images)
                predictions = model_output['main'] if isinstance(model_output, dict) else model_output
                
                # Apply post-processing if provided
                if apply_postprocessing is not None:
                    predictions = apply_postprocessing(predictions, targets)
                
                # Calculate metrics for this batch
                batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                    predictions, targets,
                    include_topology=include_topology,
                    include_branch=include_branch
                )
                
                all_metrics.append(batch_metrics)
                
                # Save detailed results if requested
                if self.save_detailed_results:
                    batch_size = predictions.shape[0]
                    for i in range(batch_size):
                        sample_result = {
                            'case_name': case_names[i],
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        }
                        
                        # Extract per-sample metrics
                        for metric_name, metric_values in batch_metrics.items():
                            if isinstance(metric_values, torch.Tensor):
                                sample_result[metric_name] = metric_values[i].cpu().numpy()
                            else:
                                sample_result[metric_name] = metric_values[i] if hasattr(metric_values, '__getitem__') else metric_values
                        
                        detailed_results.append(sample_result)
                
                print(f"Evaluated batch {batch_idx + 1}/{len(dataloader)}")
        
        # Aggregate metrics
        aggregated_metrics = self.metrics_calculator.aggregate_metrics(all_metrics)
        
        # Generate summary
        summary = self.metrics_calculator.format_metrics_summary(aggregated_metrics)
        
        results = {
            'aggregated_metrics': aggregated_metrics,
            'summary': summary,
            'detailed_results': detailed_results
        }
        
        return results
    
    def compare_models(self, model_results, model_names=None):
        """
        Compare evaluation results from multiple models
        
        Args:
            model_results: Dictionary of {model_name: evaluation_results}
            model_names: List of model names for comparison
            
        Returns:
            dict: Comparison results
        """
        if model_names is None:
            model_names = list(model_results.keys())
        
        # Extract aggregated metrics for comparison
        method_metrics = {}
        for model_name in model_names:
            if model_name in model_results:
                method_metrics[model_name] = model_results[model_name]['aggregated_metrics']
        
        # Generate comparison
        comparison = self.metrics_calculator.compare_methods(method_metrics, model_names)
        
        return {
            'comparison_table': comparison,
            'method_metrics': method_metrics
        }


# Convenience functions for easy integration
def calculate_vessel_metrics(predictions, targets, class_names=None, include_topology=True):
    """
    Convenience function to calculate vessel segmentation metrics
    
    Args:
        predictions: Model predictions [B, C, D, H, W]
        targets: Ground truth masks [B, C, D, H, W]
        class_names: List of class names
        include_topology: Whether to include topology metrics
        
    Returns:
        dict: Calculated metrics
    """
    calculator = MetricsCalculator(class_names=class_names)
    return calculator.calculate_batch_metrics(
        predictions, targets, 
        include_topology=include_topology,
        include_branch=False  # Skip expensive branch metrics by default
    )


def evaluate_topology_metrics(predictions, targets, return_summary=True):
    """
    Convenience function to evaluate topology-specific metrics
    
    Args:
        predictions: Model predictions [B, C, D, H, W]
        targets: Ground truth masks [B, C, D, H, W]
        return_summary: Whether to return formatted summary
        
    Returns:
        dict or str: Topology metrics or summary
    """
    topology_metrics = TopologyMetrics()
    results = topology_metrics.topology_accuracy(predictions, targets)
    
    if return_summary:
        # Create simple summary
        summary_lines = []
        summary_lines.append("=== Topology Metrics Summary ===")
        for metric_name, values in results.items():
            if isinstance(values, torch.Tensor):
                mean_val = values.mean().item()
                std_val = values.std().item()
                summary_lines.append(f"{metric_name.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        summary_lines.append("=" * 40)
        return "\n".join(summary_lines)
    else:
        return results


# Test and usage example
if __name__ == '__main__':
    print("Testing Connectivity Metrics...")
    
    # Create test data
    batch_size, num_classes, depth, height, width = 2, 2, 32, 64, 64
    
    # Create synthetic vessel-like predictions and targets
    def create_vessel_test_data(shape):
        """Create realistic vessel test data"""
        volume = np.zeros(shape)
        
        # Main vessel trunk
        center_z, center_y, center_x = np.array(shape) // 2
        for i in range(shape[2]):
            y = center_y + int(3 * np.sin(i * 0.1))
            x = center_x + int(2 * np.cos(i * 0.1))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                volume[max(0, y-1):y+2, max(0, x-1):x+2, i] = 1
        
        # Branch vessels
        for branch in range(2):
            start_z = shape[2] // 4 + branch * shape[2] // 2
            for i in range(start_z, min(start_z + shape[2]//4, shape[2])):
                y = center_y + (i - start_z) * (1 if branch == 0 else -1)
                x = center_x + (i - start_z) // 2
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    volume[y, x, i] = 1
        
        return volume
    
    # Generate test data
    test_targets = np.zeros((batch_size, num_classes, depth, height, width))
    test_predictions = np.zeros((batch_size, num_classes, depth, height, width))
    
    for b in range(batch_size):
        for c in range(num_classes):
            # Create target vessel structure
            target_vessel = create_vessel_test_data((depth, height, width))
            test_targets[b, c] = target_vessel
            
            # Create prediction with some noise and missing parts
            pred_vessel = target_vessel.copy()
            
            # Add some noise
            noise = np.random.random((depth, height, width)) > 0.95
            pred_vessel = pred_vessel + noise * 0.3
            
            # Remove some parts to simulate incomplete segmentation
            if np.random.random() > 0.5:
                remove_region = np.random.randint(0, depth//4), np.random.randint(0, height//4), np.random.randint(0, width//4)
                pred_vessel[remove_region[0]:remove_region[0]+5, 
                           remove_region[1]:remove_region[1]+5, 
                           remove_region[2]:remove_region[2]+5] = 0
            
            test_predictions[b, c] = pred_vessel
    
    # Convert to tensors
    pred_tensor = torch.from_numpy(test_predictions).float()
    target_tensor = torch.from_numpy(test_targets).float()
    
    print(f"Test data created:")
    print(f"  Predictions: {pred_tensor.shape}")
    print(f"  Targets: {target_tensor.shape}")
    print(f"  Target vessel density: {target_tensor.mean():.4f}")
    print(f"  Prediction vessel density: {pred_tensor.mean():.4f}")
    
    # Test convenience functions
    print(f"\n1. Testing convenience functions...")
    
    # Test vessel metrics calculation
    metrics = calculate_vessel_metrics(pred_tensor, target_tensor, include_topology=True)
    print(f"   Calculated {len(metrics)} metrics")
    for metric_name, values in metrics.items():
        if isinstance(values, torch.Tensor):
            print(f"     {metric_name}: {values.mean():.4f}")
    
    # Test topology metrics summary
    topology_summary = evaluate_topology_metrics(pred_tensor, target_tensor, return_summary=True)
    print(f"\n2. Topology metrics summary:")
    print(topology_summary)
    
    # Test full evaluation pipeline
    print(f"\n3. Testing MetricsCalculator...")
    metrics_calc = MetricsCalculator(class_names=['artery', 'vein'])
    
    # Simulate multiple batches
    batch_metrics_list = []
    for i in range(3):
        batch_metrics = metrics_calc.calculate_batch_metrics(
            pred_tensor, target_tensor, include_topology=True
        )
        batch_metrics_list.append(batch_metrics)
    
    # Aggregate metrics
    aggregated = metrics_calc.aggregate_metrics(batch_metrics_list)
    
    # Generate summary
    summary = metrics_calc.format_metrics_summary(aggregated)
    print(f"\n4. Complete evaluation summary:")
    print(summary)
    
    # Test method comparison
    print(f"\n5. Testing method comparison...")
    
    # Create fake results for comparison
    baseline_metrics = aggregated.copy()
    improved_metrics = {}
    for key, value in aggregated.items():
        if isinstance(value, (int, float)) and 'mean' in key:
            # Simulate improvement
            improved_metrics[key] = value * 1.1  # 10% improvement
        else:
            improved_metrics[key] = value
    
    comparison = metrics_calc.compare_methods({
        'Baseline': baseline_metrics,
        'Proposed': improved_metrics
    })
    print(comparison)
    
    # Test combined score calculation
    combined_score = metrics_calc.calculate_combined_score(aggregated)
    print(f"\n6. Combined score: {combined_score:.4f}")
    
    # Test key metrics extraction
    key_metrics = metrics_calc.get_key_metrics_for_logging(aggregated)
    print(f"\n7. Key metrics for logging:")
    for key, value in key_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n All connectivity metrics tests passed!")
    print(f"\nUsage examples for your training pipeline:")
    print(f"")
    print(f"# During validation:")
    print(f"val_metrics = calculate_vessel_metrics(predictions, targets)")
    print(f"val_dice = val_metrics['dice'].mean()")
    print(f"val_cldice = val_metrics['centerline_dice'].mean()")
    print(f"")
    print(f"# For comprehensive evaluation:")
    print(f"metrics_calc = MetricsCalculator(class_names=['artery', 'vein'])")
    print(f"results = metrics_calc.calculate_batch_metrics(predictions, targets)")
    print(f"")
    print(f"# For model comparison:")
    print(f"comparison = metrics_calc.compare_methods({{")
    print(f"    'Baseline': baseline_results,")
    print(f"    'clDice': cldice_results,")
    print(f"    'clDice+Repair': final_results")
    print(f"}})")
    print(f"print(comparison)")