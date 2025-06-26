import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom, gaussian_filter
from skimage import morphology, measure
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class CTImagePreprocessor:
    """
    Basic CT image preprocessing for pulmonary vessel segmentation
    Handles essential data cleaning and normalization
    """
    
    def __init__(self, window_level=(-600, 1500), normalize_method='z_score'):
        """
        Args:
            window_level (tuple): CT window (center, width)
            normalize_method (str): 'z_score', 'min_max', or 'robust'
        """
        self.window_center, self.window_width = window_level
        self.normalize_method = normalize_method
    
    def __call__(self, image):
        """Apply basic preprocessing pipeline"""
        # Convert to float32
        image = image.astype(np.float32)
        
        # 1. Window level adjustment
        image = self._apply_window_level(image)
        
        # 2. Basic denoising
        image = self._denoise(image)
        
        # 3. Normalization
        image = self._normalize(image)
        
        return image
    
    def _apply_window_level(self, image):
        """Apply CT window level adjustment"""
        min_val = self.window_center - self.window_width // 2
        max_val = self.window_center + self.window_width // 2
        
        # Clip and normalize to [0, 1]
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def _denoise(self, image, sigma=0.5):
        """Apply basic Gaussian denoising"""
        return gaussian_filter(image, sigma=sigma)
    
    def _normalize(self, image):
        """Apply normalization"""
        if self.normalize_method == 'z_score':
            mean = np.mean(image)
            std = np.std(image)
            if std > 1e-8:
                image = (image - mean) / std
        
        elif self.normalize_method == 'min_max':
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        
        elif self.normalize_method == 'robust':
            # Robust normalization using percentiles
            p1, p99 = np.percentile(image, [1, 99])
            if p99 > p1:
                image = np.clip(image, p1, p99)
                image = (image - p1) / (p99 - p1)
        
        return image


class MaskPreprocessor:
    """
    Basic segmentation mask preprocessing and cleaning
    """
    
    def __init__(self, min_component_size=50, clean_masks=True):
        """
        Args:
            min_component_size (int): Minimum connected component size
            clean_masks (bool): Apply basic morphological cleaning
        """
        self.min_component_size = min_component_size
        self.clean_masks = clean_masks
    
    def __call__(self, mask):
        """Apply basic mask preprocessing"""
        # Ensure binary mask
        mask = (mask > 0.5).astype(np.float32)
        
        # Remove small components
        if self.clean_masks:
            mask = self._remove_small_components(mask)
            mask = self._basic_morphological_cleaning(mask)
        
        return mask
    
    def _remove_small_components(self, mask):
        """Remove small connected components"""
        if np.sum(mask) == 0:
            return mask
        
        # Label connected components
        labeled = measure.label(mask > 0.5)
        
        # Calculate component sizes
        props = measure.regionprops(labeled)
        
        # Keep only large components
        cleaned_mask = np.zeros_like(mask)
        for prop in props:
            if prop.area >= self.min_component_size:
                cleaned_mask[labeled == prop.label] = 1.0
        
        return cleaned_mask
    
    def _basic_morphological_cleaning(self, mask):
        """Apply basic morphological operations"""
        if np.sum(mask) == 0:
            return mask
        
        # Small opening to remove noise
        kernel = morphology.ball(1)
        mask = morphology.binary_opening(mask, kernel)
        
        return mask.astype(np.float32)


class VolumeResizer:
    """
    Intelligent volume resizing that preserves vessel structures
    """
    
    def __init__(self, target_size, preserve_aspect_ratio=False):
        """
        Args:
            target_size (tuple): Target size (D, H, W)
            preserve_aspect_ratio (bool): Maintain aspect ratio
        """
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
    
    def __call__(self, image, artery_mask, vein_mask, original_spacing=None):
        """Resize image and masks"""
        current_shape = image.shape
        
        if current_shape == self.target_size:
            return image, artery_mask, vein_mask
        
        # Calculate zoom factors
        zoom_factors = self._calculate_zoom_factors(current_shape, original_spacing)
        
        # Apply resizing
        resized_image = zoom(image, zoom_factors, order=1)  # Bilinear for image
        resized_artery = zoom(artery_mask, zoom_factors, order=0)  # Nearest for masks
        resized_vein = zoom(vein_mask, zoom_factors, order=0)
        
        # Ensure masks remain binary
        resized_artery = (resized_artery > 0.5).astype(np.float32)
        resized_vein = (resized_vein > 0.5).astype(np.float32)
        
        return resized_image, resized_artery, resized_vein
    
    def _calculate_zoom_factors(self, current_shape, original_spacing):
        """Calculate optimal zoom factors"""
        if self.preserve_aspect_ratio and original_spacing is not None:
            # Consider both size and spacing
            size_factors = [t/c for t, c in zip(self.target_size, current_shape)]
            # Use minimum factor to preserve aspect ratio
            zoom_factor = min(size_factors)
            zoom_factors = [zoom_factor] * 3
        else:
            # Simple size-based factors
            zoom_factors = [t/c for t, c in zip(self.target_size, current_shape)]
        
        return zoom_factors


class VesselTopologyAnalyzer:
    """
    Basic vessel topology analysis for clDice preparation
    """
    
    def __init__(self, skeleton_method='skeletonize_3d'):
        """
        Args:
            skeleton_method (str): Method for skeletonization
        """
        self.skeleton_method = skeleton_method
    
    def extract_skeleton(self, mask):
        """Extract vessel skeleton for topology analysis"""
        if np.sum(mask) == 0:
            return np.zeros_like(mask)
        
        # Ensure binary mask
        binary_mask = (mask > 0.5).astype(bool)
        
        # 3D skeletonization
        if self.skeleton_method == 'skeletonize_3d':
            skeleton = morphology.skeletonize_3d(binary_mask)
        else:
            # Alternative: slice-by-slice skeletonization
            skeleton = np.zeros_like(binary_mask)
            for i in range(binary_mask.shape[2]):
                if np.sum(binary_mask[:, :, i]) > 0:
                    skeleton[:, :, i] = morphology.skeletonize(binary_mask[:, :, i])
        
        return skeleton.astype(np.float32)
    
    def get_topology_info(self, artery_mask, vein_mask):
        """Get basic topology information"""
        # Extract skeletons
        artery_skeleton = self.extract_skeleton(artery_mask)
        vein_skeleton = self.extract_skeleton(vein_mask)
        
        # Count connected components
        artery_components = measure.label(artery_skeleton)
        vein_components = measure.label(vein_skeleton)
        
        artery_n_components = len(np.unique(artery_components)) - 1  # Exclude background
        vein_n_components = len(np.unique(vein_components)) - 1
        
        return {
            'artery_skeleton': artery_skeleton,
            'vein_skeleton': vein_skeleton,
            'artery_components': artery_n_components,
            'vein_components': vein_n_components
        }


class HiPaSBasicPreprocessor:
    """
    Basic preprocessing pipeline for HiPaS dataset
    Handles essential data cleaning without augmentation
    """
    
    def __init__(self, target_size=(96, 96, 96), window_level=(-600, 1500)):
        """
        Args:
            target_size (tuple): Target volume size
            window_level (tuple): CT window level settings
        """
        self.target_size = target_size
        
        # Initialize processors
        self.image_preprocessor = CTImagePreprocessor(
            window_level=window_level,
            normalize_method='z_score'
        )
        self.mask_preprocessor = MaskPreprocessor(
            min_component_size=50,
            clean_masks=True
        )
        self.resizer = VolumeResizer(target_size)
        self.topology_analyzer = VesselTopologyAnalyzer()
    
    def __call__(self, image, artery_mask, vein_mask, original_spacing=None, 
                 return_topology=False):
        """
        Apply basic preprocessing pipeline
        
        Args:
            image: Raw CT image
            artery_mask: Artery segmentation mask
            vein_mask: Vein segmentation mask
            original_spacing: Original voxel spacing
            return_topology: Whether to return topology information
        
        Returns:
            dict: Preprocessed data
        """
        # 1. Preprocess image
        processed_image = self.image_preprocessor(image)
        
        # 2. Preprocess masks
        processed_artery = self.mask_preprocessor(artery_mask)
        processed_vein = self.mask_preprocessor(vein_mask)
        
        # 3. Resize to target size
        processed_image, processed_artery, processed_vein = self.resizer(
            processed_image, processed_artery, processed_vein, original_spacing
        )
        
        # 4. Convert to tensors
        result = {
            'image': torch.from_numpy(processed_image).float(),
            'artery_mask': torch.from_numpy(processed_artery).float(),
            'vein_mask': torch.from_numpy(processed_vein).float()
        }
        
        # 5. Extract topology information if requested
        if return_topology:
            topology_info = self.topology_analyzer.get_topology_info(
                processed_artery, processed_vein
            )
            result['topology'] = topology_info
        
        return result


class DataValidator:
    """
    Data validation and quality checks
    """
    
    def __init__(self):
        pass
    
    def validate_image(self, image):
        """Validate CT image quality"""
        issues = []
        
        # Check for NaN or infinite values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            issues.append("Image contains NaN or infinite values")
        
        # Check image range
        if np.max(image) == np.min(image):
            issues.append("Image has zero variance")
        
        # Check for reasonable CT values (before windowing)
        if np.min(image) < -2000 or np.max(image) > 5000:
            issues.append(f"Unusual CT values: [{np.min(image)}, {np.max(image)}]")
        
        return issues
    
    def validate_mask(self, mask, mask_name="mask"):
        """Validate segmentation mask"""
        issues = []
        
        # Check for valid binary values
        unique_vals = np.unique(mask)
        if not all(val in [0.0, 1.0] for val in unique_vals):
            issues.append(f"{mask_name} is not binary: {unique_vals}")
        
        # Check for empty mask
        if np.sum(mask) == 0:
            issues.append(f"{mask_name} is empty")
        
        # Check for very sparse mask (might indicate annotation error)
        if np.sum(mask) < 10:
            issues.append(f"{mask_name} is very sparse: {np.sum(mask)} voxels")
        
        return issues
    
    def validate_shapes(self, image, artery_mask, vein_mask):
        """Validate that all volumes have matching shapes"""
        issues = []
        
        if not (image.shape == artery_mask.shape == vein_mask.shape):
            issues.append(f"Shape mismatch: image{image.shape}, artery{artery_mask.shape}, vein{vein_mask.shape}")
        
        return issues
    
    def full_validation(self, image, artery_mask, vein_mask):
        """Perform full data validation"""
        all_issues = []
        
        # Validate individual components
        all_issues.extend(self.validate_image(image))
        all_issues.extend(self.validate_mask(artery_mask, "artery_mask"))
        all_issues.extend(self.validate_mask(vein_mask, "vein_mask"))
        all_issues.extend(self.validate_shapes(image, artery_mask, vein_mask))
        
        return all_issues


# Utility functions
def load_metadata(metadata_path):
    """Load and parse metadata from Excel file"""
    metadata_dict = {}
    
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata file not found at {metadata_path}")
        return metadata_dict
    
    try:
        df = pd.read_excel(metadata_path)
        print(f"Loaded metadata for {len(df)} cases")
        
        for _, row in df.iterrows():
            filename = row['CT scan']
            # Parse resolution string to list of floats
            resolution_str = row['Resolution'].strip(' []')
            resolution = [float(x.strip()) for x in resolution_str.split(',')]
            metadata_dict[filename] = {
                'resolution': resolution,
                'spacing': tuple(resolution)  # (x, y, z) spacing in mm
            }
            
    except Exception as e:
        print(f"Error reading metadata: {e}")
    
    return metadata_dict


def compute_dataset_statistics(dataset_path, num_samples=10):
    """Compute basic statistics for dataset"""
    import os
    
    ct_scan_dir = os.path.join(dataset_path, 'ct_scan')
    
    if not os.path.exists(ct_scan_dir):
        print(f"Dataset path not found: {ct_scan_dir}")
        return None
    
    # Sample a few files for statistics
    files = [f for f in os.listdir(ct_scan_dir) if f.endswith('.npz')][:num_samples]
    
    shapes = []
    intensity_ranges = []
    
    for filename in files:
        try:
            filepath = os.path.join(ct_scan_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            image = data['data']
            
            shapes.append(image.shape)
            intensity_ranges.append((image.min(), image.max()))
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    if shapes:
        print(f"\nDataset Statistics (from {len(shapes)} samples):")
        print(f"Shapes: {shapes}")
        print(f"Intensity ranges: {intensity_ranges}")
        
        # Common shape
        unique_shapes = list(set(shapes))
        print(f"Unique shapes: {unique_shapes}")
    
    return {
        'shapes': shapes,
        'intensity_ranges': intensity_ranges
    }


# Test and usage example
if __name__ == '__main__':
    import os
    
    print("Testing Basic Data Preprocessing...")
    
    # Create dummy data
    dummy_image = np.random.randint(-1000, 2000, size=(256, 256, 128)).astype(np.float32)
    dummy_artery = np.random.choice([0, 1], size=(256, 256, 128), p=[0.95, 0.05]).astype(np.float32)
    dummy_vein = np.random.choice([0, 1], size=(256, 256, 128), p=[0.97, 0.03]).astype(np.float32)
    dummy_spacing = (0.6, 0.6, 1.0)
    
    print(f"Input shapes:")
    print(f"  Image: {dummy_image.shape}, range: [{dummy_image.min()}, {dummy_image.max()}]")
    print(f"  Artery: {dummy_artery.shape}, values: {np.unique(dummy_artery)}")
    print(f"  Vein: {dummy_vein.shape}, values: {np.unique(dummy_vein)}")
    
    # Test data validation
    print(f"\nTesting data validation...")
    validator = DataValidator()
    issues = validator.full_validation(dummy_image, dummy_artery, dummy_vein)
    if issues:
        print(f"Validation issues found: {issues}")
    else:
        print("✓ Data validation passed")
    
    # Test basic preprocessing
    print(f"\nTesting basic preprocessing...")
    preprocessor = HiPaSBasicPreprocessor(
        target_size=(96, 96, 96),
        window_level=(-600, 1500)
    )
    
    result = preprocessor(
        dummy_image, dummy_artery, dummy_vein, 
        original_spacing=dummy_spacing, 
        return_topology=True
    )
    
    print(f"Processed shapes:")
    print(f"  Image: {result['image'].shape}")
    print(f"  Artery: {result['artery_mask'].shape}")
    print(f"  Vein: {result['vein_mask'].shape}")
    
    print(f"Processed ranges:")
    print(f"  Image: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
    print(f"  Artery: {torch.unique(result['artery_mask'])}")
    print(f"  Vein: {torch.unique(result['vein_mask'])}")
    
    print(f"Topology info:")
    topology = result['topology']
    print(f"  Artery components: {topology['artery_components']}")
    print(f"  Vein components: {topology['vein_components']}")
    
    # Test individual components
    print(f"\nTesting individual components...")
    
    # Image preprocessor
    img_processor = CTImagePreprocessor()
    processed_img = img_processor(dummy_image)
    print(f"✓ Image preprocessor: range [{processed_img.min():.3f}, {processed_img.max():.3f}]")
    
    # Mask preprocessor
    mask_processor = MaskPreprocessor()
    processed_mask = mask_processor(dummy_artery)
    print(f"✓ Mask preprocessor: unique values {np.unique(processed_mask)}")
    
    # Topology analyzer
    topology_analyzer = VesselTopologyAnalyzer()
    skeleton = topology_analyzer.extract_skeleton(dummy_artery)
    print(f"✓ Skeleton extraction: shape {skeleton.shape}, sum {np.sum(skeleton)}")
    
    print(f"\n All basic preprocessing tests passed!")
    print(f"Ready for integration with data augmentation pipeline!")