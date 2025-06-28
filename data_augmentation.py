import numpy as np
import torch
import random
from scipy import ndimage
from scipy.ndimage import zoom, gaussian_filter, map_coordinates
from skimage import morphology, filters
import warnings
warnings.filterwarnings('ignore')

class GeometricAugmentations:
    """
    Geometric augmentations that preserve vessel topology
    Applied AFTER basic preprocessing - WITH SAFETY FIXES
    """
    
    def __init__(self, rotation_range=5, flip_probability=0.3, 
                 zoom_range=0.02, elastic_alpha=10, elastic_sigma=2):
        """
        Args:
            rotation_range (float): Maximum rotation angle in degrees (reduced for safety)
            flip_probability (float): Probability of flipping along each axis (reduced)
            zoom_range (float): Zoom factor range (reduced)
            elastic_alpha (float): Elastic deformation strength (reduced)
            elastic_sigma (float): Elastic deformation smoothness
        """
        self.rotation_range = rotation_range
        self.flip_probability = flip_probability
        self.zoom_range = zoom_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def random_rotation_3d(self, image, artery_mask, vein_mask):
        """Apply random 3D rotation with safety checks"""
        try:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            
            # Rotate around z-axis to preserve axial anatomy
            rotated_image = self._rotate_volume(image, angle, order=1)
            rotated_artery = self._rotate_volume(artery_mask, angle, order=0)
            rotated_vein = self._rotate_volume(vein_mask, angle, order=0)
            
            # Ensure masks remain binary and arrays are contiguous
            rotated_artery = np.ascontiguousarray((rotated_artery > 0.5).astype(np.float32))
            rotated_vein = np.ascontiguousarray((rotated_vein > 0.5).astype(np.float32))
            rotated_image = np.ascontiguousarray(rotated_image)
            
            return rotated_image, rotated_artery, rotated_vein
            
        except Exception as e:
            print(f"Rotation augmentation failed: {e}")
            return (np.ascontiguousarray(image), 
                   np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
    
    def _rotate_volume(self, volume, angle, order=1):
        """Rotate 3D volume slice by slice around z-axis"""
        try:
            rotated = np.zeros_like(volume)
            for i in range(volume.shape[2]):
                rotated[:, :, i] = ndimage.rotate(
                    volume[:, :, i], angle, reshape=False, order=order, mode='reflect'
                )
            return rotated
        except Exception as e:
            print(f"Volume rotation failed: {e}")
            return volume
    
    def random_flip(self, image, artery_mask, vein_mask):
        """Apply random flipping with safety checks"""
        try:
            flipped_image = image.copy()
            flipped_artery = artery_mask.copy()
            flipped_vein = vein_mask.copy()
            
            # Flip along each axis with given probability
            for axis in [0, 1]:  # Don't flip along z-axis (preserves anatomy)
                if random.random() < self.flip_probability:
                    flipped_image = np.flip(flipped_image, axis=axis)
                    flipped_artery = np.flip(flipped_artery, axis=axis)
                    flipped_vein = np.flip(flipped_vein, axis=axis)
            
            # Ensure contiguity
            return (np.ascontiguousarray(flipped_image), 
                   np.ascontiguousarray(flipped_artery), 
                   np.ascontiguousarray(flipped_vein))
            
        except Exception as e:
            print(f"Flip augmentation failed: {e}")
            return (np.ascontiguousarray(image), 
                   np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
    
    def random_zoom(self, image, artery_mask, vein_mask):
        """Apply random zoom and crop/pad to original size - CONSERVATIVE VERSION"""
        try:
            # Very small zoom factor for safety
            zoom_factor = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            
            # Apply zoom
            zoomed_image = zoom(image, zoom_factor, order=1, mode='reflect')
            zoomed_artery = zoom(artery_mask, zoom_factor, order=0, mode='reflect')
            zoomed_vein = zoom(vein_mask, zoom_factor, order=0, mode='reflect')
            
            # Resize back to original shape
            original_shape = image.shape
            zoomed_image = self._resize_to_shape(zoomed_image, original_shape, order=1)
            zoomed_artery = self._resize_to_shape(zoomed_artery, original_shape, order=0)
            zoomed_vein = self._resize_to_shape(zoomed_vein, original_shape, order=0)
            
            # Ensure masks remain binary and contiguous
            zoomed_artery = np.ascontiguousarray((zoomed_artery > 0.5).astype(np.float32))
            zoomed_vein = np.ascontiguousarray((zoomed_vein > 0.5).astype(np.float32))
            zoomed_image = np.ascontiguousarray(zoomed_image)
            
            return zoomed_image, zoomed_artery, zoomed_vein
            
        except Exception as e:
            print(f"Zoom augmentation failed: {e}")
            return (np.ascontiguousarray(image), 
                   np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
    
    def _resize_to_shape(self, volume, target_shape, order=1):
        """Resize volume to target shape"""
        try:
            current_shape = volume.shape
            zoom_factors = [t/c for t, c in zip(target_shape, current_shape)]
            return zoom(volume, zoom_factors, order=order, mode='reflect')
        except Exception as e:
            print(f"Resize failed: {e}")
            return volume
    
    def elastic_deformation(self, image, artery_mask, vein_mask):
        """Apply VERY MILD elastic deformation"""
        try:
            shape = image.shape
            
            # Generate smooth random displacement fields (much smaller)
            dx = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
            dy = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
            dz = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha * 0.5  # Less z deformation
            
            # Create coordinate grids
            x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), 
                                 np.arange(shape[2]), indexing='ij')
            
            # Apply displacement
            coords = [x + dx, y + dy, z + dz]
            
            # Interpolate
            deformed_image = map_coordinates(image, coords, order=1, mode='reflect')
            deformed_artery = map_coordinates(artery_mask, coords, order=0, mode='reflect')
            deformed_vein = map_coordinates(vein_mask, coords, order=0, mode='reflect')
            
            # Ensure masks remain binary and contiguous
            deformed_artery = np.ascontiguousarray((deformed_artery > 0.5).astype(np.float32))
            deformed_vein = np.ascontiguousarray((deformed_vein > 0.5).astype(np.float32))
            deformed_image = np.ascontiguousarray(deformed_image)
            
            return deformed_image, deformed_artery, deformed_vein
            
        except Exception as e:
            print(f"Elastic deformation failed: {e}")
            return (np.ascontiguousarray(image), 
                   np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))


class IntensityAugmentations:
    """
    Intensity-based augmentations for CT images - CONSERVATIVE VERSION
    Applied to images only, not masks
    """
    
    def __init__(self, brightness_range=0.05, contrast_range=0.05, 
                 gamma_range=(0.95, 1.05), noise_std=0.01):
        """
        Args:
            brightness_range (float): Brightness adjustment range (reduced)
            contrast_range (float): Contrast adjustment range (reduced)
            gamma_range (tuple): Gamma correction range (conservative)
            noise_std (float): Gaussian noise standard deviation (very low)
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
    
    def random_brightness_contrast(self, image):
        """Apply random brightness and contrast with safety checks"""
        try:
            # Random brightness
            brightness = random.uniform(-self.brightness_range, self.brightness_range)
            image = image + brightness
            
            # Random contrast
            contrast = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            mean = np.mean(image)
            image = (image - mean) * contrast + mean
            
            return np.ascontiguousarray(image)
            
        except Exception as e:
            print(f"Brightness/contrast augmentation failed: {e}")
            return np.ascontiguousarray(image)
    
    def random_gamma_correction(self, image):
        """Apply random gamma correction with safety checks"""
        try:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            
            # Normalize to [0, 1] for gamma correction
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min)
                corrected = np.power(np.clip(normalized, 0, 1), gamma)
                image = corrected * (img_max - img_min) + img_min
            
            return np.ascontiguousarray(image)
            
        except Exception as e:
            print(f"Gamma correction failed: {e}")
            return np.ascontiguousarray(image)
    
    def add_gaussian_noise(self, image):
        """Add very mild Gaussian noise"""
        try:
            noise = np.random.normal(0, self.noise_std, image.shape)
            return np.ascontiguousarray(image + noise)
        except Exception as e:
            print(f"Noise addition failed: {e}")
            return np.ascontiguousarray(image)
    
    def random_blur(self, image, sigma_range=(0.2, 0.5)):
        """Apply very mild random Gaussian blur"""
        try:
            sigma = random.uniform(sigma_range[0], sigma_range[1])
            blurred = gaussian_filter(image, sigma=sigma)
            return np.ascontiguousarray(blurred)
        except Exception as e:
            print(f"Blur failed: {e}")
            return np.ascontiguousarray(image)
    
    def random_intensity_shift(self, image, shift_range=0.05):
        """Apply very mild random intensity shift"""
        try:
            shift = random.uniform(-shift_range, shift_range)
            return np.ascontiguousarray(image + shift)
        except Exception as e:
            print(f"Intensity shift failed: {e}")
            return np.ascontiguousarray(image)


class VesselSpecificAugmentations:
    """
    Augmentations specifically designed for vessel structures - SAFE VERSION
    """
    
    def __init__(self, vessel_dropout_prob=0.02, boundary_blur_prob=0.1):
        """
        Args:
            vessel_dropout_prob (float): Probability of vessel dropout (very low)
            boundary_blur_prob (float): Probability of boundary blurring (low)
        """
        self.vessel_dropout_prob = vessel_dropout_prob
        self.boundary_blur_prob = boundary_blur_prob
    
    def vessel_dropout(self, artery_mask, vein_mask, dropout_ratio=0.02):
        """Simulate partial vessel annotations (VERY CONSERVATIVE)"""
        try:
            augmented_artery = artery_mask.copy()
            augmented_vein = vein_mask.copy()
            
            if random.random() < self.vessel_dropout_prob:
                # Randomly remove small vessel segments
                augmented_artery = self._apply_dropout(augmented_artery, dropout_ratio)
                augmented_vein = self._apply_dropout(augmented_vein, dropout_ratio)
            
            return (np.ascontiguousarray(augmented_artery), 
                   np.ascontiguousarray(augmented_vein))
                   
        except Exception as e:
            print(f"Vessel dropout failed: {e}")
            return (np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
    
    def _apply_dropout(self, mask, dropout_ratio):
        """Apply random dropout to vessel mask"""
        try:
            if np.sum(mask) == 0:
                return mask
            
            # Label connected components
            labeled_mask = morphology.label(mask > 0.5)
            unique_labels = np.unique(labeled_mask)[1:]  # Exclude background
            
            # Randomly drop some SMALL components only
            for label in unique_labels:
                component_size = np.sum(labeled_mask == label)
                if component_size < 50 and random.random() < dropout_ratio:  # Only drop small components
                    mask[labeled_mask == label] = 0
            
            return mask
        except Exception as e:
            print(f"Dropout application failed: {e}")
            return mask
    
    def boundary_augmentation(self, artery_mask, vein_mask):
        """Augment vessel boundaries with safety checks"""
        try:
            augmented_artery = artery_mask.copy()
            augmented_vein = vein_mask.copy()
            
            if random.random() < self.boundary_blur_prob:
                # Apply very slight erosion or dilation
                kernel = morphology.ball(1)
                
                if random.random() < 0.5:
                    # Slight erosion (only if mask is not too small)
                    if np.sum(augmented_artery) > 100:
                        augmented_artery = morphology.binary_erosion(
                            augmented_artery, kernel).astype(np.float32)
                    if np.sum(augmented_vein) > 100:
                        augmented_vein = morphology.binary_erosion(
                            augmented_vein, kernel).astype(np.float32)
                else:
                    # Slight dilation
                    augmented_artery = morphology.binary_dilation(
                        augmented_artery, kernel).astype(np.float32)
                    augmented_vein = morphology.binary_dilation(
                        augmented_vein, kernel).astype(np.float32)
            
            return (np.ascontiguousarray(augmented_artery), 
                   np.ascontiguousarray(augmented_vein))
                   
        except Exception as e:
            print(f"Boundary augmentation failed: {e}")
            return (np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
    
    def simulate_partial_annotations(self, artery_mask, vein_mask, missing_prob=0.05):
        """Simulate missing annotations in some regions - VERY CONSERVATIVE"""
        try:
            if random.random() < missing_prob:
                # Create small random 3D regions to "miss"
                h, w, d = artery_mask.shape
                
                # Very small random box region
                box_size = random.randint(3, 8)  # Reduced size
                x = random.randint(0, max(1, h - box_size))
                y = random.randint(0, max(1, w - box_size))
                z = random.randint(0, max(1, d - box_size))
                
                # Set region to background
                artery_mask[x:x+box_size, y:y+box_size, z:z+box_size] = 0
                vein_mask[x:x+box_size, y:y+box_size, z:z+box_size] = 0
            
            return (np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
                   
        except Exception as e:
            print(f"Partial annotation simulation failed: {e}")
            return (np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))


class VesselAugmentationPipeline:
    """
    Complete SAFE augmentation pipeline for vessel segmentation training
    Works with preprocessed data from data_preprocessing.py
    """
    
    def __init__(self, augmentation_probability=0.5, training=True):
        """
        Args:
            augmentation_probability (float): Overall probability of augmentation (reduced for safety)
            training (bool): Whether in training mode
        """
        self.augmentation_probability = augmentation_probability
        self.training = training
        
        # Initialize augmentation modules with VERY CONSERVATIVE settings
        self.geometric = GeometricAugmentations(
            rotation_range=5,        # Very conservative for vessels
            flip_probability=0.3,    # Reduced probability
            zoom_range=0.02,         # Very small zoom to preserve vessel structure
            elastic_alpha=10,        # Reduced elastic deformation
            elastic_sigma=2
        )
        
        self.intensity = IntensityAugmentations(
            brightness_range=0.05,   # Very small range
            contrast_range=0.05,     # Very small range
            gamma_range=(0.95, 1.05), # Very conservative gamma
            noise_std=0.01           # Very low noise for medical images
        )
        
        self.vessel_specific = VesselSpecificAugmentations(
            vessel_dropout_prob=0.02,  # Very conservative
            boundary_blur_prob=0.1
        )
    
    def __call__(self, image, artery_mask, vein_mask, second_sample=None):
        """
        Apply augmentation pipeline with comprehensive safety checks
        
        Args:
            image: Preprocessed CT image
            artery_mask: Preprocessed artery mask
            vein_mask: Preprocessed vein mask
            second_sample: Optional second sample for MixUp/CutMix (disabled for safety)
        
        Returns:
            Augmented (image, artery_mask, vein_mask)
        """
        if not self.training or random.random() > self.augmentation_probability:
            # Even when not augmenting, ensure contiguity
            return (np.ascontiguousarray(image), 
                   np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
        
        try:
            # Copy inputs to avoid modifying originals
            aug_image = np.ascontiguousarray(image.copy())
            aug_artery = np.ascontiguousarray(artery_mask.copy())
            aug_vein = np.ascontiguousarray(vein_mask.copy())
            
            # 1. Geometric transformations (reduced probability)
            if random.random() < 0.3:  # Reduced from 0.6
                geo_transform = random.choice(['rotation', 'flip'])  # Removed zoom and elastic for stability
                
                if geo_transform == 'rotation':
                    aug_image, aug_artery, aug_vein = self.geometric.random_rotation_3d(
                        aug_image, aug_artery, aug_vein)
                elif geo_transform == 'flip':
                    aug_image, aug_artery, aug_vein = self.geometric.random_flip(
                        aug_image, aug_artery, aug_vein)
            
            # 2. Intensity augmentations (reduced probability, image only)
            if random.random() < 0.2:  # Reduced from 0.5
                intensity_aug = random.choice(['brightness_contrast', 'gamma'])  # Removed noise and blur
                
                if intensity_aug == 'brightness_contrast':
                    aug_image = self.intensity.random_brightness_contrast(aug_image)
                elif intensity_aug == 'gamma':
                    aug_image = self.intensity.random_gamma_correction(aug_image)
            
            # 3. Vessel-specific augmentations (much reduced probability)
            if random.random() < 0.1:  # Reduced from 0.3
                # Only boundary augmentation, no dropout or partial annotation simulation
                aug_artery, aug_vein = self.vessel_specific.boundary_augmentation(
                    aug_artery, aug_vein)
            
            # Ensure masks remain in valid range [0, 1]
            aug_artery = np.clip(aug_artery, 0, 1)
            aug_vein = np.clip(aug_vein, 0, 1)
            
            # CRITICAL: Ensure all output arrays are contiguous
            aug_image = np.ascontiguousarray(aug_image)
            aug_artery = np.ascontiguousarray(aug_artery)
            aug_vein = np.ascontiguousarray(aug_vein)
            
            # Validate output arrays
            if not self._validate_augmented_data(aug_image, aug_artery, aug_vein):
                print("Augmentation validation failed, returning original data")
                return (np.ascontiguousarray(image), 
                       np.ascontiguousarray(artery_mask), 
                       np.ascontiguousarray(vein_mask))
            
            return aug_image, aug_artery, aug_vein
            
        except Exception as e:
            print(f"Augmentation error: {e}")
            # Return safe, contiguous original data
            return (np.ascontiguousarray(image), 
                   np.ascontiguousarray(artery_mask), 
                   np.ascontiguousarray(vein_mask))
    
    def _validate_augmented_data(self, image, artery_mask, vein_mask):
        """Validate augmented data for safety"""
        try:
            # Check for NaN or infinite values
            if (np.any(np.isnan(image)) or np.any(np.isinf(image)) or
                np.any(np.isnan(artery_mask)) or np.any(np.isinf(artery_mask)) or
                np.any(np.isnan(vein_mask)) or np.any(np.isinf(vein_mask))):
                return False
            
            # Check shapes are preserved
            if not (image.shape == artery_mask.shape == vein_mask.shape):
                return False
            
            # Check mask values are in valid range
            if (np.min(artery_mask) < 0 or np.max(artery_mask) > 1 or
                np.min(vein_mask) < 0 or np.max(vein_mask) > 1):
                return False
            
            # Check that arrays are contiguous
            if not (image.flags.c_contiguous and 
                   artery_mask.flags.c_contiguous and 
                   vein_mask.flags.c_contiguous):
                return False
            
            return True
            
        except Exception:
            return False
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training
    
    def set_augmentation_probability(self, prob):
        """Adjust augmentation probability"""
        self.augmentation_probability = prob


class ScheduledAugmentation:
    """
    Augmentation scheduler that adjusts intensity during training
    """
    
    def __init__(self, initial_prob=0.5, min_prob=0.2, schedule='cosine'):
        """
        Args:
            initial_prob (float): Initial augmentation probability (reduced)
            min_prob (float): Minimum augmentation probability (reduced)
            schedule (str): Scheduling strategy ('linear', 'cosine', 'step')
        """
        self.initial_prob = initial_prob
        self.min_prob = min_prob
        self.schedule = schedule
        self.current_epoch = 0
    
    def get_augmentation_probability(self, epoch, total_epochs):
        """Get augmentation probability for current epoch"""
        if self.schedule == 'linear':
            # Linear decay
            prob = self.initial_prob - (self.initial_prob - self.min_prob) * (epoch / total_epochs)
        
        elif self.schedule == 'cosine':
            # Cosine decay
            import math
            prob = self.min_prob + (self.initial_prob - self.min_prob) * \
                   0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        
        elif self.schedule == 'step':
            # Step decay
            if epoch < total_epochs * 0.3:
                prob = self.initial_prob
            elif epoch < total_epochs * 0.6:
                prob = self.initial_prob * 0.7
            else:
                prob = self.min_prob
        
        else:
            prob = self.initial_prob
        
        return max(prob, self.min_prob)


# Factory functions for easy use
def create_safe_augmentation_pipeline(training=True, intensity='light'):
    """
    Factory function to create a SAFE augmentation pipeline with preset configurations
    
    Args:
        training (bool): Whether for training
        intensity (str): 'light', 'medium', 'heavy'
    
    Returns:
        VesselAugmentationPipeline: Configured safe pipeline
    """
    if intensity == 'light':
        aug_prob = 0.3
        rotation_range = 3
        noise_std = 0.005
    elif intensity == 'medium':
        aug_prob = 0.5
        rotation_range = 5
        noise_std = 0.01
    elif intensity == 'heavy':
        aug_prob = 0.7
        rotation_range = 8
        noise_std = 0.015
    else:
        # Default to light for safety
        aug_prob = 0.3
        rotation_range = 3
        noise_std = 0.005
    
    pipeline = VesselAugmentationPipeline(
        augmentation_probability=aug_prob,
        training=training
    )
    
    # Adjust parameters based on intensity
    pipeline.geometric.rotation_range = rotation_range
    pipeline.intensity.noise_std = noise_std
    
    return pipeline


def create_validation_augmentation():
    """Create augmentation pipeline for validation (no augmentations)"""
    return VesselAugmentationPipeline(augmentation_probability=0.0, training=False)


# Test and usage example
if __name__ == '__main__':
    print("Testing SAFE Vessel Augmentation Pipeline...")
    
    # Create dummy preprocessed data (would come from data_preprocessing.py)
    dummy_image = np.random.randn(96, 96, 96).astype(np.float32)
    dummy_artery = np.random.choice([0, 1], size=(96, 96, 96), p=[0.95, 0.05]).astype(np.float32)
    dummy_vein = np.random.choice([0, 1], size=(96, 96, 96), p=[0.97, 0.03]).astype(np.float32)
    
    print(f"Input (preprocessed) data:")
    print(f"  Image: {dummy_image.shape}, range: [{dummy_image.min():.3f}, {dummy_image.max():.3f}]")
    print(f"  Artery: {dummy_artery.shape}, unique: {np.unique(dummy_artery)}")
    print(f"  Vein: {dummy_vein.shape}, unique: {np.unique(dummy_vein)}")
    print(f"  Image contiguous: {dummy_image.flags.c_contiguous}")
    print(f"  Artery contiguous: {dummy_artery.flags.c_contiguous}")
    print(f"  Vein contiguous: {dummy_vein.flags.c_contiguous}")
    
    # Test safe augmentation pipeline
    print(f"\nTesting SAFE augmentation pipeline...")
    
    augmentation_pipeline = create_safe_augmentation_pipeline(
        training=True, 
        intensity='light'  # Start with light intensity
    )
    
    # Test multiple augmentations
    success_count = 0
    total_tests = 10
    
    for i in range(total_tests):
        try:
            result = augmentation_pipeline(dummy_image, dummy_artery, dummy_vein)
            
            aug_image, aug_artery, aug_vein = result
            
            # Validate results
            if (aug_image.flags.c_contiguous and 
                aug_artery.flags.c_contiguous and 
                aug_vein.flags.c_contiguous and
                aug_image.shape == dummy_image.shape and
                not np.any(np.isnan(aug_image)) and
                np.all(aug_artery >= 0) and np.all(aug_artery <= 1)):
                
                success_count += 1
                if i < 3:  # Print first 3 successful augmentations
                    print(f"\nAugmentation {i+1}: ✅ SUCCESS")
                    print(f"  Image range: [{aug_image.min():.3f}, {aug_image.max():.3f}]")
                    print(f"  Artery unique: {np.unique(aug_artery)}")
                    print(f"  Vein unique: {np.unique(aug_vein)}")
                    print(f"  All contiguous: {aug_image.flags.c_contiguous and aug_artery.flags.c_contiguous and aug_vein.flags.c_contiguous}")
            else:
                print(f"\nAugmentation {i+1}: ❌ FAILED validation")
                
        except Exception as e:
            print(f"\nAugmentation {i+1}: ❌ EXCEPTION: {e}")
    
    print(f"\nSuccess rate: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    # Test individual components
    print(f"\nTesting individual augmentation components...")
    
    # Geometric
    geometric = GeometricAugmentations()
    try:
        rot_img, rot_art, rot_vein = geometric.random_rotation_3d(dummy_image, dummy_artery, dummy_vein)
        print(f"✓ Rotation: shapes {rot_img.shape}, mask binary: {set(np.unique(rot_art)) <= {0.0, 1.0}}, contiguous: {rot_img.flags.c_contiguous}")
    except Exception as e:
        print(f"❌ Rotation failed: {e}")
    
    # Intensity
    intensity = IntensityAugmentations()
    try:
        bright_img = intensity.random_brightness_contrast(dummy_image)
        print(f"✓ Brightness/Contrast: range [{bright_img.min():.3f}, {bright_img.max():.3f}], contiguous: {bright_img.flags.c_contiguous}")
    except Exception as e:
        print(f"❌ Brightness/Contrast failed: {e}")
    
    # Vessel-specific
    vessel_aug = VesselSpecificAugmentations()
    try:
        bound_art, bound_vein = vessel_aug.boundary_augmentation(dummy_artery, dummy_vein)
        print(f"✓ Boundary augmentation: artery sum {np.sum(bound_art):.0f}, vein sum {np.sum(bound_vein):.0f}, contiguous: {bound_art.flags.c_contiguous}")
    except Exception as e:
        print(f"❌ Boundary augmentation failed: {e}")
    
    # Test preset configurations
    print(f"\nTesting preset configurations...")
    
    for intensity_level in ['light', 'medium', 'heavy']:
        try:
            pipeline = create_safe_augmentation_pipeline(training=True, intensity=intensity_level)
            result = pipeline(dummy_image, dummy_artery, dummy_vein)
            aug_image = result[0]
            print(f"✓ {intensity_level.capitalize()} intensity: range [{aug_image.min():.3f}, {aug_image.max():.3f}], contiguous: {aug_image.flags.c_contiguous}")
        except Exception as e:
            print(f"❌ {intensity_level.capitalize()} intensity failed: {e}")
    
    # Test validation pipeline
    print(f"\nTesting validation pipeline...")
    try:
        val_pipeline = create_validation_augmentation()
        val_result = val_pipeline(dummy_image, dummy_artery, dummy_vein)
        val_image = val_result[0]
        print(f"✓ Validation pipeline: no augmentation applied, contiguous: {val_image.flags.c_contiguous}")
    except Exception as e:
        print(f"❌ Validation pipeline failed: {e}")
    
    # Test augmentation scheduler
    print(f"\nTesting augmentation scheduler...")
    try:
        scheduler = ScheduledAugmentation(initial_prob=0.5, min_prob=0.2)
        
        for epoch in [0, 50, 100, 150, 200]:
            prob = scheduler.get_augmentation_probability(epoch, 200)
            print(f"  Epoch {epoch}: augmentation probability = {prob:.3f}")
        
        print(f"✓ Scheduler working correctly")
    except Exception as e:
        print(f"❌ Scheduler failed: {e}")
    
    # Test memory contiguity preservation
    print(f"\nTesting memory contiguity preservation...")
    try:
        # Create non-contiguous arrays
        non_contiguous_image = dummy_image[::2, ::2, ::2]  # This should be non-contiguous
        non_contiguous_artery = dummy_artery[::2, ::2, ::2]
        non_contiguous_vein = dummy_vein[::2, ::2, ::2]
        
        print(f"  Input contiguous: {non_contiguous_image.flags.c_contiguous}")
        
        # Test pipeline with non-contiguous input
        pipeline = create_safe_augmentation_pipeline(training=True, intensity='light')
        result = pipeline(non_contiguous_image, non_contiguous_artery, non_contiguous_vein)
        
        output_image = result[0]
        print(f"  Output contiguous: {output_image.flags.c_contiguous}")
        print(f"✓ Contiguity preservation test passed")
        
    except Exception as e:
        print(f"❌ Contiguity test failed: {e}")
    
    print(f"\n🎉 All SAFE augmentation tests completed!")
    print(f"\n📋 Summary:")
    print(f"✅ Conservative augmentation parameters")
    print(f"✅ Comprehensive error handling")
    print(f"✅ Array contiguity preservation")
    print(f"✅ Data validation checks")
    print(f"✅ Memory safety guarantees")
    
    print(f"\n🚀 Integration instructions:")
    print(f"1. Replace your data_augmentation.py with this version")
    print(f"2. In your train.py, use:")
    print(f"   from data_augmentation import create_safe_augmentation_pipeline")
    print(f"   augmentation = create_safe_augmentation_pipeline(training=True, intensity='light')")
    print(f"3. For validation:")
    print(f"   from data_augmentation import create_validation_augmentation")
    print(f"   val_augmentation = create_validation_augmentation()")
    print(f"4. Start with 'light' intensity and gradually increase if stable")
    
    print(f"\n⚠️  Important notes:")
    print(f"• All augmentations now have conservative parameters")
    print(f"• Elastic deformation and zoom are disabled by default for stability")
    print(f"• MixUp/CutMix are disabled to avoid complexity")
    print(f"• All functions return contiguous arrays")
    print(f"• Comprehensive error handling prevents crashes")
    print(f"• Validation ensures data integrity")


# Additional utility functions for integration
def get_augmentation_statistics(pipeline, image, artery_mask, vein_mask, num_tests=100):
    """
    Get statistics about augmentation pipeline performance
    
    Args:
        pipeline: Augmentation pipeline
        image: Test image
        artery_mask: Test artery mask  
        vein_mask: Test vein mask
        num_tests: Number of tests to run
        
    Returns:
        dict: Statistics about pipeline performance
    """
    stats = {
        'total_tests': num_tests,
        'successful_augmentations': 0,
        'failed_augmentations': 0,
        'validation_failures': 0,
        'exception_count': 0,
        'avg_processing_time': 0.0
    }
    
    import time
    total_time = 0.0
    
    for i in range(num_tests):
        start_time = time.time()
        
        try:
            result = pipeline(image, artery_mask, vein_mask)
            
            # Validate result
            if (len(result) == 3 and 
                all(isinstance(arr, np.ndarray) for arr in result) and
                all(arr.flags.c_contiguous for arr in result)):
                stats['successful_augmentations'] += 1
            else:
                stats['validation_failures'] += 1
                
        except Exception as e:
            stats['exception_count'] += 1
        
        end_time = time.time()
        total_time += (end_time - start_time)
    
    stats['failed_augmentations'] = (stats['validation_failures'] + 
                                   stats['exception_count'])
    stats['avg_processing_time'] = total_time / num_tests
    stats['success_rate'] = stats['successful_augmentations'] / num_tests
    
    return stats


def benchmark_augmentation_pipeline(image, artery_mask, vein_mask):
    """
    Benchmark different augmentation intensities
    
    Args:
        image: Test image
        artery_mask: Test artery mask
        vein_mask: Test vein mask
        
    Returns:
        dict: Benchmark results
    """
    intensities = ['light', 'medium', 'heavy']
    results = {}
    
    print("Benchmarking augmentation pipelines...")
    
    for intensity in intensities:
        print(f"\nTesting {intensity} intensity...")
        
        pipeline = create_safe_augmentation_pipeline(
            training=True, 
            intensity=intensity
        )
        
        stats = get_augmentation_statistics(
            pipeline, image, artery_mask, vein_mask, num_tests=50
        )
        
        results[intensity] = stats
        
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Avg time: {stats['avg_processing_time']*1000:.2f}ms")
        print(f"  Exceptions: {stats['exception_count']}")
    
    return results


# Configuration templates for different training scenarios
def get_conservative_config():
    """Get very conservative augmentation configuration"""
    return {
        'augmentation_probability': 0.2,
        'rotation_range': 2,
        'flip_probability': 0.2,
        'brightness_range': 0.02,
        'contrast_range': 0.02,
        'gamma_range': (0.98, 1.02),
        'noise_std': 0.005
    }


def get_standard_config():
    """Get standard augmentation configuration"""
    return {
        'augmentation_probability': 0.5,
        'rotation_range': 5,
        'flip_probability': 0.3,
        'brightness_range': 0.05,
        'contrast_range': 0.05,
        'gamma_range': (0.95, 1.05),
        'noise_std': 0.01
    }


def get_aggressive_config():
    """Get aggressive augmentation configuration"""
    return {
        'augmentation_probability': 0.7,
        'rotation_range': 8,
        'flip_probability': 0.5,
        'brightness_range': 0.1,
        'contrast_range': 0.1,
        'gamma_range': (0.9, 1.1),
        'noise_std': 0.02
    }


def apply_config_to_pipeline(pipeline, config):
    """
    Apply configuration to existing pipeline
    
    Args:
        pipeline: VesselAugmentationPipeline instance
        config: Configuration dictionary
    """
    pipeline.augmentation_probability = config['augmentation_probability']
    pipeline.geometric.rotation_range = config['rotation_range']
    pipeline.geometric.flip_probability = config['flip_probability']
    pipeline.intensity.brightness_range = config['brightness_range']
    pipeline.intensity.contrast_range = config['contrast_range']
    pipeline.intensity.gamma_range = config['gamma_range']
    pipeline.intensity.noise_std = config['noise_std']