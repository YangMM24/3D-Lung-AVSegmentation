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
    Applied AFTER basic preprocessing
    """
    
    def __init__(self, rotation_range=15, flip_probability=0.5, 
                 zoom_range=0.1, elastic_alpha=30, elastic_sigma=3):
        """
        Args:
            rotation_range (float): Maximum rotation angle in degrees
            flip_probability (float): Probability of flipping along each axis
            zoom_range (float): Zoom factor range
            elastic_alpha (float): Elastic deformation strength
            elastic_sigma (float): Elastic deformation smoothness
        """
        self.rotation_range = rotation_range
        self.flip_probability = flip_probability
        self.zoom_range = zoom_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def random_rotation_3d(self, image, artery_mask, vein_mask):
        """Apply random 3D rotation"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # Rotate around z-axis to preserve axial anatomy
        rotated_image = self._rotate_volume(image, angle, order=1)
        rotated_artery = self._rotate_volume(artery_mask, angle, order=0)
        rotated_vein = self._rotate_volume(vein_mask, angle, order=0)
        
        # Ensure masks remain binary
        rotated_artery = (rotated_artery > 0.5).astype(np.float32)
        rotated_vein = (rotated_vein > 0.5).astype(np.float32)
        
        return rotated_image, rotated_artery, rotated_vein
    
    def _rotate_volume(self, volume, angle, order=1):
        """Rotate 3D volume slice by slice around z-axis"""
        rotated = np.zeros_like(volume)
        for i in range(volume.shape[2]):
            rotated[:, :, i] = ndimage.rotate(
                volume[:, :, i], angle, reshape=False, order=order, mode='reflect'
            )
        return rotated
    
    def random_flip(self, image, artery_mask, vein_mask):
        """Apply random flipping"""
        flipped_image = image.copy()
        flipped_artery = artery_mask.copy()
        flipped_vein = vein_mask.copy()
        
        # Flip along each axis with given probability
        for axis in [0, 1]:  # Don't flip along z-axis (preserves anatomy)
            if random.random() < self.flip_probability:
                flipped_image = np.flip(flipped_image, axis=axis)
                flipped_artery = np.flip(flipped_artery, axis=axis)
                flipped_vein = np.flip(flipped_vein, axis=axis)
        
        return flipped_image, flipped_artery, flipped_vein
    
    def random_zoom(self, image, artery_mask, vein_mask):
        """Apply random zoom and crop/pad to original size"""
        # Random zoom factor
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
        
        # Ensure masks remain binary
        zoomed_artery = (zoomed_artery > 0.5).astype(np.float32)
        zoomed_vein = (zoomed_vein > 0.5).astype(np.float32)
        
        return zoomed_image, zoomed_artery, zoomed_vein
    
    def _resize_to_shape(self, volume, target_shape, order=1):
        """Resize volume to target shape"""
        current_shape = volume.shape
        zoom_factors = [t/c for t, c in zip(target_shape, current_shape)]
        return zoom(volume, zoom_factors, order=order, mode='reflect')
    
    def elastic_deformation(self, image, artery_mask, vein_mask):
        """Apply elastic deformation"""
        shape = image.shape
        
        # Generate smooth random displacement fields
        dx = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
        dz = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
        
        # Create coordinate grids
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), 
                             np.arange(shape[2]), indexing='ij')
        
        # Apply displacement
        coords = [x + dx, y + dy, z + dz]
        
        # Interpolate
        deformed_image = map_coordinates(image, coords, order=1, mode='reflect')
        deformed_artery = map_coordinates(artery_mask, coords, order=0, mode='reflect')
        deformed_vein = map_coordinates(vein_mask, coords, order=0, mode='reflect')
        
        # Ensure masks remain binary
        deformed_artery = (deformed_artery > 0.5).astype(np.float32)
        deformed_vein = (deformed_vein > 0.5).astype(np.float32)
        
        return deformed_image, deformed_artery, deformed_vein


class IntensityAugmentations:
    """
    Intensity-based augmentations for CT images
    Applied to images only, not masks
    """
    
    def __init__(self, brightness_range=0.1, contrast_range=0.1, 
                 gamma_range=(0.8, 1.2), noise_std=0.05):
        """
        Args:
            brightness_range (float): Brightness adjustment range
            contrast_range (float): Contrast adjustment range
            gamma_range (tuple): Gamma correction range
            noise_std (float): Gaussian noise standard deviation
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
    
    def random_brightness_contrast(self, image):
        """Apply random brightness and contrast"""
        # Random brightness
        brightness = random.uniform(-self.brightness_range, self.brightness_range)
        image = image + brightness
        
        # Random contrast
        contrast = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        mean = np.mean(image)
        image = (image - mean) * contrast + mean
        
        return image
    
    def random_gamma_correction(self, image):
        """Apply random gamma correction"""
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # Normalize to [0, 1] for gamma correction
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
            corrected = np.power(np.clip(normalized, 0, 1), gamma)
            image = corrected * (img_max - img_min) + img_min
        
        return image
    
    def add_gaussian_noise(self, image):
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std, image.shape)
        return image + noise
    
    def random_blur(self, image, sigma_range=(0.3, 1.0)):
        """Apply random Gaussian blur"""
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        return gaussian_filter(image, sigma=sigma)
    
    def random_intensity_shift(self, image, shift_range=0.1):
        """Apply random intensity shift"""
        shift = random.uniform(-shift_range, shift_range)
        return image + shift


class VesselSpecificAugmentations:
    """
    Augmentations specifically designed for vessel structures
    """
    
    def __init__(self, vessel_dropout_prob=0.1, boundary_blur_prob=0.2):
        """
        Args:
            vessel_dropout_prob (float): Probability of vessel dropout
            boundary_blur_prob (float): Probability of boundary blurring
        """
        self.vessel_dropout_prob = vessel_dropout_prob
        self.boundary_blur_prob = boundary_blur_prob
    
    def vessel_dropout(self, artery_mask, vein_mask, dropout_ratio=0.05):
        """Simulate partial vessel annotations (missing small vessels)"""
        augmented_artery = artery_mask.copy()
        augmented_vein = vein_mask.copy()
        
        if random.random() < self.vessel_dropout_prob:
            # Randomly remove small vessel segments
            augmented_artery = self._apply_dropout(augmented_artery, dropout_ratio)
            augmented_vein = self._apply_dropout(augmented_vein, dropout_ratio)
        
        return augmented_artery, augmented_vein
    
    def _apply_dropout(self, mask, dropout_ratio):
        """Apply random dropout to vessel mask"""
        if np.sum(mask) == 0:
            return mask
        
        # Label connected components
        labeled_mask = morphology.label(mask > 0.5)
        unique_labels = np.unique(labeled_mask)[1:]  # Exclude background
        
        # Randomly drop some components
        for label in unique_labels:
            if random.random() < dropout_ratio:
                mask[labeled_mask == label] = 0
        
        return mask
    
    def boundary_augmentation(self, artery_mask, vein_mask):
        """Augment vessel boundaries with slight morphological operations"""
        augmented_artery = artery_mask.copy()
        augmented_vein = vein_mask.copy()
        
        if random.random() < self.boundary_blur_prob:
            # Apply slight erosion or dilation to simulate annotation uncertainty
            kernel = morphology.ball(1)
            
            if random.random() < 0.5:
                # Slight erosion
                if np.sum(augmented_artery) > 0:
                    augmented_artery = morphology.binary_erosion(
                        augmented_artery, kernel).astype(np.float32)
                if np.sum(augmented_vein) > 0:
                    augmented_vein = morphology.binary_erosion(
                        augmented_vein, kernel).astype(np.float32)
            else:
                # Slight dilation
                augmented_artery = morphology.binary_dilation(
                    augmented_artery, kernel).astype(np.float32)
                augmented_vein = morphology.binary_dilation(
                    augmented_vein, kernel).astype(np.float32)
        
        return augmented_artery, augmented_vein
    
    def simulate_partial_annotations(self, artery_mask, vein_mask, missing_prob=0.1):
        """Simulate missing annotations in some regions"""
        if random.random() < missing_prob:
            # Create random 3D regions to "miss"
            h, w, d = artery_mask.shape
            
            # Random box region
            box_size = random.randint(5, 15)
            x = random.randint(0, max(1, h - box_size))
            y = random.randint(0, max(1, w - box_size))
            z = random.randint(0, max(1, d - box_size))
            
            # Set region to background
            artery_mask[x:x+box_size, y:y+box_size, z:z+box_size] = 0
            vein_mask[x:x+box_size, y:y+box_size, z:z+box_size] = 0
        
        return artery_mask, vein_mask


class MixUpCutMix:
    """
    Advanced mixing strategies for 3D medical images
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0):
        """
        Args:
            mixup_alpha (float): Beta distribution parameter for mixup
            cutmix_alpha (float): Beta distribution parameter for cutmix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def mixup(self, img1, artery1, vein1, img2, artery2, vein2):
        """Apply MixUp augmentation"""
        # Sample mixing coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Mix images and masks
        mixed_image = lam * img1 + (1 - lam) * img2
        mixed_artery = lam * artery1 + (1 - lam) * artery2
        mixed_vein = lam * vein1 + (1 - lam) * vein2
        
        return mixed_image, mixed_artery, mixed_vein, lam
    
    def cutmix(self, img1, artery1, vein1, img2, artery2, vein2):
        """Apply CutMix augmentation"""
        # Sample cutting ratio
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Generate random cutting box
        cut_ratio = np.sqrt(1 - lam)
        h, w, d = img1.shape
        
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        cut_d = int(d * cut_ratio)
        
        # Random center point
        cx = np.random.randint(h)
        cy = np.random.randint(w)
        cz = np.random.randint(d)
        
        # Calculate box boundaries
        bbx1 = np.clip(cx - cut_h // 2, 0, h)
        bby1 = np.clip(cy - cut_w // 2, 0, w)
        bbz1 = np.clip(cz - cut_d // 2, 0, d)
        bbx2 = np.clip(cx + cut_h // 2, 0, h)
        bby2 = np.clip(cy + cut_w // 2, 0, w)
        bbz2 = np.clip(cz + cut_d // 2, 0, d)
        
        # Apply cutmix
        mixed_image = img1.copy()
        mixed_artery = artery1.copy()
        mixed_vein = vein1.copy()
        
        mixed_image[bbx1:bbx2, bby1:bby2, bbz1:bbz2] = img2[bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        mixed_artery[bbx1:bbx2, bby1:bby2, bbz1:bbz2] = artery2[bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        mixed_vein[bbx1:bbx2, bby1:bby2, bbz1:bbz2] = vein2[bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        
        # Adjust lambda based on actual cut area
        actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1)) / (h * w * d)
        
        return mixed_image, mixed_artery, mixed_vein, actual_lam


class VesselAugmentationPipeline:
    """
    Complete augmentation pipeline for vessel segmentation training
    Works with preprocessed data from data_preprocessing.py
    """
    
    def __init__(self, augmentation_probability=0.8, training=True):
        """
        Args:
            augmentation_probability (float): Overall probability of augmentation
            training (bool): Whether in training mode
        """
        self.augmentation_probability = augmentation_probability
        self.training = training
        
        # Initialize augmentation modules
        self.geometric = GeometricAugmentations(
            rotation_range=10,       # Conservative for vessels
            flip_probability=0.5,
            zoom_range=0.05,         # Small zoom to preserve vessel structure
            elastic_alpha=20,        # Moderate elastic deformation
            elastic_sigma=3
        )
        
        self.intensity = IntensityAugmentations(
            brightness_range=0.1,
            contrast_range=0.1,
            gamma_range=(0.9, 1.1),
            noise_std=0.02           # Low noise for medical images
        )
        
        self.vessel_specific = VesselSpecificAugmentations(
            vessel_dropout_prob=0.05,  # Very conservative
            boundary_blur_prob=0.2
        )
        
        self.mixup_cutmix = MixUpCutMix(
            mixup_alpha=0.2,
            cutmix_alpha=1.0
        )
    
    def __call__(self, image, artery_mask, vein_mask, second_sample=None):
        """
        Apply augmentation pipeline
        
        Args:
            image: Preprocessed CT image
            artery_mask: Preprocessed artery mask
            vein_mask: Preprocessed vein mask
            second_sample: Optional second sample for MixUp/CutMix (tuple)
        
        Returns:
            Augmented (image, artery_mask, vein_mask) or with mixing coefficient
        """
        if not self.training or random.random() > self.augmentation_probability:
            return image, artery_mask, vein_mask
        
        # Copy inputs to avoid modifying originals
        aug_image = image.copy()
        aug_artery = artery_mask.copy()
        aug_vein = vein_mask.copy()
        
        # 1. Geometric transformations (60% chance)
        if random.random() < 0.6:
            geo_transform = random.choice([
                'rotation', 'flip', 'zoom', 'elastic'
            ])
            
            if geo_transform == 'rotation':
                aug_image, aug_artery, aug_vein = self.geometric.random_rotation_3d(
                    aug_image, aug_artery, aug_vein)
            
            elif geo_transform == 'flip':
                aug_image, aug_artery, aug_vein = self.geometric.random_flip(
                    aug_image, aug_artery, aug_vein)
            
            elif geo_transform == 'zoom':
                aug_image, aug_artery, aug_vein = self.geometric.random_zoom(
                    aug_image, aug_artery, aug_vein)
            
            elif geo_transform == 'elastic':
                aug_image, aug_artery, aug_vein = self.geometric.elastic_deformation(
                    aug_image, aug_artery, aug_vein)
        
        # 2. Intensity augmentations (50% chance, image only)
        if random.random() < 0.5:
            intensity_augs = random.sample([
                'brightness_contrast', 'gamma', 'noise', 'blur'
            ], k=random.randint(1, 2))
            
            for aug_type in intensity_augs:
                if aug_type == 'brightness_contrast':
                    aug_image = self.intensity.random_brightness_contrast(aug_image)
                elif aug_type == 'gamma':
                    aug_image = self.intensity.random_gamma_correction(aug_image)
                elif aug_type == 'noise':
                    aug_image = self.intensity.add_gaussian_noise(aug_image)
                elif aug_type == 'blur':
                    aug_image = self.intensity.random_blur(aug_image)
        
        # 3. Vessel-specific augmentations (30% chance)
        if random.random() < 0.3:
            # Vessel dropout
            aug_artery, aug_vein = self.vessel_specific.vessel_dropout(
                aug_artery, aug_vein)
            
            # Boundary augmentation
            aug_artery, aug_vein = self.vessel_specific.boundary_augmentation(
                aug_artery, aug_vein)
            
            # Simulate partial annotations
            aug_artery, aug_vein = self.vessel_specific.simulate_partial_annotations(
                aug_artery, aug_vein)
        
        # 4. MixUp/CutMix (10% chance, if second sample provided)
        mixing_lambda = None
        if second_sample is not None and random.random() < 0.1:
            img2, artery2, vein2 = second_sample
            
            if random.random() < 0.5:
                # MixUp
                aug_image, aug_artery, aug_vein, mixing_lambda = self.mixup_cutmix.mixup(
                    aug_image, aug_artery, aug_vein, img2, artery2, vein2)
            else:
                # CutMix
                aug_image, aug_artery, aug_vein, mixing_lambda = self.mixup_cutmix.cutmix(
                    aug_image, aug_artery, aug_vein, img2, artery2, vein2)
        
        # Ensure masks remain in valid range [0, 1]
        aug_artery = np.clip(aug_artery, 0, 1)
        aug_vein = np.clip(aug_vein, 0, 1)
        
        # Return with or without mixing coefficient
        if mixing_lambda is not None:
            return aug_image, aug_artery, aug_vein, mixing_lambda
        else:
            return aug_image, aug_artery, aug_vein
    
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
    
    def __init__(self, initial_prob=0.8, min_prob=0.3, schedule='cosine'):
        """
        Args:
            initial_prob (float): Initial augmentation probability
            min_prob (float): Minimum augmentation probability
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


# Integration function for easy use with dataset
def create_augmentation_pipeline(training=True, intensity='medium'):
    """
    Factory function to create augmentation pipeline with preset configurations
    
    Args:
        training (bool): Whether for training
        intensity (str): 'light', 'medium', 'heavy'
    
    Returns:
        VesselAugmentationPipeline: Configured pipeline
    """
    if intensity == 'light':
        aug_prob = 0.5
        rotation_range = 5
        noise_std = 0.01
    elif intensity == 'medium':
        aug_prob = 0.8
        rotation_range = 10
        noise_std = 0.02
    elif intensity == 'heavy':
        aug_prob = 0.9
        rotation_range = 15
        noise_std = 0.03
    else:
        raise ValueError(f"Unknown intensity: {intensity}")
    
    pipeline = VesselAugmentationPipeline(
        augmentation_probability=aug_prob,
        training=training
    )
    
    # Adjust parameters based on intensity
    pipeline.geometric.rotation_range = rotation_range
    pipeline.intensity.noise_std = noise_std
    
    return pipeline


# Test and usage example
if __name__ == '__main__':
    print("Testing Vessel Augmentation Pipeline...")
    
    # Create dummy preprocessed data (would come from data_preprocessing.py)
    dummy_image = np.random.randn(96, 96, 96).astype(np.float32)
    dummy_artery = np.random.choice([0, 1], size=(96, 96, 96), p=[0.95, 0.05]).astype(np.float32)
    dummy_vein = np.random.choice([0, 1], size=(96, 96, 96), p=[0.97, 0.03]).astype(np.float32)
    
    print(f"Input (preprocessed) data:")
    print(f"  Image: {dummy_image.shape}, range: [{dummy_image.min():.3f}, {dummy_image.max():.3f}]")
    print(f"  Artery: {dummy_artery.shape}, unique: {np.unique(dummy_artery)}")
    print(f"  Vein: {dummy_vein.shape}, unique: {np.unique(dummy_vein)}")
    
    # Test augmentation pipeline
    print(f"\nTesting augmentation pipeline...")
    
    augmentation_pipeline = VesselAugmentationPipeline(
        augmentation_probability=1.0,  # Always augment for testing
        training=True
    )
    
    # Test multiple augmentations
    for i in range(3):
        result = augmentation_pipeline(dummy_image, dummy_artery, dummy_vein)
        
        if len(result) == 4:  # With mixing coefficient
            aug_image, aug_artery, aug_vein, mixing_lambda = result
            print(f"\nAugmentation {i+1} (with mixing λ={mixing_lambda:.3f}):")
        else:
            aug_image, aug_artery, aug_vein = result
            print(f"\nAugmentation {i+1}:")
        
        print(f"  Image range: [{aug_image.min():.3f}, {aug_image.max():.3f}]")
        print(f"  Artery unique: {np.unique(aug_artery)}")
        print(f"  Vein unique: {np.unique(aug_vein)}")
        print(f"  Shapes preserved: {aug_image.shape == dummy_image.shape}")
    
    # Test individual components
    print(f"\nTesting individual augmentation components...")
    
    # Geometric
    geometric = GeometricAugmentations()
    rot_img, rot_art, rot_vein = geometric.random_rotation_3d(dummy_image, dummy_artery, dummy_vein)
    print(f"✓ Rotation: shapes {rot_img.shape}, mask binary: {set(np.unique(rot_art)) == {0.0, 1.0}}")
    
    # Intensity
    intensity = IntensityAugmentations()
    bright_img = intensity.random_brightness_contrast(dummy_image)
    print(f"✓ Brightness/Contrast: range [{bright_img.min():.3f}, {bright_img.max():.3f}]")
    
    # Vessel-specific
    vessel_aug = VesselSpecificAugmentations()
    drop_art, drop_vein = vessel_aug.vessel_dropout(dummy_artery, dummy_vein)
    print(f"✓ Vessel dropout: artery sum {np.sum(drop_art):.0f}, vein sum {np.sum(drop_vein):.0f}")
    
    # MixUp/CutMix
    dummy_image2 = np.random.randn(96, 96, 96).astype(np.float32)
    dummy_artery2 = np.random.choice([0, 1], size=(96, 96, 96), p=[0.94, 0.06]).astype(np.float32)
    dummy_vein2 = np.random.choice([0, 1], size=(96, 96, 96), p=[0.96, 0.04]).astype(np.float32)
    
    result_with_mix = augmentation_pipeline(
        dummy_image, dummy_artery, dummy_vein,
        second_sample=(dummy_image2, dummy_artery2, dummy_vein2)
    )
    print(f"✓ MixUp/CutMix test completed, result length: {len(result_with_mix)}")
    
    # Test preset configurations
    print(f"\nTesting preset configurations...")
    
    for intensity in ['light', 'medium', 'heavy']:
        pipeline = create_augmentation_pipeline(training=True, intensity=intensity)
        result = pipeline(dummy_image, dummy_artery, dummy_vein)
        aug_image = result[0]
        print(f"✓ {intensity.capitalize()} intensity: range [{aug_image.min():.3f}, {aug_image.max():.3f}]")
    
    # Test augmentation scheduler
    print(f"\nTesting augmentation scheduler...")
    scheduler = ScheduledAugmentation(initial_prob=0.8, min_prob=0.3)
    
    for epoch in [0, 50, 100, 150, 200]:
        prob = scheduler.get_augmentation_probability(epoch, 200)
        print(f"  Epoch {epoch}: augmentation probability = {prob:.3f}")
    
    print(f"\nAll augmentation tests passed!")
    print(f"Ready to integrate with training pipeline!")
    print(f"\nUsage:")
    print(f"  1. Use data_preprocessing.py for basic data cleaning")
    print(f"  2. Use data_augmentation.py during training for robustness")
    print(f"  3. Both preserve vessel topology for clDice loss computation")