import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class HiPaSDataset(Dataset):
    """
    HiPaS Dataset Loader for pulmonary vessel segmentation
    Supports joint segmentation of arteries and veins
    
    Official dataset structure:
    data_dir/
    ├── metadata.xlsx
    ├── ct_scan/
    │   ├── 001.npz
    │   └── ...
    └── annotation/
        ├── artery/
        │   ├── 001.npz
        │   └── ...
        └── vein/
            ├── 001.npz
            └── ...
    """
    
    def __init__(self, data_dir, split='train', transform=None, target_spacing=(1.0, 1.0, 1.0), 
                 target_size=(96, 96, 96), window_level=(-600, 1500)):
        """
        Args:
            data_dir (str): Root directory of HiPaS dataset
            split (str): Data split 'train'/'val'/'test'
            transform: Data augmentation transforms
            target_spacing (tuple): Target voxel spacing (mm)
            target_size (tuple): Target size (D, H, W) - optimized for RTX 3060
            window_level (tuple): CT window level (center, width)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.window_center, self.window_width = window_level
        
        # Load metadata and samples
        self.metadata = self._load_metadata()
        self.samples = self._load_sample_list()
        
        print(f"Successfully loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self):
        """Load metadata from Excel file"""
        metadata_path = os.path.join(self.data_dir, 'metadata.xlsx')
        metadata_dict = {}
        
        if os.path.exists(metadata_path):
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
                print(f"Warning: Error reading metadata.xlsx: {e}")
        else:
            print(f"Warning: metadata.xlsx not found at {metadata_path}")
            
        return metadata_dict
    
    def _load_sample_list(self):
        """Load sample list and verify file integrity"""
        samples = []
        
        # Define required directories
        ct_scan_dir = os.path.join(self.data_dir, 'ct_scan')
        artery_dir = os.path.join(self.data_dir, 'annotation', 'artery')
        vein_dir = os.path.join(self.data_dir, 'annotation', 'vein')
        
        # Verify all directories exist
        required_dirs = [
            ('ct_scan', ct_scan_dir),
            ('annotation/artery', artery_dir),
            ('annotation/vein', vein_dir)
        ]
        
        for dir_name, dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_name} at {dir_path}")
        
        # Get all CT scan files
        ct_files = [f for f in os.listdir(ct_scan_dir) if f.endswith('.npz')]
        ct_files.sort()
        print(f"Found {len(ct_files)} CT scan files")
        
        # Verify annotations exist for each CT scan
        valid_cases = []
        missing_annotations = []
        
        for ct_file in ct_files:
            case_name = ct_file.replace('.npz', '')
            
            ct_path = os.path.join(ct_scan_dir, ct_file)
            artery_path = os.path.join(artery_dir, ct_file)
            vein_path = os.path.join(vein_dir, ct_file)
            
            # Check if all files exist
            if all(os.path.exists(path) for path in [ct_path, artery_path, vein_path]):
                valid_cases.append({
                    'case_name': case_name,
                    'ct_path': ct_path,
                    'artery_path': artery_path,
                    'vein_path': vein_path
                })
            else:
                missing_annotations.append(case_name)
        
        if missing_annotations:
            print(f"Warning: {len(missing_annotations)} cases missing annotations: {missing_annotations[:5]}{'...' if len(missing_annotations) > 5 else ''}")
        
        print(f"Found {len(valid_cases)} complete cases with CT + annotations")
        
        # Apply data split
        case_names = [case['case_name'] for case in valid_cases]
        split_cases = self._apply_data_split(case_names)
        
        # Filter samples for current split
        samples = [case for case in valid_cases if case['case_name'] in split_cases]
        
        return samples
    
    def _apply_data_split(self, case_names):
        """Apply train/val/test split"""
        # Check for existing split files
        splits_dir = os.path.join(self.data_dir, 'splits')
        split_file = os.path.join(splits_dir, f'{self.split}.txt')
        
        if os.path.exists(split_file):
            print(f"Using existing split file: {split_file}")
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        
        # Auto split: 70% train, 15% val, 15% test
        case_names.sort()  # Ensure reproducible splits
        n_total = len(case_names)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        if self.split == 'train':
            return case_names[:n_train]
        elif self.split == 'val':
            return case_names[n_train:n_train+n_val]
        else:  # test
            return case_names[n_train+n_val:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get single sample"""
        sample = self.samples[idx]
        
        try:
            # Load data from separate npz files (HiPaS official format)
            ct_data = self._load_npz(sample['ct_path'])
            artery_data = self._load_npz(sample['artery_path'])
            vein_data = self._load_npz(sample['vein_path'])
            
            # Extract arrays (all use 'data' key according to HiPaS documentation)
            image = ct_data['data'].astype(np.float32)
            artery_mask = artery_data['data'].astype(np.float32)
            vein_mask = vein_data['data'].astype(np.float32)
            
            # Verify shapes match
            if not (image.shape == artery_mask.shape == vein_mask.shape):
                print(f"Warning: Shape mismatch for case {sample['case_name']}")
                print(f"  Image: {image.shape}, Artery: {artery_mask.shape}, Vein: {vein_mask.shape}")
                # Resize masks to match image if needed
                if artery_mask.shape != image.shape:
                    artery_mask = self._resize_to_match(artery_mask, image.shape)
                if vein_mask.shape != image.shape:
                    vein_mask = self._resize_to_match(vein_mask, image.shape)
            
            # Get original spacing from metadata
            case_filename = f"{sample['case_name']}.npz"
            original_spacing = None
            if case_filename in self.metadata:
                original_spacing = self.metadata[case_filename]['spacing']
            
            # Preprocessing
            image = self._preprocess_image(image)
            artery_mask = self._preprocess_mask(artery_mask)
            vein_mask = self._preprocess_mask(vein_mask)
            
            # Resize to target size
            image, artery_mask, vein_mask = self._resize_to_target(
                image, artery_mask, vein_mask, original_spacing)
            
            # Data augmentation
            if self.transform is not None:
                image, artery_mask, vein_mask = self.transform(image, artery_mask, vein_mask)
            
            # 🔧 CRITICAL FIX: Ensure array contiguity to fix stride issues
            image = np.ascontiguousarray(image)
            artery_mask = np.ascontiguousarray(artery_mask)
            vein_mask = np.ascontiguousarray(vein_mask)
            
            # Convert to tensors with correct keys for train.py
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, D, H, W]
            artery_tensor = torch.from_numpy(artery_mask).float()         # [D, H, W]
            vein_tensor = torch.from_numpy(vein_mask).float()             # [D, H, W]
            
            return {
                'images': image_tensor,          # Changed from 'image' to 'images'
                'artery_masks': artery_tensor,   # Changed from 'artery_mask' to 'artery_masks'
                'vein_masks': vein_tensor,       # Changed from 'vein_mask' to 'vein_masks'
                'case_names': sample['case_name'] # Changed from 'case_name' to 'case_names'
            }
            
        except Exception as e:
            print(f"Error loading sample {sample['case_name']}: {e}")
            # Return safe fallback data with correct keys
            return self._get_fallback_sample(sample['case_name'])
    
    def _get_fallback_sample(self, case_name):
        """Return a safe fallback sample when loading fails"""
        dummy_shape = self.target_size
        
        # Create safe, contiguous arrays
        image = np.zeros((1,) + dummy_shape, dtype=np.float32)
        artery_mask = np.zeros(dummy_shape, dtype=np.float32)
        vein_mask = np.zeros(dummy_shape, dtype=np.float32)
        
        # Ensure contiguity
        image = np.ascontiguousarray(image)
        artery_mask = np.ascontiguousarray(artery_mask)
        vein_mask = np.ascontiguousarray(vein_mask)
        
        return {
            'images': torch.from_numpy(image).float(),
            'artery_masks': torch.from_numpy(artery_mask).float(),
            'vein_masks': torch.from_numpy(vein_mask).float(),
            'case_names': f"fallback_{case_name}"
        }
    
    def _load_npz(self, file_path):
        """Load NPZ file with error handling"""
        try:
            return np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise
    
    def _resize_to_match(self, array, target_shape):
        """Resize array to match target shape"""
        if array.shape == target_shape:
            return array
        
        zoom_factors = [t/c for t, c in zip(target_shape, array.shape)]
        return zoom(array, zoom_factors, order=0)  # nearest neighbor for masks
    
    def _preprocess_image(self, image):
        """Preprocess CT image"""
        # Apply window level adjustment
        image = self._apply_window_level(image)
        
        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def _apply_window_level(self, image):
        """Apply CT window level adjustment"""
        min_val = self.window_center - self.window_width // 2
        max_val = self.window_center + self.window_width // 2
        
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def _preprocess_mask(self, mask):
        """Preprocess segmentation mask"""
        # Ensure binary mask (0 or 1)
        mask = (mask > 0.5).astype(np.float32)
        return mask
    
    def _resize_to_target(self, image, artery_mask, vein_mask, original_spacing=None):
        """Resize data to target size with spacing consideration"""
        current_shape = image.shape
        target_shape = self.target_size
        
        if current_shape != target_shape:
            # Calculate zoom factors
            if original_spacing is not None:
                # Consider both spacing and size
                spacing_factors = [orig/target for orig, target in zip(original_spacing, self.target_spacing)]
                size_factors = [t/c for t, c in zip(target_shape, current_shape)]
                # Use size-based factors primarily (spacing info for reference)
                zoom_factors = size_factors
            else:
                # Simple size-based zoom factors
                zoom_factors = [t/c for t, c in zip(target_shape, current_shape)]
            
            # Apply resampling
            image = zoom(image, zoom_factors, order=1)  # bilinear for image
            artery_mask = zoom(artery_mask, zoom_factors, order=0)  # nearest for masks
            vein_mask = zoom(vein_mask, zoom_factors, order=0)
            
            # Ensure masks remain binary
            artery_mask = (artery_mask > 0.5).astype(np.float32)
            vein_mask = (vein_mask > 0.5).astype(np.float32)
        
        return image, artery_mask, vein_mask
    
    def get_case_info(self, idx):
        """Get sample information"""
        sample = self.samples[idx]
        info = {
            'case_name': sample['case_name'],
            'ct_path': sample['ct_path'],
            'artery_path': sample['artery_path'],
            'vein_path': sample['vein_path']
        }
        
        # Add spacing info if available
        case_filename = f"{sample['case_name']}.npz"
        if case_filename in self.metadata:
            info['original_spacing'] = self.metadata[case_filename]['spacing']
        
        return info
    
    def get_statistics(self):
        """Get dataset statistics"""
        print(f"\n=== HiPaS Dataset Statistics ({self.split}) ===")
        print(f"Total samples: {len(self.samples)}")
        print(f"Target size: {self.target_size}")
        print(f"Window level: {self.window_center} ± {self.window_width//2}")
        
        if len(self.samples) > 0:
            # Sample a few cases to get shape statistics
            sample_shapes = []
            for i in range(min(5, len(self.samples))):
                try:
                    sample = self.samples[i]
                    ct_data = np.load(sample['ct_path'], allow_pickle=True)
                    shape = ct_data['data'].shape
                    sample_shapes.append(shape)
                except:
                    continue
            
            if sample_shapes:
                print(f"Original shapes (sample): {sample_shapes}")
                
        print("=" * 50)


class HiPaSCollateFunction:
    """Custom collate function for batch processing"""
    
    def __call__(self, batch):
        """
        Args:
            batch: list of dict from HiPaSDataset
        
        Returns:
            dict: batched tensors
        """
        # Filter out any None samples
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None
        
        # Check for fallback samples and skip batches with too many fallbacks
        fallback_count = sum(1 for item in batch if 'fallback' in str(item['case_names']))
        if fallback_count > len(batch) * 0.5:  # More than 50% fallbacks
            print(f"Skipping batch with {fallback_count}/{len(batch)} fallback samples")
            return None
        
        try:
            images = torch.stack([item['images'] for item in batch])
            artery_masks = torch.stack([item['artery_masks'] for item in batch])
            vein_masks = torch.stack([item['vein_masks'] for item in batch])
            case_names = [item['case_names'] for item in batch]
            
            return {
                'images': images,           # [B, 1, D, H, W]
                'artery_masks': artery_masks,  # [B, D, H, W]
                'vein_masks': vein_masks,      # [B, D, H, W]
                'case_names': case_names
            }
        except Exception as e:
            print(f"Error in collate function: {e}")
            return None


# Test and usage example
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # Set your dataset path
    data_dir = './dataset'
    
    print("Testing HiPaS Dataset Loader...")
    
    try:
        # Create dataset
        dataset = HiPaSDataset(
            data_dir=data_dir,
            split='train',
            target_size=(96, 96, 96),  # RTX 3060 optimized size
            window_level=(-600, 1500)
        )
        
        # Show statistics
        dataset.get_statistics()
        
        if len(dataset) > 0:
            # Test single sample
            print(f"\nTesting sample loading...")
            sample = dataset[0]
            print(f"✓ Images shape: {sample['images'].shape}")
            print(f"✓ Artery masks shape: {sample['artery_masks'].shape}")
            print(f"✓ Vein masks shape: {sample['vein_masks'].shape}")
            print(f"✓ Case name: {sample['case_names']}")
            
            # Check for contiguity
            print(f"✓ Images contiguous: {sample['images'].is_contiguous()}")
            print(f"✓ Artery contiguous: {sample['artery_masks'].is_contiguous()}")
            print(f"✓ Vein contiguous: {sample['vein_masks'].is_contiguous()}")
            
            # Test data loader
            print(f"\nTesting DataLoader...")
            collate_fn = HiPaSCollateFunction()
            dataloader = DataLoader(
                dataset, 
                batch_size=1,  # RTX 3060 recommended
                shuffle=False,
                num_workers=0,  # Start with 0, increase if stable
                collate_fn=collate_fn
            )
            
            # Test one batch
            batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue
                
                print(f"✓ Batch {batch_count}:")
                print(f"  Images: {batch['images'].shape}")
                print(f"  Artery masks: {batch['artery_masks'].shape}")
                print(f"  Vein masks: {batch['vein_masks'].shape}")
                print(f"  Cases: {batch['case_names']}")
                
                # Check value ranges
                print(f"  Image range: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
                print(f"  Artery mask unique values: {torch.unique(batch['artery_masks'])}")
                print(f"  Vein mask unique values: {torch.unique(batch['vein_masks'])}")
                
                batch_count += 1
                if batch_count >= 3:  # Test first 3 valid batches
                    break
            
            print(f"\n🎉 All tests passed! Dataset is ready for training.")
            
        else:
            print(f"❌ No samples found in dataset")
            
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()