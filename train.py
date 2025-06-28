import os
import sys
import time
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules with error handling
try:
    from hipas_dataset import HiPaSDataset, HiPaSCollateFunction
except ImportError as e:
    print(f"Warning: Could not import dataset modules: {e}")
    HiPaSDataset = None
    HiPaSCollateFunction = None

try:
    from data_augmentation import create_safe_augmentation_pipeline, create_validation_augmentation
except ImportError as e:
    print(f"Warning: Could not import augmentation modules: {e}")
    create_safe_augmentation_pipeline = None
    create_validation_augmentation = None

try:
    from unet3d import create_unet3d
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")
    create_unet3d = None

try:
    from dice_loss import BaselineLossFactory
except ImportError as e:
    print(f"Warning: Could not import dice loss modules: {e}")
    BaselineLossFactory = None

try:
    from cldice_loss import create_topology_loss
except ImportError as e:
    print(f"Warning: Could not import cldice loss modules: {e}")
    create_topology_loss = None

try:
    from connectivity_metrics import MetricsCalculator
except ImportError as e:
    print(f"Warning: Could not import metrics modules: {e}")
    MetricsCalculator = None

try:
    from skeleton_postprocessing import PostProcessingPipeline
except ImportError as e:
    print(f"Warning: Could not import postprocessing modules: {e}")
    PostProcessingPipeline = None


class VesselSegmentationTrainer:
    """
    Complete training pipeline for vessel segmentation with clDice and skeleton repair
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config (dict): Training configuration
        """
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._setup_directories()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_metrics()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.patience_counter = 0
        
        print(f"🚀 Trainer initialized successfully!")
        print(f"   Device: {self.device}")
        print(f"   Model: {config['model']['type']}")
        print(f"   Loss: {config['loss']['type']}")
        print(f"   Dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def _setup_directories(self):
        """Setup output directories"""
        self.output_dir = self.config['output_dir']
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _setup_data(self):
        """Setup datasets and data loaders"""
        if HiPaSDataset is None or HiPaSCollateFunction is None:
            raise ImportError("Dataset modules not available")
        
        data_config = self.config['data']
        
        # 🔧 修改：使用正确的增强管道
        if create_safe_augmentation_pipeline is not None:
            self.augmentation = create_safe_augmentation_pipeline(
                training=True,
                intensity=data_config.get('augmentation_intensity', 'light')
            )
        else:
            print("⚠️ Augmentation pipeline not available, using None")
            self.augmentation = None
        
        # 🔧 修改：验证时不使用增强
        if create_validation_augmentation is not None:
            self.val_augmentation = create_validation_augmentation()
        else:
            self.val_augmentation = None
        
        # Create datasets
        self.train_dataset = HiPaSDataset(
            data_dir=data_config['data_dir'],
            split='train',
            transform=self.augmentation,
            target_size=tuple(data_config['target_size']),
            window_level=tuple(data_config['window_level'])
        )
        
        self.val_dataset = HiPaSDataset(
            data_dir=data_config['data_dir'],
            split='val',
            transform=self.val_augmentation,  # 🔧 修改：使用验证专用的转换
            target_size=tuple(data_config['target_size']),
            window_level=tuple(data_config['window_level'])
        )
        
        # Create data loaders
        collate_fn = HiPaSCollateFunction()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True  # 🔧 添加：避免单样本批次问题
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=data_config['val_batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        print(f"📊 Data loaded:")
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
    
    def _setup_model(self):
        """Setup model architecture"""
        if create_unet3d is None:
            raise ImportError("Model creation function not available")
        
        model_config = self.config['model']
        
        # Create model
        if model_config['type'] == 'unet3d':
            self.model = create_unet3d(
                model_type=model_config.get('variant', 'standard'),
                in_channels=model_config['in_channels'],
                num_classes=model_config['num_classes'],
                features=model_config.get('features', [32, 64, 128, 256, 512]),
                bilinear=model_config.get('bilinear', True),
                dropout=model_config.get('dropout', 0.1),
                attention=model_config.get('attention', False),
                deep_supervision=model_config.get('deep_supervision', False)
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🏗️ Model created:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Load pretrained weights if specified
        if model_config.get('pretrained_path'):
            self._load_pretrained_weights(model_config['pretrained_path'])
    
    def _setup_loss(self):
        """Setup loss function"""
        loss_config = self.config['loss']
        
        if loss_config['type'] == 'baseline':
            # 🔧 修改：使用正确的基线损失
            if BaselineLossFactory is None:
                raise ImportError("Baseline loss factory not available")
            
            self.criterion = BaselineLossFactory.create_loss(
                loss_type='dice_bce',
                **loss_config.get('params', {})
            )
        
        elif loss_config['type'] in ['cldice', 'combined', 'adaptive']:
            # 🔧 修改：使用正确的拓扑损失，过滤不支持的参数
            if create_topology_loss is None:
                raise ImportError("Topology loss creation function not available")
            
            # 准备损失函数参数，只传递支持的参数
            loss_kwargs = {
                'loss_type': loss_config['type'],
                'dice_weight': loss_config.get('dice_weight', 1.0),
                'cldice_weight': loss_config.get('cldice_weight', 1.0),
            }
            
            # 添加 params 中的参数
            if 'params' in loss_config:
                loss_kwargs.update(loss_config['params'])
            
            # 🔧 关键修复：不传递可能不支持的参数
            # 检查并移除可能导致错误的参数
            unsupported_params = ['alpha', 'beta']
            for param in unsupported_params:
                if param in loss_kwargs:
                    print(f"   Removing unsupported parameter: {param}={loss_kwargs[param]}")
                    del loss_kwargs[param]
            
            try:
                self.criterion = create_topology_loss(**loss_kwargs)
            except TypeError as e:
                print(f"⚠️ Loss creation failed with full params: {e}")
                # 回退到最小参数集
                minimal_kwargs = {
                    'loss_type': loss_config['type'],
                    'dice_weight': loss_config.get('dice_weight', 1.0),
                    'cldice_weight': loss_config.get('cldice_weight', 1.0),
                }
                print(f"   Retrying with minimal parameters: {minimal_kwargs}")
                self.criterion = create_topology_loss(**minimal_kwargs)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_config['type']}")
        
        print(f"📉 Loss function: {loss_config['type']}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        optim_config = self.config['optimizer']
        
        # Create optimizer
        if optim_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_config['lr'],
                weight_decay=optim_config.get('weight_decay', 0),
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optim_config['lr'],
                weight_decay=optim_config.get('weight_decay', 0.01)
            )
        elif optim_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optim_config['lr'],
                momentum=optim_config.get('momentum', 0.9),
                weight_decay=optim_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optim_config['type']}")
        
        # Create scheduler
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config.get('type') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_config.get('type') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        print(f"⚙️ Optimizer: {optim_config['type']}")
        print(f"   Learning rate: {optim_config['lr']}")
        if self.scheduler:
            print(f"   Scheduler: {scheduler_config['type']}")
    
    def _setup_metrics(self):
        """Setup metrics calculator"""
        if MetricsCalculator is None:
            print("⚠️ Metrics calculator not available")
            self.metrics_calculator = None
            return
        
        self.metrics_calculator = MetricsCalculator(
            class_names=['artery', 'vein'],
            smooth=1e-6
        )
        
        # Configure metrics computation
        metrics_config = self.config.get('metrics', {})
        self.compute_topology_metrics = metrics_config.get('include_topology', True)
        self.compute_branch_metrics = metrics_config.get('include_branch', False)
        
        print(f"📊 Metrics:")
        print(f"   Topology metrics: {self.compute_topology_metrics}")
        print(f"   Branch metrics: {self.compute_branch_metrics}")

    def _setup_logging(self):
        """Setup logging and monitoring"""
        # ✅ 强制转换为绝对路径 + unicode string
        self.log_dir = os.path.abspath(self.log_dir)
        self.log_dir = str(self.log_dir)

        # ✅ 确保目录存在
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            for _ in range(5):
                if os.path.exists(self.log_dir):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError(f"Log directory {self.log_dir} was not created in time.")
            print(f"📝 Log directory confirmed: {self.log_dir}")
        except Exception as e:
            print(f"📝 Failed to create or verify log directory: {e}")
            raise e

        # ✅ 初始化 SummaryWriter（使用绝对路径）
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir, max_queue=1, flush_secs=1)
            print(f"📝 TensorBoard writer initialized at: {self.log_dir}")
        except Exception as e:
            print(f"📝 Failed to initialize SummaryWriter: {e}")
            raise e

        # ✅ 初始化历史记录
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_cldice': [],
            'val_connectivity': [],
            'learning_rate': []
        }

        print(f"📝 Logging setup completed.")
    
    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained model weights"""
        if os.path.exists(pretrained_path):
            print(f"🔄 Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"✅ Pretrained weights loaded successfully")
        else:
            print(f"⚠️ Pretrained weights not found at {pretrained_path}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Set augmentation training mode
        if hasattr(self.augmentation, 'set_training'):
            self.augmentation.set_training(True)
        
        # Update adaptive loss epoch if applicable
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(self.current_epoch)
        
        epoch_losses = []
        epoch_metrics = []
        
        # Progress tracking
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 🔧 添加：检查批次是否有效
            if batch is None:
                print(f"⚠️ Skipping None batch at index {batch_idx}")
                continue
            
            try:
                # Move data to device
                images = batch['images'].to(self.device, non_blocking=True)
                artery_masks = batch['artery_masks'].to(self.device, non_blocking=True)
                vein_masks = batch['vein_masks'].to(self.device, non_blocking=True)
                
                # Stack targets: [B, 2, D, H, W] (artery, vein)
                targets = torch.stack([artery_masks, vein_masks], dim=1)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                predictions = outputs['main'] if isinstance(outputs, dict) else outputs
                
                # Calculate loss
                loss_result = self.criterion(predictions, targets)
                
                if isinstance(loss_result, dict):
                    total_loss = loss_result['total_loss']
                    loss_components = {k: v.item() for k, v in loss_result.items() if isinstance(v, torch.Tensor)}
                else:
                    total_loss = loss_result
                    loss_components = {'total_loss': total_loss.item()}
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['grad_clip_norm']
                    )
                
                self.optimizer.step()
                
                # Record loss
                epoch_losses.append(loss_components)
                
                # Calculate metrics (less frequently to save time)
                if batch_idx % self.config['training'].get('metric_freq', 10) == 0:
                    if self.metrics_calculator is not None:
                        with torch.no_grad():
                            batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                                predictions, targets,
                                include_topology=False,  # Skip topology metrics during training for speed
                                include_branch=False
                            )
                            epoch_metrics.append(batch_metrics)
                
                # Log progress
                if batch_idx % self.config['training'].get('log_freq', 50) == 0:
                    progress = batch_idx / num_batches * 100
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    print(f"Epoch {self.current_epoch} [{progress:6.1f}%] "
                          f"Loss: {total_loss.item():.4f} "
                          f"LR: {current_lr:.2e}")
                    
                    # Log to tensorboard
                    step = self.current_epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/Loss_Step', total_loss.item(), step)
                    self.writer.add_scalar('Train/LR', current_lr, step)
                    
                    # Log loss components
                    for component, value in loss_components.items():
                        self.writer.add_scalar(f'Train/{component}_Step', value, step)
            
            except Exception as e:
                print(f"⚠️ Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch statistics
        if epoch_losses:
            avg_loss = np.mean([loss['total_loss'] for loss in epoch_losses])
        else:
            avg_loss = float('inf')
        
        # Aggregate metrics if any were computed
        if epoch_metrics and self.metrics_calculator is not None:
            aggregated_metrics = self.metrics_calculator.aggregate_metrics(epoch_metrics)
            avg_dice = aggregated_metrics.get('dice_mean', 0.0)
        else:
            avg_dice = 0.0
        
        return {
            'avg_loss': avg_loss,
            'avg_dice': avg_dice,
            'loss_components': epoch_losses
        }
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 🔧 添加：检查批次是否有效
                if batch is None:
                    print(f"⚠️ Skipping None validation batch at index {batch_idx}")
                    continue
                
                try:
                    # Move data to device
                    images = batch['images'].to(self.device, non_blocking=True)
                    artery_masks = batch['artery_masks'].to(self.device, non_blocking=True)
                    vein_masks = batch['vein_masks'].to(self.device, non_blocking=True)
                    
                    # Stack targets
                    targets = torch.stack([artery_masks, vein_masks], dim=1)
                    
                    # Forward pass
                    outputs = self.model(images)
                    predictions = outputs['main'] if isinstance(outputs, dict) else outputs
                    
                    # Calculate loss
                    loss_result = self.criterion(predictions, targets)
                    
                    if isinstance(loss_result, dict):
                        total_loss = loss_result['total_loss']
                        loss_components = {k: v.item() for k, v in loss_result.items() if isinstance(v, torch.Tensor)}
                    else:
                        total_loss = loss_result
                        loss_components = {'total_loss': total_loss.item()}
                    
                    val_losses.append(loss_components)
                    
                    # Apply post-processing if enabled
                    processed_predictions = predictions
                    if self.config.get('postprocessing', {}).get('enabled', False):
                        # 🔧 修改：正确的后处理调用
                        if PostProcessingPipeline is not None:
                            postproc_pipeline = PostProcessingPipeline(
                                skeleton_method=self.config['postprocessing'].get('skeleton_method', 'skeletonize_3d'),
                                repair_strategy=self.config['postprocessing'].get('repair_strategy', 'adaptive'),
                                apply_morphological_cleanup=self.config['postprocessing'].get('morphological_cleanup', True),
                                final_dilation_radius=self.config['postprocessing'].get('dilation_radius', 1)
                            )
                            
                            # 转换为numpy进行处理
                            pred_numpy = predictions.detach().cpu().numpy()
                            target_numpy = targets.detach().cpu().numpy()
                            
                            # 应用后处理
                            try:
                                postproc_result = postproc_pipeline(pred_numpy, target_numpy)
                                processed_predictions = torch.from_numpy(postproc_result['repaired_masks']).to(self.device)
                            except Exception as postproc_error:
                                print(f"⚠️ Post-processing failed: {postproc_error}")
                                # 使用原始预测
                                processed_predictions = predictions
                    
                    # Calculate comprehensive metrics
                    if self.metrics_calculator is not None:
                        batch_metrics = self.metrics_calculator.calculate_batch_metrics(
                            processed_predictions, targets,
                            include_topology=self.compute_topology_metrics,
                            include_branch=self.compute_branch_metrics
                        )
                        val_metrics.append(batch_metrics)
                    
                    # Progress update
                    if batch_idx % 10 == 0:
                        progress = batch_idx / len(self.val_loader) * 100
                        print(f"Validation [{progress:6.1f}%] Loss: {total_loss.item():.4f}")
                
                except Exception as e:
                    print(f"⚠️ Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Aggregate results
        if val_losses:
            avg_loss = np.mean([loss['total_loss'] for loss in val_losses])
        else:
            avg_loss = float('inf')
        
        if val_metrics and self.metrics_calculator is not None:
            aggregated_metrics = self.metrics_calculator.aggregate_metrics(val_metrics)
            # Extract key metrics
            key_metrics = self.metrics_calculator.get_key_metrics_for_logging(aggregated_metrics)
            # Calculate combined score for model selection
            combined_score = self.metrics_calculator.calculate_combined_score(
                aggregated_metrics,
                weights=self.config.get('model_selection', {}).get('weights')
            )
        else:
            aggregated_metrics = {}
            key_metrics = {'dice_mean': 0.0}
            combined_score = 0.0
        
        return {
            'avg_loss': avg_loss,
            'metrics': aggregated_metrics,
            'key_metrics': key_metrics,
            'combined_score': combined_score
        }
    
    def save_checkpoint(self, is_best=False, additional_info=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config,
            'train_history': self.train_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved! Score: {self.best_score:.4f}")
        
        # Save epoch checkpoint
        if self.current_epoch % self.config['training'].get('save_freq', 50) == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{self.current_epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_score = checkpoint['best_score']
            self.train_history = checkpoint.get('train_history', self.train_history)
            
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Resuming from epoch {self.current_epoch}")
            print(f"   Best score: {self.best_score:.4f}")
            
            return True
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}")
            return False
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {self.config['training']['epochs']}")
        print(f"   Early stopping patience: {self.config['training'].get('early_stopping_patience', 'disabled')}")
        print(f"   Device: {self.device}")
        
        start_epoch = self.current_epoch
        total_epochs = self.config['training']['epochs']
        
        # Resume from checkpoint if specified
        if self.config.get('resume_from'):
            self.load_checkpoint(self.config['resume_from'])
        
        try:
            for epoch in range(start_epoch, total_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                print(f"\nEpoch {epoch+1}/{total_epochs}")
                print("-" * 50)
                
                # Training phase
                train_results = self.train_epoch()
                
                # Validation phase
                val_results = self.validate_epoch()
                
                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_results['combined_score'])
                    else:
                        self.scheduler.step()
                
                # Log results
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start_time
                
                # Update history
                self.train_history['epoch'].append(epoch)
                self.train_history['train_loss'].append(train_results['avg_loss'])
                self.train_history['val_loss'].append(val_results['avg_loss'])
                self.train_history['val_dice'].append(val_results['key_metrics'].get('dice_mean', 0))
                self.train_history['val_cldice'].append(val_results['key_metrics'].get('centerline_dice_mean', 0))
                self.train_history['val_connectivity'].append(val_results['key_metrics'].get('connectivity_accuracy_mean', 0))
                self.train_history['learning_rate'].append(current_lr)
                
                # Print results
                print(f"\n📊 Epoch {epoch+1} Results:")
                print(f"   Train Loss: {train_results['avg_loss']:.4f}")
                print(f"   Val Loss: {val_results['avg_loss']:.4f}")
                print(f"   Val Dice: {val_results['key_metrics'].get('dice_mean', 0):.4f}")
                if 'centerline_dice_mean' in val_results['key_metrics']:
                    print(f"   Val clDice: {val_results['key_metrics']['centerline_dice_mean']:.4f}")
                if 'connectivity_accuracy_mean' in val_results['key_metrics']:
                    print(f"   Val Connectivity: {val_results['key_metrics']['connectivity_accuracy_mean']:.4f}")
                print(f"   Combined Score: {val_results['combined_score']:.4f}")
                print(f"   Learning Rate: {current_lr:.2e}")
                print(f"   Time: {epoch_time:.1f}s")
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Train/Loss_Epoch', train_results['avg_loss'], epoch)
                    self.writer.add_scalar('Val/Loss_Epoch', val_results['avg_loss'], epoch)
                    self.writer.add_scalar('Val/Combined_Score', val_results['combined_score'], epoch)
                    self.writer.add_scalar('Train/LR_Epoch', current_lr, epoch)
                    
                    # Log key metrics
                    for metric_name, value in val_results['key_metrics'].items():
                        self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
                
                # Model selection and checkpointing
                is_best = val_results['combined_score'] > self.best_score
                if is_best:
                    self.best_score = val_results['combined_score']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(
                    is_best=is_best,
                    additional_info={
                        'val_results': val_results,
                        'train_results': train_results
                    }
                )
                
                # Early stopping
                early_stopping_patience = self.config['training'].get('early_stopping_patience')
                if early_stopping_patience and self.patience_counter >= early_stopping_patience:
                    print(f"\n⏹️ Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    break
                
                # Memory cleanup
                torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Training interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Final save and cleanup
            self.save_checkpoint(is_best=False)
            if self.writer:
                self.writer.close()
            
            print(f"\n🎉 Training completed!")
            print(f"   Best score: {self.best_score:.4f}")
            print(f"   Total epochs: {self.current_epoch + 1}")
            print(f"   Checkpoints saved to: {self.checkpoint_dir}")


def create_default_config():
    """Create default training configuration"""
    return {
        # Data configuration - 与 hipas_dataset.py 兼容
        'data': {
            'data_dir': './dataset',
            'target_size': [96, 96, 96],  # RTX 3060 友好的尺寸
            'window_level': [-600, 1500],
            'batch_size': 1,              # 🔧 修改：更安全的批次大小
            'val_batch_size': 1,
            'num_workers': 2,
            'augmentation_intensity': 'light'  # 🔧 添加：增强强度控制
        },
        
        # Model configuration - 与 unet3d.py 兼容
        'model': {
            'type': 'unet3d',
            'variant': 'standard',
            'in_channels': 1,
            'num_classes': 2,
            'features': [32, 64, 128, 256, 512],  # 🔧 修改：更适中的特征数
            'bilinear': True,
            'dropout': 0.1,
            'attention': False,
            'deep_supervision': False,
            'pretrained_path': None
        },
        
        # Loss configuration - 与 cldice_loss.py 兼容
        'loss': {
            'type': 'combined',
            'dice_weight': 1.0,
            'cldice_weight': 1.0,
            'params': {
                'smooth': 1e-6,
                'use_soft_skeleton': True,
                'num_iter': 40           # 🔧 添加：软骨架迭代次数
            }
        },
        
        # Optimizer configuration
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },
        
        # Scheduler configuration
        'scheduler': {
            'type': 'cosine',  # 'cosine', 'plateau', 'step', None
            'min_lr': 1e-6,
            'factor': 0.5,
            'patience': 10
        },
        
        # Training configuration
        'training': {
            'epochs': 200,
            'early_stopping_patience': 30,
            'grad_clip_norm': 1.0,
            'log_freq': 20,
            'metric_freq': 10,
            'save_freq': 50
        },
        
        # Metrics configuration
        'metrics': {
            'include_topology': True,
            'include_branch': False
        },
        
        # Model selection configuration
        'model_selection': {
            'weights': {
                'dice_mean': 0.3,
                'centerline_dice_mean': 0.3,
                'connectivity_accuracy_mean': 0.2,
                'iou_mean': 0.2
            }
        },
        
        # Post-processing configuration
        'postprocessing': {
            'enabled': False,
            'skeleton_method': 'skeletonize_3d',
            'repair_strategy': 'adaptive',
            'morphological_cleanup': True,
            'dilation_radius': 1
        },
        
        # System configuration
        'device': 'cuda',
        'output_dir': './outputs',
        'resume_from': None
    }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train 3D Vessel Segmentation Model')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['baseline', 'cldice', 'combined', 'adaptive'],
                        help='Loss function type')
    parser.add_argument('--model_variant', type=str, default='standard',
                        choices=['standard', 'attention'],
                        help='Model variant')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--postprocessing', action='store_true',
                        help='Enable skeleton post-processing during validation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with smaller dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Weight for Dice loss')
    parser.add_argument('--cldice_weight', type=float, default=1.0, help='Weight for clDice loss')

    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    if config_path and os.path.exists(config_path):
        print(f"📁 Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"📁 Using default configuration")
        return create_default_config()


def update_config_from_args(config, args):
    """Update configuration with command line arguments"""
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.resume_from:
        config['resume_from'] = args.resume_from
    
    if args.loss_type:
        config['loss']['type'] = args.loss_type
    
    if args.model_variant:
        config['model']['variant'] = args.model_variant
    
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    if args.lr:
        config['optimizer']['lr'] = args.lr
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.early_stopping:
        config['training']['early_stopping_patience'] = args.early_stopping
    
    if args.postprocessing:
        config['postprocessing']['enabled'] = True
    
    if args.device:
        config['device'] = args.device
    
    if args.debug:
        # Debug mode: smaller batches, fewer epochs, more frequent logging
        config['data']['batch_size'] = 1
        config['data']['val_batch_size'] = 1
        config['training']['epochs'] = 10
        config['training']['log_freq'] = 5
        config['training']['metric_freq'] = 5
        config['training']['save_freq'] = 5
        config['training']['early_stopping_patience'] = 5
        config['data']['augmentation_intensity'] = 'light'
        print("🐛 Debug mode enabled")
    
    return config


def setup_experiment_name(config):
    """Generate experiment name based on configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive name
    name_parts = [
        config['model']['type'],
        config['model']['variant'],
        config['loss']['type']
    ]
    
    if config['loss']['type'] == 'combined':
        name_parts.append(f"d{config['loss']['dice_weight']}_c{config['loss']['cldice_weight']}")
    
    if config['model'].get('attention'):
        name_parts.append('att')
    
    if config.get('postprocessing', {}).get('enabled', False):
        name_parts.append('postproc')
    
    experiment_name = "_".join(name_parts) + f"_{timestamp}"
    
    # Update output directory
    config['output_dir'] = os.path.join(config['output_dir'], experiment_name)
    
    return experiment_name


def validate_config(config):
    """Validate configuration"""
    errors = []
    
    # Check data directory
    if not os.path.exists(config['data']['data_dir']):
        errors.append(f"Data directory not found: {config['data']['data_dir']}")
    
    # Check required keys
    required_keys = [
        'data', 'model', 'loss', 'optimizer', 'training'
    ]
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")
    
    # Check model parameters
    if config['model'].get('num_classes', 2) != 2:
        errors.append("num_classes must be 2 for artery/vein segmentation")
    
    # Check loss type compatibility
    loss_type = config['loss']['type']
    if loss_type not in ['baseline', 'cldice', 'combined', 'adaptive']:
        errors.append(f"Unknown loss type: {loss_type}")
    
    # Check device availability
    if config.get('device') == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        config['device'] = 'cpu'
    
    # Check batch size with GPU memory
    if config.get('device') == 'cuda':
        batch_size = config['data']['batch_size']
        target_size = config['data']['target_size']
        
        # Rough memory estimation (in GB)
        estimated_memory = batch_size * np.prod(target_size) * 4 / (1024**3)  # 4 bytes per float32
        estimated_memory *= 10  # Factor for gradients, activations, etc.
        
        if estimated_memory > 8:  # Assume 8GB GPU
            print(f"⚠️ High memory usage estimated: {estimated_memory:.1f}GB")
            print(f"   Consider reducing batch size or target size")
    
    if errors:
        for error in errors:
            print(f"❌ Configuration error: {error}")
        raise ValueError("Configuration validation failed")
    
    print("✅ Configuration validated successfully")


def print_experiment_info(config, experiment_name):
    """Print experiment information"""
    print("\n" + "="*70)
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print("="*70)
    print(f"📁 Output directory: {config['output_dir']}")
    print(f"📊 Dataset: {config['data']['data_dir']}")
    print(f"🏗️ Model: {config['model']['type']} ({config['model']['variant']})")
    print(f"📉 Loss: {config['loss']['type']}")
    
    if config['loss']['type'] == 'combined':
        print(f"   - Dice weight: {config['loss']['dice_weight']}")
        print(f"   - clDice weight: {config['loss']['cldice_weight']}")
    
    print(f"⚙️ Optimizer: {config['optimizer']['type']} (lr={config['optimizer']['lr']})")
    print(f"📈 Scheduler: {config.get('scheduler', {}).get('type', 'None')}")
    print(f"🔄 Epochs: {config['training']['epochs']}")
    print(f"📦 Batch size: {config['data']['batch_size']}")
    print(f"📏 Target size: {config['data']['target_size']}")
    
    post_enabled = config.get('postprocessing', {}).get('enabled', False)
    post_method = config.get('postprocessing', {}).get('repair_strategy', 'N/A')
    print(f"🔧 Post-processing: {post_enabled} (Strategy: {post_method})")
    print(f"💻 Device: {config['device']}")
    print("="*70)


def main():
    """Main training function"""
    print("🧠 3D Vessel Segmentation Training Pipeline")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load and update configuration
        config = load_config(args.config)
        config = update_config_from_args(config, args)
        
        # Setup experiment
        experiment_name = setup_experiment_name(config)
        
        # Validate configuration
        validate_config(config)
        
        # Print experiment info
        print_experiment_info(config, experiment_name)
        
        # Create trainer and start training
        trainer = VesselSegmentationTrainer(config)
        trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config['output_dir']}")
        print(f"🏆 Best score achieved: {trainer.best_score:.4f}")
        
        # Print final metrics summary
        if trainer.train_history['val_dice']:
            final_dice = trainer.train_history['val_dice'][-1]
            final_cldice = trainer.train_history['val_cldice'][-1]
            final_connectivity = trainer.train_history['val_connectivity'][-1]
            
            print(f"\n📊 Final Validation Metrics:")
            print(f"   Dice Score: {final_dice:.4f}")
            print(f"   clDice Score: {final_cldice:.4f}")
            print(f"   Connectivity Accuracy: {final_connectivity:.4f}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Additional utility functions for advanced usage
def create_experiment_configs():
    """Create predefined experiment configurations for paper"""
    configs = {}
    
    # Baseline experiment
    configs['baseline'] = create_default_config()
    configs['baseline']['loss']['type'] = 'baseline'
    configs['baseline']['loss']['params'] = {'loss_type': 'dice_bce'}
    
    # clDice experiment
    configs['cldice'] = create_default_config()
    configs['cldice']['loss']['type'] = 'cldice'
    
    # Combined experiment (main proposal)
    configs['combined'] = create_default_config()
    configs['combined']['loss']['type'] = 'combined'
    configs['combined']['loss']['dice_weight'] = 1.0
    configs['combined']['loss']['cldice_weight'] = 1.0
    
    # Combined with post-processing (full method)
    configs['full_method'] = create_default_config()
    configs['full_method']['loss']['type'] = 'combined'
    configs['full_method']['postprocessing']['enabled'] = True
    configs['full_method']['postprocessing']['repair_strategy'] = 'adaptive'
    
    # Attention U-Net variant
    configs['attention'] = create_default_config()
    configs['attention']['model']['variant'] = 'attention'
    configs['attention']['model']['attention'] = True
    configs['attention']['loss']['type'] = 'combined'
    
    return configs


def save_experiment_configs(output_dir='./configs'):
    """Save predefined experiment configurations"""
    configs = create_experiment_configs()
    os.makedirs(output_dir, exist_ok=True)
    
    for name, config in configs.items():
        config_path = os.path.join(output_dir, f'{name}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Saved config: {config_path}")


# Example usage and testing
if __name__ == '__main__':
    if len(sys.argv) == 1:
        # If run without arguments, show help and create example configs
        print("🧠 3D Vessel Segmentation Training System")
        print("=" * 60)
        print("")
        print("🚀 Quick Start:")
        print("  python train.py                                    # Train with default config")
        print("  python train.py --config configs/baseline.json     # Train baseline method")
        print("  python train.py --config configs/combined.json     # Train combined method")
        print("  python train.py --loss_type cldice --epochs 100    # Custom training")
        print("  python train.py --debug                            # Quick test run")
        print("")
        print("📋 Available Loss Functions:")
        print("  • baseline: Standard Dice + BCE loss")
        print("  • cldice: Topology-aware clDice loss")
        print("  • combined: Dice + clDice (main proposal)")
        print("  • adaptive: Adaptive clDice weighting")
        print("")
        print("🏗️ Model Variants:")
        print("  • standard: Basic 3D U-Net")
        print("  • attention: Attention-enhanced U-Net")
        print("")
        print("⚙️ Key Features:")
        print("  ✓ Topology-aware loss functions")
        print("  ✓ Skeleton-based post-processing")
        print("  ✓ Comprehensive metrics tracking")
        print("  ✓ TensorBoard logging")
        print("  ✓ Automatic checkpointing")
        print("  ✓ Early stopping")
        print("")
        print("📁 Creating example configurations...")
        try:
            save_experiment_configs()
            print("✅ Example configs created in ./configs/")
        except Exception as e:
            print(f"⚠️ Could not create configs: {e}")
        print("")
        print("🔗 Integration:")
        print("  • Works with hipas_dataset.py for data loading")
        print("  • Compatible with data_augmentation.py")
        print("  • Integrates with skeleton_postprocessing.py")
        print("  • Uses connectivity_metrics.py for evaluation")
        print("")
        print("💡 Tips:")
        print("  • Use --debug for quick testing")
        print("  • Monitor with: tensorboard --logdir ./outputs")
        print("  • Resume with: --resume_from path/to/checkpoint.pth")
        print("  • Enable post-processing with: --postprocessing")
        print("  • Start with batch_size=1 for RTX 3060")
    else:
        main()