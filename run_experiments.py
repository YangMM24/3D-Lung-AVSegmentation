import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
import shutil
import logging
import yaml
from typing import Dict, List, Optional, Any
import warnings
import traceback
warnings.filterwarnings('ignore')

# Import your custom modules with better error handling
try:
    from train import main as train_main
except ImportError as e:
    print(f"Warning: Could not import train module: {e}")
    train_main = None

try:
    from evaluate import main as evaluate_main
except ImportError as e:
    print(f"Warning: Could not import evaluate module: {e}")
    evaluate_main = None

try:
    from comparison_study import run_paper_comparison_study
except ImportError as e:
    print(f"Warning: Could not import comparison_study module: {e}")
    run_paper_comparison_study = None

try:
    from visualization import create_paper_figures_from_results
except ImportError as e:
    print(f"Warning: Could not import visualization module: {e}")
    create_paper_figures_from_results = None


class ExperimentRunner:
    """
    Comprehensive experiment runner for the paper:
    "Connectivity-aware 3D Segmentation of Pulmonary Arteries and Veins with Topology Repair"
    
    Manages the complete experimental pipeline from training to evaluation to comparison.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize experiment runner
        
        Args:
            config_path: Path to experiment configuration file
        """
        self.config = self._load_config(config_path)
        self.base_output_dir = Path(self.config['base_output_dir'])
        self.experiment_name = self.config['experiment_name']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.experiment_dir = self.base_output_dir / f"{self.experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Track experiment state
        self.completed_experiments = {}
        self.failed_experiments = {}
        
        self.logger.info(f"Experiment Runner Initialized")
        self.logger.info(f"   Experiment: {self.experiment_name}")
        self.logger.info(f"   Output directory: {self.experiment_dir}")
        self.logger.info(f"   Number of experiments: {len(self.config['experiments'])}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load experiment configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default experiment configuration for your paper"""
        return {
            'experiment_name': 'connectivity_aware_vessel_segmentation',
            'base_output_dir': './experiments',
            'data_config': {
                'data_dir': './dataset',
                'target_size': [96, 96, 96],
                'window_level': [-600, 1500],
                'batch_size': 2,
                'val_batch_size': 2,
                'num_workers': 2,
                'augmentation_prob': 0.3
            },
            'training_config': {
                'epochs': 200,
                'early_stopping_patience': 30,
                'save_freq': 50,
                'log_freq': 20
            },
            'experiments': {
                # Baseline methods
                'baseline_dice_bce': {
                    'description': 'Baseline 3D U-Net with Dice+BCE loss',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'baseline',
                        'dice_weight': 1.0,
                        'bce_weight': 1.0
                    },
                    'priority': 1,
                    'category': 'baseline'
                },
                # 'baseline_focal': {
                #     'description': 'Baseline 3D U-Net with Focal loss',
                #     'model_config': {
                #         'type': 'unet3d',
                #         'variant': 'standard', 
                #         'features': [32, 64, 128, 256, 512],
                #         'in_channels': 1,
                #         'out_channels': 2,
                #         'num_classes': 2
                #     },
                #     'loss_config': {
                #         'type': 'focal',
                #         'alpha': 0.25,
                #         'gamma': 2.0
                #     },
                #     'priority': 2,
                #     'category': 'baseline'
                # },
                
                # clDice experiments (Your main innovation)
                'cldice_only': {
                    'description': 'Pure clDice loss (topology-only)',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'cldice',
                        'smooth': 1e-6,
                        'use_soft_skeleton': True,
                        'params': {
                            'smooth': 1e-6,
                            'use_soft_skeleton': True,
                            'num_iter': 40
                        }
                    },
                    'priority': 3,
                    'category': 'topology_loss'
                },
                'combined_equal': {
                    'description': 'Combined Dice+clDice (1:1 ratio) - Main proposal',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'combined',
                        'dice_weight': 1.0,
                        'cldice_weight': 1.0,
                        'alpha': 0.5,
                        'beta': 0.5,
                        'params': {
                            'smooth': 1e-6,
                            'use_soft_skeleton': True,
                            'cldice_iter': 40
                        }
                    },
                    'priority': 4,
                    'category': 'topology_loss'
                },
                'combined_cldice_heavy': {
                    'description': 'Combined Dice+clDice (1:1.5 ratio) - Topology-focused',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'combined',
                        'dice_weight': 1.0,
                        'cldice_weight': 1.5,
                        'alpha': 0.4,
                        'beta': 0.6,
                        'params': {
                            'smooth': 1e-6,
                            'use_soft_skeleton': True
                        }
                    },
                    'priority': 5,
                    'category': 'topology_loss'
                },
                'adaptive_cldice': {
                    'description': 'Adaptive clDice with dynamic weighting',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'adaptive',
                        'initial_dice_weight': 1.0,
                        'initial_cldice_weight': 0.1,
                        'max_cldice_weight': 1.0,
                        'warmup_epochs': 50,
                        'params': {
                            'smooth': 1e-6,
                            'use_soft_skeleton': True
                        }
                    },
                    'priority': 6,
                    'category': 'topology_loss'
                },
                
                # Architecture variants
                'attention_unet_combined': {
                    'description': 'Attention U-Net with combined loss',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'attention',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'attention': True,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'combined',
                        'dice_weight': 1.0,
                        'cldice_weight': 1.0,
                        'alpha': 0.5,
                        'beta': 0.5,
                        'params': {
                            'smooth': 1e-6,
                            'use_soft_skeleton': True
                        }
                    },
                    'priority': 7,
                    'category': 'architecture_variant'
                },
                
                # Post-processing experiments (Your second innovation)
                'combined_with_repair': {
                    'description': 'Combined loss + skeleton repair (Full method)',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'combined',
                        'dice_weight': 1.0,
                        'cldice_weight': 1.0,
                        'alpha': 0.5,
                        'beta': 0.5,
                        'params': {
                            'smooth': 1e-6,
                            'use_soft_skeleton': True
                        }
                    },
                    'postprocessing': {
                        'enabled': True,
                        'repair_strategy': 'adaptive',
                        'morphological_cleanup': True,
                        'dilation_radius': 1,
                        'connection_threshold': 5.0,
                        'min_component_size': 50
                    },
                    'priority': 8,
                    'category': 'full_method'
                }
            },
            'evaluation_config': {
                'include_topology': True,
                'include_branch': False,
                'apply_postprocessing': False,
                'save_detailed_results': True,
                'connectivity_metrics': ['skeleton_connectivity', 'branch_point_accuracy', 'topology_preservation']
            },
            'comparison_config': {
                'generate_visualizations': True,
                'statistical_tests': ['paired_t_test', 'wilcoxon', 'effect_size'],
                'significance_level': 0.05
            },
            'resource_management': {
                'gpu_memory_limit': 8,  # GB
                'max_parallel_jobs': 1,
                'cleanup_intermediate': False,
                'checkpoint_freq': 10
            }
        }
    
    def _setup_logging(self):
        """Setup logging for experiment tracking"""
        log_file = self.experiment_dir / 'experiment_log.txt'
        
        # Create custom formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Setup logger
        self.logger = logging.getLogger(f"ExperimentRunner_{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def run_complete_experiment_suite(self):
        """
        Run the complete experiment suite for your paper
        """
        self.logger.info("Starting Complete Experiment Suite")
        self.logger.info("=" * 60)
        
        suite_start_time = time.time()
        results = {}
        
        try:
            # Phase 1: Training all methods
            self.logger.info("\n🚀 Phase 1: Training Experiments")
            training_results = self._run_training_experiments()
            results['training_results'] = training_results
            
            # Phase 2: Evaluation
            self.logger.info("\n🔬 Phase 2: Model Evaluation")
            evaluation_results = self._run_evaluation_experiments()
            results['evaluation_results'] = evaluation_results
            
            # Phase 3: Comparison Study
            self.logger.info("\n📊 Phase 3: Comparison Study")
            comparison_results = self._run_comparison_study()
            results['comparison_results'] = comparison_results
            
            # Phase 4: Visualization Generation
            self.logger.info("\n📈 Phase 4: Visualization Generation")
            visualization_results = self._generate_visualizations()
            results['visualization_results'] = visualization_results
            
            # Phase 5: Final Report
            self.logger.info("\n📄 Phase 5: Final Report Generation")
            self._generate_final_report()
            
            suite_end_time = time.time()
            total_duration = (suite_end_time - suite_start_time) / 3600  # Convert to hours
            
            self.logger.info(f"\n🎉 Complete experiment suite finished successfully!")
            self.logger.info(f"⏱️  Total duration: {total_duration:.2f} hours")
            self.logger.info(f"📁 Results saved to: {self.experiment_dir}")
            
            results.update({
                'experiment_dir': str(self.experiment_dir),
                'total_duration_hours': total_duration,
                'status': 'completed'
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Experiment suite failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            results.update({
                'error': str(e),
                'status': 'failed',
                'experiment_dir': str(self.experiment_dir)
            })
            
            return results
    
    def _run_training_experiments(self):
        """Run all training experiments"""
        self.logger.info(f"Starting training experiments...")
        
        # Sort experiments by priority
        sorted_experiments = sorted(
            self.config['experiments'].items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        training_results = {}
        
        for exp_name, exp_config in sorted_experiments:
            self.logger.info(f"\n🏋️ Training: {exp_name}")
            self.logger.info(f"   📝 Description: {exp_config['description']}")
            self.logger.info(f"   🏷️  Category: {exp_config['category']}")
            
            try:
                # Create experiment-specific output directory
                exp_output_dir = self.experiment_dir / 'training' / exp_name
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create training configuration
                train_config = self._create_training_config(exp_config, exp_output_dir)
                
                # Save training config to file for debugging
                config_file = exp_output_dir / 'train_config.json'
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(train_config, f, indent=2, ensure_ascii=False)
                
                # Run training
                start_time = time.time()
                train_result = self._execute_training(train_config)
                end_time = time.time()
                
                training_duration = end_time - start_time
                
                # Store results
                training_results[exp_name] = {
                    'config': exp_config,
                    'output_dir': str(exp_output_dir),
                    'training_duration': training_duration,
                    'result': train_result,
                    'status': 'completed'
                }
                
                self.completed_experiments[exp_name] = training_results[exp_name]
                self.logger.info(f"   ✅ Training completed in {training_duration/3600:.2f} hours")
                
            except Exception as e:
                self.logger.error(f"   ❌ Training failed: {e}")
                self.logger.error(f"   Traceback: {traceback.format_exc()}")
                
                training_results[exp_name] = {
                    'config': exp_config,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'status': 'failed'
                }
                self.failed_experiments[exp_name] = training_results[exp_name]
                
                # Continue with next experiment
                continue
        
        # Save training summary
        self._save_training_summary(training_results)
        
        return training_results
    
    def _create_training_config(self, exp_config: Dict, output_dir: Path) -> Dict:
        """Create training configuration for specific experiment"""
        train_config = {
            'data': self.config['data_config'].copy(),
            'model': exp_config['model_config'].copy(),
            'loss': exp_config['loss_config'].copy(),
            'optimizer': {
                'type': 'adamw',
                'lr': 1e-4,
                'weight_decay': 0.01,
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6,
                'warmup_epochs': 10
            },
            'training': self.config['training_config'].copy(),
            'output_dir': str(output_dir),
            'device': 'cuda' if self._check_cuda_available() else 'cpu'
        }
        
        # Add postprocessing config if specified
        if 'postprocessing' in exp_config:
            train_config['postprocessing'] = exp_config['postprocessing']
        
        # Add resource management
        train_config['resource_management'] = self.config['resource_management']
        
        return train_config
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _execute_training(self, train_config: Dict) -> Dict:
        """Execute training with subprocess approach"""
        try:
            # Create training arguments for subprocess
            train_args = [
                '--data_dir', train_config['data']['data_dir'],
                '--output_dir', train_config['output_dir'],
                '--loss_type', train_config['loss']['type'],
                '--batch_size', str(train_config['data']['batch_size']),
                '--lr', str(train_config['optimizer']['lr']),
                '--epochs', str(train_config['training']['epochs']),
                '--device', train_config['device']
            ]
            
            # Add additional arguments based on loss type
            if train_config['loss']['type'] == 'combined':
                train_args.extend([
                    '--dice_weight', str(train_config['loss'].get('dice_weight', 1.0)),
                    '--cldice_weight', str(train_config['loss'].get('cldice_weight', 1.0))
                ])
            
            if train_config.get('postprocessing', {}).get('enabled', False):
                train_args.append('--postprocessing')
            
            # Use subprocess to run training
            cmd = [sys.executable, 'train.py'] + train_args
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd='.', 
                timeout=train_config['training']['epochs'] * 300  # 5 min per epoch timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Training subprocess failed: {result.stderr}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            return {
                'subprocess_output': result.stdout,
                'status': 'completed',
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Training process timed out")
        except Exception as e:
            self.logger.error(f"Training execution failed: {e}")
            raise
    
    def _save_training_summary(self, training_results: Dict):
        """Save training summary"""
        summary_path = self.experiment_dir / 'training_summary.json'
        
        summary = {
            'total_experiments': len(training_results),
            'completed': len([r for r in training_results.values() if r['status'] == 'completed']),
            'failed': len([r for r in training_results.values() if r['status'] == 'failed']),
            'total_training_time_hours': sum([
                r.get('training_duration', 0) for r in training_results.values() 
                if r['status'] == 'completed'
            ]) / 3600,
            'experiments': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Training summary saved: {summary_path}")
    
    def _run_evaluation_experiments(self):
        """Run evaluation for all trained models"""
        self.logger.info(f"Starting evaluation experiments...")
        
        evaluation_results = {}
        eval_output_dir = self.experiment_dir / 'evaluation'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect trained models
        trained_models = {}
        for exp_name, exp_result in self.completed_experiments.items():
            if exp_result['status'] == 'completed':
                # Look for best model checkpoint
                checkpoint_paths = [
                    Path(exp_result['output_dir']) / 'checkpoints' / 'best.pth',
                    Path(exp_result['output_dir']) / 'best_model.pth',
                    Path(exp_result['output_dir']) / 'model_best.pth'
                ]
                
                best_model_path = None
                for path in checkpoint_paths:
                    if path.exists():
                        best_model_path = path
                        break
                
                if best_model_path:
                    trained_models[exp_name] = {
                        'name': exp_name,
                        'path': str(best_model_path),
                        'config': exp_result['config']
                    }
                else:
                    self.logger.warning(f"No best model checkpoint found for {exp_name}")
        
        if not trained_models:
            self.logger.warning("⚠️  No trained models found for evaluation")
            return evaluation_results
        
        self.logger.info(f"🔍 Evaluating {len(trained_models)} trained models...")
        
        # Run evaluation for each model
        for exp_name, model_info in trained_models.items():
            self.logger.info(f"🔬 Evaluating: {exp_name}")
            
            try:
                # Run evaluation using subprocess
                eval_result = self._execute_evaluation(model_info, eval_output_dir)
                
                evaluation_results[exp_name] = {
                    'model_info': model_info,
                    'evaluation_result': eval_result,
                    'status': 'completed'
                }
                
                self.logger.info(f"   ✅ Evaluation completed")
                
            except Exception as e:
                self.logger.error(f"   ❌ Evaluation failed: {e}")
                self.logger.error(f"   Traceback: {traceback.format_exc()}")
                
                evaluation_results[exp_name] = {
                    'model_info': model_info,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'status': 'failed'
                }
        
        # Save evaluation summary
        self._save_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _execute_evaluation(self, model_info: Dict, eval_output_dir: Path) -> Dict:
        """Execute evaluation using subprocess"""
        try:
            # Create evaluation arguments
            eval_args = [
                '--model_path', model_info['path'],
                '--model_name', model_info['name'],
                '--data_dir', self.config['data_config']['data_dir'],
                '--output_dir', str(eval_output_dir),
                '--batch_size', str(self.config['data_config']['batch_size'])
            ]
            
            if self.config['evaluation_config']['include_topology']:
                eval_args.append('--topology_metrics')
            
            if self.config['evaluation_config']['apply_postprocessing']:
                eval_args.append('--postprocessing')
            
            # Use subprocess to run evaluation
            cmd = [sys.executable, 'evaluate.py'] + eval_args
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='.',
                timeout=3600  # 1 hour timeout for evaluation
            )
            
            if result.returncode != 0:
                error_msg = f"Evaluation subprocess failed: {result.stderr}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            return {
                'subprocess_output': result.stdout,
                'status': 'completed',
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Evaluation process timed out")
        except Exception as e:
            self.logger.error(f"Evaluation execution failed: {e}")
            raise
    
    def _save_evaluation_summary(self, evaluation_results: Dict):
        """Save evaluation summary"""
        summary_path = self.experiment_dir / 'evaluation_summary.json'
        
        summary = {
            'total_evaluations': len(evaluation_results),
            'completed': len([r for r in evaluation_results.values() if r['status'] == 'completed']),
            'failed': len([r for r in evaluation_results.values() if r['status'] == 'failed']),
            'evaluations': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Evaluation summary saved: {summary_path}")
    
    def _run_comparison_study(self):
        """Run comprehensive comparison study"""
        self.logger.info(f"Starting comparison study...")
        
        comparison_output_dir = self.experiment_dir / 'comparison'
        comparison_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create comparison configuration
            comparison_config = self._create_comparison_config(comparison_output_dir)
            
            # Save comparison config
            config_path = comparison_output_dir / 'comparison_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_config, f, indent=2, ensure_ascii=False)
            
            # Run comparison study using subprocess
            comparison_results = self._execute_comparison_study(comparison_config)
            
            self.logger.info(f"✅ Comparison study completed")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"❌ Comparison study failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'status': 'failed'}
    
    def _create_comparison_config(self, output_dir: Path) -> Dict:
        """Create comparison study configuration"""
        # Collect all trained models for comparison
        model_configs = {}
        
        for exp_name, exp_result in self.completed_experiments.items():
            if exp_result['status'] == 'completed':
                # Look for best model checkpoint
                checkpoint_paths = [
                    Path(exp_result['output_dir']) / 'checkpoints' / 'best.pth',
                    Path(exp_result['output_dir']) / 'best_model.pth',
                    Path(exp_result['output_dir']) / 'model_best.pth'
                ]
                
                best_model_path = None
                for path in checkpoint_paths:
                    if path.exists():
                        best_model_path = path
                        break
                
                if best_model_path:
                    model_configs[exp_name] = {
                        'name': exp_result['config']['description'],
                        'path': str(best_model_path),
                        'description': exp_result['config']['description'],
                        'category': exp_result['config']['category'],
                        'color': self._get_category_color(exp_result['config']['category'])
                    }
        
        comparison_config = {
            'study_focus': self.config['experiment_name'],
            'evaluation': self.config['evaluation_config'].copy(),
            'methods': model_configs,
            'statistical_tests': self.config['comparison_config']['statistical_tests'],
            'significance_level': self.config['comparison_config']['significance_level'],
            'output_dir': str(output_dir),
            'save_detailed_results': True,
            'generate_visualizations': self.config['comparison_config']['generate_visualizations']
        }
        
        return comparison_config
    
    def _get_category_color(self, category: str) -> str:
        """Get color for experiment category"""
        colors = {
            'baseline': '#FF6B6B',           # Red for baseline methods
            'topology_loss': '#4ECDC4',      # Teal for clDice innovations  
            'architecture_variant': '#9B59B6', # Purple for architecture variants
            'full_method': '#F39C12'         # Orange for complete method
        }
        return colors.get(category, '#95A5A6')  # Default gray
    
    def _execute_comparison_study(self, comparison_config: Dict) -> Dict:
        """Execute comparison study using subprocess"""
        try:
            # Use subprocess to run comparison study
            cmd = [
                sys.executable, 'comparison_study.py',
                '--mode', 'full',
                '--config', str(comparison_config['output_dir'] + '/comparison_config.json'),
                '--output_dir', comparison_config['output_dir']
            ]
            
            self.logger.info(f"   Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='.',
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Comparison subprocess failed: {result.stderr}"
                self.logger.error(error_msg)
                # Don't raise exception, return partial results
                return {
                    'subprocess_output': result.stdout,
                    'error': error_msg,
                    'status': 'partial'
                }
            
            return {
                'subprocess_output': result.stdout,
                'status': 'completed',
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Comparison study timed out, continuing with other phases...")
            return {'error': 'Comparison study timed out', 'status': 'timeout'}
        except Exception as e:
            self.logger.warning(f"Comparison study failed: {e}, continuing...")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_visualizations(self):
        """Generate all visualizations"""
        self.logger.info(f"Generating visualizations...")
        
        viz_output_dir = self.experiment_dir / 'visualizations'
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try to generate visualizations
            if create_paper_figures_from_results is not None:
                # Collect evaluation data
                evaluation_data = self._collect_evaluation_data()
                
                # Generate paper figures
                viz_results = create_paper_figures_from_results(
                    evaluation_data,
                    output_dir=str(viz_output_dir)
                )
                
                self.logger.info(f"✅ Visualizations generated")
                return {'paper_figures': viz_results, 'status': 'completed'}
            else:
                # Create basic visualization summary
                self._create_basic_visualization_summary(viz_output_dir)
                return {'basic_summary': True, 'status': 'partial'}
                
        except Exception as e:
            self.logger.error(f"❌ Visualization generation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create basic summary as fallback
            try:
                self._create_basic_visualization_summary(viz_output_dir)
                return {'basic_summary': True, 'error': str(e), 'status': 'partial'}
            except Exception as summary_error:
                return {'error': str(summary_error), 'status': 'failed'}
    
    def _collect_evaluation_data(self) -> Dict[str, Any]:
        """Collect evaluation data for visualization"""
        evaluation_data = {
            'comparison_results': {},
            'sample_data': None,
            'experiment_configs': self.config['experiments'],
            'completed_experiments': self.completed_experiments,
            'failed_experiments': self.failed_experiments
        }
        
        # Try to load evaluation results
        try:
            eval_summary_path = self.experiment_dir / 'evaluation_summary.json'
            if eval_summary_path.exists():
                with open(eval_summary_path, 'r', encoding='utf-8') as f:
                    eval_summary = json.load(f)
                evaluation_data['evaluation_results'] = eval_summary
        except Exception as e:
            self.logger.warning(f"Could not load evaluation results: {e}")
        
        # Try to load comparison results
        try:
            comparison_files = list((self.experiment_dir / 'comparison').glob('*.json'))
            for comp_file in comparison_files:
                with open(comp_file, 'r', encoding='utf-8') as f:
                    comp_data = json.load(f)
                evaluation_data['comparison_results'][comp_file.stem] = comp_data
        except Exception as e:
            self.logger.warning(f"Could not load comparison results: {e}")
        
        return evaluation_data
    
    def _create_basic_visualization_summary(self, output_dir: Path):
        """Create basic visualization summary when full visualization fails"""
        summary_path = output_dir / 'visualization_summary.md'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Visualization Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Experiment overview
            f.write("## Experiment Overview\n\n")
            f.write(f"- **Total Experiments**: {len(self.config['experiments'])}\n")
            f.write(f"- **Completed**: {len(self.completed_experiments)}\n")
            f.write(f"- **Failed**: {len(self.failed_experiments)}\n\n")
            
            # Completed experiments by category
            if self.completed_experiments:
                f.write("## Completed Experiments by Category\n\n")
                categories = {}
                for exp_name, exp_result in self.completed_experiments.items():
                    category = exp_result['config']['category']
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(exp_name)
                
                for category, experiments in categories.items():
                    f.write(f"### {category.replace('_', ' ').title()}\n")
                    for exp in experiments:
                        f.write(f"- {exp}: {self.completed_experiments[exp]['config']['description']}\n")
                    f.write("\n")
            
            # Failed experiments
            if self.failed_experiments:
                f.write("## Failed Experiments\n\n")
                for exp_name, exp_result in self.failed_experiments.items():
                    f.write(f"### {exp_name}\n")
                    f.write(f"- **Error**: {exp_result.get('error', 'Unknown error')}\n\n")
        
        self.logger.info(f"📊 Basic visualization summary created: {summary_path}")
    
    def _generate_final_report(self):
        """Generate final experiment report"""
        self.logger.info(f"Generating final report...")
        
        report_path = self.experiment_dir / 'final_experiment_report.md'
        
        # Calculate summary statistics
        total_experiments = len(self.config['experiments'])
        completed_count = len(self.completed_experiments)
        failed_count = len(self.failed_experiments)
        
        total_training_time = sum([
            exp_result.get('training_duration', 0) 
            for exp_result in self.completed_experiments.values()
        ]) / 3600  # Convert to hours
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Paper Title:** Connectivity-aware 3D Segmentation of Pulmonary Arteries and Veins with Topology Repair\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiment Directory:** `{self.experiment_dir}`\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report summarizes the comprehensive experimental evaluation of the proposed ")
            f.write(f"connectivity-aware 3D vessel segmentation method with topology repair.\n\n")
            f.write(f"**Key Results:**\n")
            f.write(f"- Successfully completed {completed_count}/{total_experiments} experiments\n")
            f.write(f"- Total training time: {total_training_time:.2f} hours\n")
            f.write(f"- Evaluated baseline methods, clDice innovations, and complete topology repair pipeline\n\n")
            
            # Summary Statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Experiments | {total_experiments} |\n")
            f.write(f"| Completed | {completed_count} |\n")
            f.write(f"| Failed | {failed_count} |\n")
            f.write(f"| Success Rate | {(completed_count/total_experiments)*100:.1f}% |\n")
            f.write(f"| Total Training Time | {total_training_time:.2f} hours |\n\n")
            
            # Experiment Categories
            if self.completed_experiments:
                f.write("## Completed Experiments\n\n")
                
                # Group by category
                categories = {}
                for exp_name, exp_result in self.completed_experiments.items():
                    category = exp_result['config']['category']
                    if category not in categories:
                        categories[category] = []
                    categories[category].append((exp_name, exp_result))
                
                for category, experiments in categories.items():
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    
                    for exp_name, exp_result in experiments:
                        f.write(f"#### {exp_name}\n")
                        f.write(f"- **Description:** {exp_result['config']['description']}\n")
                        f.write(f"- **Training Duration:** {exp_result.get('training_duration', 0)/3600:.2f} hours\n")
                        f.write(f"- **Output Directory:** `{exp_result['output_dir']}`\n")
                        
                        # Add loss configuration details
                        loss_config = exp_result['config']['loss_config']
                        f.write(f"- **Loss Configuration:**\n")
                        for key, value in loss_config.items():
                            if key != 'params':  # Skip nested params for readability
                                f.write(f"  - {key}: {value}\n")
                        
                        # Add postprocessing info if available
                        if 'postprocessing' in exp_result['config']:
                            f.write(f"- **Postprocessing:** Enabled\n")
                        
                        f.write("\n")
            
            # Failed Experiments
            if self.failed_experiments:
                f.write("## Failed Experiments\n\n")
                for exp_name, exp_result in self.failed_experiments.items():
                    f.write(f"### {exp_name}\n")
                    f.write(f"- **Description:** {exp_result['config']['description']}\n")
                    f.write(f"- **Category:** {exp_result['config']['category']}\n")
                    f.write(f"- **Error:** {exp_result.get('error', 'Unknown error')}\n\n")
            
            # Methodology Summary
            f.write("## Methodology Summary\n\n")
            f.write("### Core Innovations\n\n")
            f.write("1. **clDice Loss Function**\n")
            f.write("   - Topology-aware loss combining prediction and skeleton overlap\n")
            f.write("   - Formula: clDice = 2⋅|S(P)∩G|⋅|S(G)∩P| / (|S(P)∩G| + |S(G)∩P|)\n")
            f.write("   - Combined with Dice loss: L = α⋅DiceLoss + β⋅(1-clDice)\n\n")
            
            f.write("2. **Skeleton-based Topology Repair**\n")
            f.write("   - Post-processing step to repair vessel connectivity\n")
            f.write("   - 3D skeleton extraction using skimage.morphology.skeletonize_3d\n")
            f.write("   - Breakage detection and adaptive repair strategies\n\n")
            
            f.write("### Experimental Design\n\n")
            f.write("- **Baseline Methods:** Standard Dice+BCE, Focal Loss\n")
            f.write("- **Topology Methods:** Pure clDice, Combined losses with different ratios\n")
            f.write("- **Architecture Variants:** Attention U-Net\n")
            f.write("- **Complete Method:** Combined loss + skeleton repair\n\n")
            
            # Directory Structure
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.experiment_dir.name}/\n")
            f.write("├── training/                    # Training results for each experiment\n")
            f.write("│   ├── baseline_dice_bce/\n")
            f.write("│   ├── baseline_focal/\n")
            f.write("│   ├── cldice_only/\n")
            f.write("│   ├── combined_equal/\n")
            f.write("│   ├── combined_cldice_heavy/\n")
            f.write("│   ├── adaptive_cldice/\n")
            f.write("│   ├── attention_unet_combined/\n")
            f.write("│   └── combined_with_repair/\n")
            f.write("├── evaluation/                  # Evaluation results and metrics\n")
            f.write("├── comparison/                  # Statistical comparison studies\n")
            f.write("├── visualizations/              # Generated figures and plots\n")
            f.write("├── training_summary.json        # Training phase summary\n")
            f.write("├── evaluation_summary.json      # Evaluation phase summary\n")
            f.write("├── experiment_log.txt           # Detailed execution log\n")
            f.write("└── final_experiment_report.md   # This report\n")
            f.write("```\n\n")
            
            # Next Steps
            f.write("## Next Steps\n\n")
            f.write("1. **Analysis Phase:**\n")
            f.write("   - Review evaluation metrics from `evaluation_summary.json`\n")
            f.write("   - Examine statistical significance from comparison studies\n")
            f.write("   - Analyze generated visualizations\n\n")
            
            f.write("2. **Paper Writing:**\n")
            f.write("   - Use results for Results section\n")
            f.write("   - Include visualizations in figures\n")
            f.write("   - Reference statistical tests for significance claims\n\n")
            
            f.write("3. **Further Experiments (if needed):**\n")
            f.write("   - Hyperparameter tuning based on initial results\n")
            f.write("   - Additional baseline comparisons\n")
            f.write("   - Ablation studies on repair strategies\n\n")
            
            # Troubleshooting section
            f.write("## Troubleshooting\n\n")
            if self.failed_experiments:
                f.write("### Common Issues\n\n")
                for exp_name, exp_result in self.failed_experiments.items():
                    error_msg = exp_result.get('error', 'Unknown error')
                    f.write(f"**{exp_name}**: {error_msg}\n\n")
                    
                    # Provide solutions for common errors
                    if 'CUDA' in error_msg or 'GPU' in error_msg:
                        f.write("**Solution**: Check GPU availability and memory. Consider reducing batch size.\n\n")
                    elif 'FileNotFoundError' in error_msg or 'dataset' in error_msg.lower():
                        f.write("**Solution**: Verify dataset path and file structure.\n\n")
                    elif 'ImportError' in error_msg:
                        f.write("**Solution**: Check module dependencies and installation.\n\n")
        
        self.logger.info(f"📄 Final report generated: {report_path}")
    
    def run(self):
        """Simplified run method for command-line usage"""
        print("🚀 Starting full experiment suite...")
        print(f"📁 Output directory: {self.experiment_dir}")
        
        try:
            results = self.run_complete_experiment_suite()
            
            if results.get('status') == 'completed':
                print(f"\n🎉 Experiment suite completed successfully!")
                print(f"⏱️  Total duration: {results.get('total_duration_hours', 0):.2f} hours")
                print(f"📁 Results saved to: {results['experiment_dir']}")
            else:
                print(f"\n⚠️  Experiment suite completed with errors")
                print(f"❌ Error: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n❌ Experiment suite failed: {e}")
            print(f"📋 Check log file: {self.experiment_dir / 'experiment_log.txt'}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for Connectivity-aware 3D Vessel Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                          # Run with default config
  python run_experiments.py --config my_config.yaml # Run with custom config
  python run_experiments.py --config my_config.json # Run with JSON config
  python run_experiments.py --dry-run               # Test configuration
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to experiment configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running experiments"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from previous experiment directory"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase (use existing models)"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase"
    )
    
    parser.add_argument(
        "--only-visualization",
        action="store_true",
        help="Only run visualization phase"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner(config_path=args.config)
        
        if args.dry_run:
            print("🔍 Dry run mode - Configuration loaded successfully")
            print(f"📋 Experiment: {runner.experiment_name}")
            print(f"📁 Output directory: {runner.experiment_dir}")
            print(f"🧪 Number of experiments: {len(runner.config['experiments'])}")
            print("\nExperiments to run:")
            for i, (exp_name, exp_config) in enumerate(runner.config['experiments'].items(), 1):
                print(f"  {i}. {exp_name}: {exp_config['description']}")
            print(f"\nConfiguration validated successfully!")
            return
        
        # Handle special modes
        if args.only_visualization:
            print("📈 Running visualization only...")
            results = runner._generate_visualizations()
            print(f"Visualization results: {results}")
            return
        
        # Run experiments
        runner.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Utility functions for creating specific experiment configurations
def create_baseline_config():
    """Create configuration for baseline experiments only"""
    runner = ExperimentRunner()
    config = runner._get_default_config()
    
    # Keep only baseline experiments
    baseline_experiments = {
        k: v for k, v in config['experiments'].items() 
        if v['category'] == 'baseline'
    }
    config['experiments'] = baseline_experiments
    config['experiment_name'] = 'baseline_only'
    
    return config


def create_topology_config():
    """Create configuration for topology-focused experiments"""
    runner = ExperimentRunner()
    config = runner._get_default_config()
    
    # Keep only topology experiments
    topology_experiments = {
        k: v for k, v in config['experiments'].items() 
        if v['category'] in ['topology_loss', 'full_method']
    }
    config['experiments'] = topology_experiments
    config['experiment_name'] = 'topology_focused'
    
    return config


def create_quick_test_config():
    """Create configuration for quick testing"""
    runner = ExperimentRunner()
    config = runner._get_default_config()
    
    # Reduce training time for testing
    config['training_config']['epochs'] = 10
    config['training_config']['early_stopping_patience'] = 5
    config['data_config']['batch_size'] = 1
    config['experiment_name'] = 'quick_test'
    
    # Keep only one experiment from each category
    test_experiments = {
        'baseline_dice_bce': config['experiments']['baseline_dice_bce'],
        'combined_equal': config['experiments']['combined_equal']
    }
    config['experiments'] = test_experiments
    
    return config


# Example usage and integration
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔬 Connectivity-aware 3D Vessel Segmentation - Experiment Runner")
        print("=" * 70)
        print("")
        print("This script runs comprehensive experiments for your paper:")
        print("'Connectivity-aware 3D Segmentation of Pulmonary Arteries and Veins'")
        print("")
        print("🚀 Quick Start:")
        print("  python run_experiments.py                    # Run all experiments")
        print("  python run_experiments.py --dry-run          # Test configuration")
        print("  python run_experiments.py --config custom.yaml  # Custom config")
        print("")
        print("📋 Experiment Categories:")
        print("  • Baseline Methods: Dice+BCE, Focal Loss")
        print("  • Topology Methods: clDice, Combined losses")
        print("  • Architecture Variants: Attention U-Net")
        print("  • Complete Method: Combined + Skeleton Repair")
        print("")
        print("📊 What it does:")
        print("  1. Trains all method variants")
        print("  2. Evaluates trained models")
        print("  3. Runs statistical comparisons")
        print("  4. Generates paper figures")
        print("  5. Creates comprehensive report")
        print("")
        print("⚙️  Advanced Options:")
        print("  --skip-training      Skip training (use existing models)")
        print("  --skip-evaluation    Skip evaluation phase")
        print("  --only-visualization Only generate visualizations")
        print("  --resume DIR         Resume from previous experiment")
        print("")
        print("📁 Output Structure:")
        print("  experiments/")
        print("  └── connectivity_aware_vessel_segmentation_TIMESTAMP/")
        print("      ├── training/           # Model checkpoints")
        print("      ├── evaluation/         # Evaluation results")
        print("      ├── comparison/         # Statistical analysis")
        print("      ├── visualizations/     # Paper figures")
        print("      └── final_experiment_report.md")
        print("")
        print("💡 Tips:")
        print("  • Use --dry-run first to validate configuration")
        print("  • Monitor GPU memory usage during training")
        print("  • Check experiment_log.txt for detailed progress")
        print("  • Results are automatically saved and timestamped")
        
    else:
        main()