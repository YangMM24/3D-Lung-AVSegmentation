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
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules
try:
    from train import main as train_main
    from evaluate import main as evaluate_main
    from comparison_study import run_paper_comparison_study
    from visualization import create_paper_figures_from_results
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


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
        self.base_output_dir = self.config['base_output_dir']
        self.experiment_name = self.config['experiment_name']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.experiment_dir = os.path.join(self.base_output_dir, f"{self.experiment_name}_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
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
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
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
                'baseline_focal': {
                    'description': 'Baseline 3D U-Net with Focal loss',
                    'model_config': {
                        'type': 'unet3d',
                        'variant': 'standard',
                        'features': [32, 64, 128, 256, 512],
                        'in_channels': 1,
                        'out_channels': 2,
                        'num_classes': 2
                    },
                    'loss_config': {
                        'type': 'focal',
                        'alpha': 0.25,
                        'gamma': 2.0
                    },
                    'priority': 2,
                    'category': 'baseline'
                },
                
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
                        'use_soft_skeleton': True
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
                        'cldice_weight': 1.0
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
                        'cldice_weight': 1.5
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
                        'warmup_epochs': 50
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
                        'cldice_weight': 1.0
                    },
                    'priority': 7,
                    'category': 'architecture_variant'
                },
                
                # Post-processing experiments
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
                        'cldice_weight': 1.0
                    },
                    'postprocessing': {
                        'enabled': True,
                        'repair_strategy': 'adaptive',
                        'morphological_cleanup': True,
                        'dilation_radius': 1
                    },
                    'priority': 8,
                    'category': 'full_method'
                }
            },
            'evaluation_config': {
                'include_topology': True,
                'include_branch': False,
                'apply_postprocessing': False,
                'save_detailed_results': True
            },
            'comparison_config': {
                'generate_visualizations': True,
                'statistical_tests': ['paired_t_test', 'wilcoxon', 'effect_size'],
                'significance_level': 0.05
            },
            'resource_management': {
                'gpu_memory_limit': 8,  # GB
                'max_parallel_jobs': 1,
                'cleanup_intermediate': False
            }
        }
    
    def _setup_logging(self):
        """Setup logging for experiment tracking"""
        log_file = os.path.join(self.experiment_dir, 'experiment_log.txt')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_complete_experiment_suite(self):
        """
        Run the complete experiment suite for your paper
        """
        self.logger.info("Starting Complete Experiment Suite")
        self.logger.info("=" * 60)
        
        try:
            # Phase 1: Training all methods
            self.logger.info("\n--- Phase 1: Training Experiments ---")
            training_results = self._run_training_experiments()
            
            # Phase 2: Evaluation
            self.logger.info("\n--- Phase 2: Model Evaluation ---")
            evaluation_results = self._run_evaluation_experiments()
            
            # Phase 3: Comparison Study
            self.logger.info("\n--- Phase 3: Comparison Study ---")
            comparison_results = self._run_comparison_study()
            
            # Phase 4: Visualization Generation
            self.logger.info("\n--- Phase 4: Visualization Generation ---")
            visualization_results = self._generate_visualizations()
            
            # Phase 5: Final Report
            self.logger.info("\n--- Phase 5: Final Report Generation ---")
            self._generate_final_report()
            
            self.logger.info("🎉 Complete experiment suite finished successfully!")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'comparison_results': comparison_results,
                'visualization_results': visualization_results,
                'experiment_dir': self.experiment_dir
            }
            
        except Exception as e:
            self.logger.error(f"Experiment suite failed: {e}")
            raise
    
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
            self.logger.info(f"\nTraining: {exp_name}")
            self.logger.info(f"   Description: {exp_config['description']}")
            
            try:
                # Create experiment-specific output directory
                exp_output_dir = os.path.join(self.experiment_dir, 'training', exp_name)
                os.makedirs(exp_output_dir, exist_ok=True)
                
                # Create training configuration
                train_config = self._create_training_config(exp_config, exp_output_dir)
                
                # Run training
                start_time = time.time()
                train_result = self._execute_training(train_config)
                end_time = time.time()
                
                training_duration = end_time - start_time
                
                # Store results
                training_results[exp_name] = {
                    'config': exp_config,
                    'output_dir': exp_output_dir,
                    'training_duration': training_duration,
                    'result': train_result,
                    'status': 'completed'
                }
                
                self.completed_experiments[exp_name] = training_results[exp_name]
                self.logger.info(f"   Training completed in {training_duration/3600:.2f} hours")
                
            except Exception as e:
                self.logger.error(f"   Training failed: {e}")
                training_results[exp_name] = {
                    'config': exp_config,
                    'error': str(e),
                    'status': 'failed'
                }
                self.failed_experiments[exp_name] = training_results[exp_name]
                
                # Continue with next experiment
                continue
        
        # Save training summary
        self._save_training_summary(training_results)
        
        return training_results
    
    def _create_training_config(self, exp_config, output_dir):
        """Create training configuration for specific experiment"""
        train_config = {
            'data': self.config['data_config'].copy(),
            'model': exp_config['model_config'].copy(),
            'loss': exp_config['loss_config'].copy(),
            'optimizer': {
                'type': 'adamw',
                'lr': 1e-4,
                'weight_decay': 0.01
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6
            },
            'training': self.config['training_config'].copy(),
            'output_dir': output_dir,
            'device': 'cuda'
        }
        
        # Add postprocessing config if specified
        if 'postprocessing' in exp_config:
            train_config['postprocessing'] = exp_config['postprocessing']
        
        return train_config
    
    def _execute_training(self, train_config):
        """Execute training with given configuration"""
        # Save config for this training run
        config_path = os.path.join(train_config['output_dir'], 'train_config.json')
        with open(config_path, 'w') as f:
            json.dump(train_config, f, indent=2)
        
        # Create training script arguments
        train_args = [
            '--config', config_path,
            '--data_dir', train_config['data']['data_dir'],
            '--output_dir', train_config['output_dir'],
            '--loss_type', train_config['loss']['type'],
            '--batch_size', str(train_config['data']['batch_size']),
            '--lr', str(train_config['optimizer']['lr']),
            '--epochs', str(train_config['training']['epochs'])
        ]
        
        if train_config.get('postprocessing', {}).get('enabled', False):
            train_args.append('--postprocessing')
        
        # Execute training
        # You can either call train_main directly or use subprocess
        try:
            # Direct call (recommended)
            original_argv = sys.argv.copy()
            sys.argv = ['train.py'] + train_args
            
            # Import and run training
            from train import main as train_main
            result = train_main()
            
            # Restore original argv
            sys.argv = original_argv
            
            return result
            
        except Exception as e:
            # Fallback to subprocess if direct call fails
            cmd = [sys.executable, 'train.py'] + train_args
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                raise Exception(f"Training subprocess failed: {result.stderr}")
            
            return {'subprocess_output': result.stdout}
    
    def _save_training_summary(self, training_results):
        """Save training summary"""
        summary_path = os.path.join(self.experiment_dir, 'training_summary.json')
        
        summary = {
            'total_experiments': len(training_results),
            'completed': len([r for r in training_results.values() if r['status'] == 'completed']),
            'failed': len([r for r in training_results.values() if r['status'] == 'failed']),
            'total_training_time': sum([
                r.get('training_duration', 0) for r in training_results.values() 
                if r['status'] == 'completed'
            ]),
            'experiments': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved: {summary_path}")
    
    def _run_evaluation_experiments(self):
        """Run evaluation for all trained models"""
        self.logger.info(f"Starting evaluation experiments...")
        
        evaluation_results = {}
        eval_output_dir = os.path.join(self.experiment_dir, 'evaluation')
        os.makedirs(eval_output_dir, exist_ok=True)
        
        # Collect trained models
        trained_models = {}
        for exp_name, exp_result in self.completed_experiments.items():
            if exp_result['status'] == 'completed':
                best_model_path = os.path.join(exp_result['output_dir'], 'checkpoints', 'best.pth')
                if os.path.exists(best_model_path):
                    trained_models[exp_name] = {
                        'name': exp_name,
                        'path': best_model_path,
                        'config': exp_result['config']
                    }
        
        if not trained_models:
            self.logger.warning(" No trained models found for evaluation")
            return evaluation_results
        
        self.logger.info(f"Evaluating {len(trained_models)} trained models...")
        
        # Create evaluation configuration
        eval_config = self._create_evaluation_config(eval_output_dir)
        
        # Run evaluation for each model
        for exp_name, model_info in trained_models.items():
            self.logger.info(f"🔬 Evaluating: {exp_name}")
            
            try:
                # Run evaluation
                eval_result = self._execute_evaluation(model_info, eval_config)
                
                evaluation_results[exp_name] = {
                    'model_info': model_info,
                    'evaluation_result': eval_result,
                    'status': 'completed'
                }
                
                self.logger.info(f"   Evaluation completed")
                
            except Exception as e:
                self.logger.error(f"   Evaluation failed: {e}")
                evaluation_results[exp_name] = {
                    'model_info': model_info,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Save evaluation summary
        self._save_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _create_evaluation_config(self, output_dir):
        """Create evaluation configuration"""
        eval_config = {
            'data': self.config['data_config'].copy(),
            'evaluation': self.config['evaluation_config'].copy(),
            'output_dir': output_dir,
            'device': 'cuda'
        }
        
        return eval_config
    
    def _execute_evaluation(self, model_info, eval_config):
        """Execute evaluation for specific model"""
        # Create evaluation arguments
        eval_args = [
            '--model_path', model_info['path'],
            '--model_name', model_info['name'],
            '--data_dir', eval_config['data']['data_dir'],
            '--output_dir', eval_config['output_dir'],
            '--batch_size', str(eval_config['data']['batch_size'])
        ]
        
        if eval_config['evaluation']['include_topology']:
            eval_args.append('--topology_metrics')
        
        if eval_config['evaluation']['apply_postprocessing']:
            eval_args.append('--postprocessing')
        
        # Execute evaluation
        try:
            # Direct call
            original_argv = sys.argv.copy()
            sys.argv = ['evaluate.py'] + eval_args
            
            from evaluate import main as evaluate_main
            result = evaluate_main()
            
            sys.argv = original_argv
            return result
            
        except Exception as e:
            # Fallback to subprocess
            cmd = [sys.executable, 'evaluate.py'] + eval_args
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                raise Exception(f"Evaluation subprocess failed: {result.stderr}")
            
            return {'subprocess_output': result.stdout}
    
    def _save_evaluation_summary(self, evaluation_results):
        """Save evaluation summary"""
        summary_path = os.path.join(self.experiment_dir, 'evaluation_summary.json')
        
        summary = {
            'total_evaluations': len(evaluation_results),
            'completed': len([r for r in evaluation_results.values() if r['status'] == 'completed']),
            'failed': len([r for r in evaluation_results.values() if r['status'] == 'failed']),
            'evaluations': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Evaluation summary saved: {summary_path}")
    
    def _run_comparison_study(self):
        """Run comprehensive comparison study"""
        self.logger.info(f"Starting comparison study...")
        
        comparison_output_dir = os.path.join(self.experiment_dir, 'comparison')
        os.makedirs(comparison_output_dir, exist_ok=True)
        
        try:
            # Create comparison configuration
            comparison_config = self._create_comparison_config(comparison_output_dir)
            
            # Run comparison study
            comparison_results = self._execute_comparison_study(comparison_config)
            
            self.logger.info(f"Comparison study completed")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Comparison study failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _create_comparison_config(self, output_dir):
        """Create comparison study configuration"""
        # Collect all trained models for comparison
        model_configs = {}
        
        for exp_name, exp_result in self.completed_experiments.items():
            if exp_result['status'] == 'completed':
                best_model_path = os.path.join(exp_result['output_dir'], 'checkpoints', 'best.pth')
                if os.path.exists(best_model_path):
                    model_configs[exp_name] = {
                        'name': exp_result['config']['description'],
                        'path': best_model_path,
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
            'output_dir': output_dir,
            'save_detailed_results': True,
            'generate_visualizations': self.config['comparison_config']['generate_visualizations']
        }
        
        return comparison_config
    
    def _get_category_color(self, category):
        """Get color for experiment category"""
        colors = {
            'baseline': '#FF6B6B',
            'topology_loss': '#4ECDC4',
            'architecture_variant': '#9B59B6',
            'full_method': '#F39C12'
        }
        return colors.get(category, '#95A5A6')
    
    def _execute_comparison_study(self, comparison_config):
        """Execute comparison study"""
        try:
            # Direct call to comparison study
            from comparison_study import ComprehensiveComparisonStudy
            
            study = ComprehensiveComparisonStudy(comparison_config)
            results = study.run_comprehensive_comparison()
            
            return results
            
        except Exception as e:
            # Fallback approach
            self.logger.warning(f"Direct comparison call failed: {e}")
            self.logger.info("Attempting alternative comparison approach...")
            
            # Alternative: run comparison_study.py as subprocess
            cmd = [sys.executable, 'comparison_study.py', '--mode', 'full', 
                   '--output_dir', comparison_config['output_dir']]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                raise Exception(f"Comparison subprocess failed: {result.stderr}")
            
            return {'subprocess_output': result.stdout}
    
    def _generate_visualizations(self):
        """Generate all visualizations"""
        self.logger.info(f"Generating visualizations...")
        
        viz_output_dir = os.path.join(self.experiment_dir, 'visualizations')
        os.makedirs(viz_output_dir, exist_ok=True)
        
        try:
            # Collect evaluation results
            evaluation_data = self._collect_evaluation_data()
            
            # Generate paper figures
            viz_results = self._execute_visualization_generation(evaluation_data, viz_output_dir)
            
            self.logger.info(f"Visualizations generated")
            return viz_results
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _collect_evaluation_data(self):
        """Collect evaluation data for visualization"""
        # This would collect data from completed evaluations
        # For now, return a placeholder structure
        return {
            'comparison_results': {},
            'sample_data': None
        }
    
    def _execute_visualization_generation(self, evaluation_data, output_dir):
        """Execute visualization generation"""
        try:
            from visualization import create_paper_figures_from_results
            
            paper_figures = create_paper_figures_from_results(
                evaluation_data,
                output_dir=output_dir
            )
            
            return {'paper_figures': paper_figures, 'status': 'completed'}
            
        except Exception as e:
            self.logger.warning(f"Paper figures generation failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_final_report(self):
        """Generate final experiment report"""
        self.logger.info(f"Generating final report...")
        
        report_path = os.path.join(self.experiment_dir, 'final_experiment_report.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiment Directory:** `{self.experiment_dir}`\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- **Total Experiments:** {len(self.config['experiments'])}\n")
            f.write(f"- **Completed:** {len(self.completed_experiments)}\n")
            f.write(f"- **Failed:** {len(self.failed_experiments)}\n\n")
            
            # Experiment results
            if self.completed_experiments:
                f.write("## Completed Experiments\n\n")
                for exp_name, exp_result in self.completed_experiments.items():
                    f.write(f"### {exp_name}\n")
                    f.write(f"- **Description:** {exp_result['config']['description']}\n")
                    f.write(f"- **Category:** {exp_result['config']['category']}\n")
                    f.write(f"- **Training Duration:** {exp_result.get('training_duration', 0)/3600:.2f} hours\n")
                    f.write(f"- **Output Directory:** `{exp_result['output_dir']}`\n\n")
            
            # Failed experiments
            if self.failed_experiments:
                f.write("## Failed Experiments\n\n")
                for exp_name, exp_result in self.failed_experiments.items():
                    f.write(f"### {exp_name}\n")
                    f.write(f"- **Error:** {exp_result.get('error', 'Unknown error')}\n\n")
            
            # Directory structure
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.experiment_dir}/\n")
            f.write("├── training/\n")
            f.write("│   ├── experiment1/\n")
            f.write("│   └── experiment2/\n")
            f.write("├── evaluation/\n")
            f.write("├── comparison/\n")
            f.write("├── visualizations/\n")
            f.write("├── training_summary.json\n")
            f.write("├── evaluation_summary.json\n")
            f.write("└── final_experiment_report.md\n")
            f.write("```\n\n")
    def run(self):
        print("Starting full experiment suite...")

        # Phase 1: Training
        try:
            print("[Phase 1] Training experiments...")
            self._run_training_experiments()
        except Exception as e:
            print(f"[Warning] Training phase failed: {e}")

        # Phase 2: Evaluation
        try:
            print("[Phase 2] Evaluating experiments...")
            self._run_evaluation_experiments()
        except Exception as e:
            print(f"[Warning] Evaluation phase failed: {e}")

        # Phase 3: Comparison study
        try:
            print("[Phase 3] Running comparison analysis...")
            self._run_comparison_study()
        except Exception as e:
            print(f"[Warning] Comparison phase failed: {e}")

        # Phase 4: Visualization
        try:
            print("[Phase 4] Generating visualizations...")
            self._generate_visualizations()
        except Exception as e:
            print(f"[Warning] Visualization phase failed: {e}")

        # Phase 5: Final Report (optional, not implemented)
        try:
            print("[Phase 5] Generating final report...")
            self._generate_final_report()
        except Exception as e:
            print(f"[Warning] Final report generation failed: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    args = parser.parse_args()

    runner = ExperimentRunner(config_path=args.config)
    runner.run()
