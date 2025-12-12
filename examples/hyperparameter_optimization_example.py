#!/usr/bin/env python3
"""
Hyperparameter Optimization Example
Demonstrates how to use the hyperparameter optimization system
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.config import TrainingConfig, get_config_for_scenario
from training.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
from training.experiment_tracker import ExperimentLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_optimization():
    """Example of basic hyperparameter optimization"""
    print("="*60)
    print("BASIC HYPERPARAMETER OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Create base configuration
    base_config = get_config_for_scenario('development')
    base_config.dataset_path = 'data/chest_xray_final'  # Update to your dataset path
    
    print(f"Base configuration:")
    print(f"  Model: {base_config.model_architecture}")
    print(f"  Learning rate: {base_config.learning_rate}")
    print(f"  Batch size: {base_config.batch_size}")
    print(f"  Epochs: {base_config.epochs}")
    
    # Create optimization configuration
    opt_config = OptimizationConfig(
        n_trials=5,  # Small number for demo
        max_epochs_per_trial=3,  # Short training for demo
        primary_metric='val_f1_score',
        study_name='demo_optimization',
        output_dir='outputs/demo_optimization'
    )
    
    print(f"\nOptimization configuration:")
    print(f"  Number of trials: {opt_config.n_trials}")
    print(f"  Max epochs per trial: {opt_config.max_epochs_per_trial}")
    print(f"  Primary metric: {opt_config.primary_metric}")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        optimization_config=opt_config
    )
    
    print(f"\nOptimizer created successfully!")
    print(f"Study name: {opt_config.study_name}")
    print(f"Output directory: {opt_config.output_dir}")
    
    # Note: Actual optimization would require dataset
    print(f"\nTo run optimization, you would call:")
    print(f"  study = optimizer.optimize()")
    print(f"\nThis requires a valid dataset at: {base_config.dataset_path}")


def example_auto_optimization():
    """Example of automatic optimization triggered by poor performance"""
    print("\n" + "="*60)
    print("AUTOMATIC OPTIMIZATION EXAMPLE")
    print("="*60)
    
    from training.hyperparameter_optimizer import trigger_optimization_on_poor_performance
    
    # Simulate poor performance scenario
    current_accuracy = 0.78  # Below threshold
    threshold = 0.85
    
    print(f"Current model accuracy: {current_accuracy:.3f}")
    print(f"Accuracy threshold: {threshold:.3f}")
    print(f"Performance is below threshold - optimization needed!")
    
    # Create base config
    base_config = get_config_for_scenario('development')
    base_config.dataset_path = 'data/chest_xray_final'
    
    print(f"\nBase configuration for optimization:")
    print(f"  Model: {base_config.model_architecture}")
    print(f"  Dataset: {base_config.dataset_path}")
    
    # Note: This would trigger actual optimization
    print(f"\nTo trigger automatic optimization, you would call:")
    print(f"  optimized_config = trigger_optimization_on_poor_performance(")
    print(f"      current_accuracy={current_accuracy},")
    print(f"      base_config=base_config,")
    print(f"      threshold={threshold}")
    print(f"  )")
    print(f"\nThis would return an optimized TrainingConfig if successful.")


def example_production_config():
    """Example of creating production config from optimization results"""
    print("\n" + "="*60)
    print("PRODUCTION CONFIG EXAMPLE")
    print("="*60)
    
    # Simulate optimization results
    print("After optimization completes, you can create a production config:")
    
    # Create mock optimizer with best trial
    base_config = get_config_for_scenario('development')
    opt_config = OptimizationConfig(study_name='demo_optimization')
    
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        optimization_config=opt_config
    )
    
    # Simulate best trial results
    optimizer.best_trial = {
        'params': {
            'model_architecture': 'efficientnet_b4',
            'learning_rate': 5e-4,
            'batch_size': 32,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'scheduler': 'cosine',
            'loss_function': 'cross_entropy',
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'fine_tuning_strategy': 'gradual',
            'freeze_epochs': 10,
            'augmentation_config': {
                'rotation_range': 15,
                'brightness_range': 0.15,
                'contrast_range': 0.15,
                'horizontal_flip': True,
                'vertical_flip': False,
                'zoom_range': 0.1,
                'shear_range': 5,
                'fill_mode': 'reflect'
            }
        }
    }
    
    # Create production config
    prod_config = optimizer.create_production_config(extended_epochs=100)
    
    if prod_config:
        print(f"Production configuration created:")
        print(f"  Model: {prod_config.model_architecture}")
        print(f"  Learning rate: {prod_config.learning_rate}")
        print(f"  Batch size: {prod_config.batch_size}")
        print(f"  Optimizer: {prod_config.optimizer}")
        print(f"  Epochs: {prod_config.epochs}")
        print(f"  Early stopping patience: {prod_config.early_stopping_patience}")
        
        # Save config
        config_path = Path('outputs/demo_production_config.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        prod_config.save_config(str(config_path))
        print(f"\nProduction config saved to: {config_path}")


def example_cli_usage():
    """Example of CLI usage"""
    print("\n" + "="*60)
    print("CLI USAGE EXAMPLES")
    print("="*60)
    
    print("1. Basic optimization:")
    print("   python -m training.optimize_hyperparameters optimize \\")
    print("     --preset development \\")
    print("     --dataset-path data/chest_xray_final \\")
    print("     --n-trials 20 \\")
    print("     --max-epochs-per-trial 15")
    
    print("\n2. Automatic optimization for poor performance:")
    print("   python -m training.optimize_hyperparameters auto \\")
    print("     --current-accuracy 0.78 \\")
    print("     --accuracy-threshold 0.85 \\")
    print("     --dataset-path data/chest_xray_final")
    
    print("\n3. Advanced optimization with MLflow:")
    print("   python -m training.optimize_hyperparameters optimize \\")
    print("     --base-config my_config.json \\")
    print("     --n-trials 50 \\")
    print("     --primary-metric val_auc_roc \\")
    print("     --use-mlflow \\")
    print("     --mlflow-uri sqlite:///optimization.db")


def main():
    """Run all examples"""
    print("HYPERPARAMETER OPTIMIZATION EXAMPLES")
    print("This script demonstrates the hyperparameter optimization system")
    print("Note: Actual optimization requires a valid dataset")
    
    try:
        example_basic_optimization()
        example_auto_optimization()
        example_production_config()
        example_cli_usage()
        
        print("\n" + "="*60)
        print("EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Next steps:")
        print("1. Prepare your dataset at the specified path")
        print("2. Run optimization using the CLI or Python API")
        print("3. Use the optimized config for production training")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()