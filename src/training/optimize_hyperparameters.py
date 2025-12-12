#!/usr/bin/env python3
"""
Hyperparameter Optimization CLI
Command-line interface for running automated hyperparameter optimization
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

from .config import TrainingConfig, get_config_for_scenario
from .hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig, trigger_optimization_on_poor_performance
from .experiment_tracker import ExperimentLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_optimization_config(args) -> OptimizationConfig:
    """Create optimization configuration from CLI arguments"""
    return OptimizationConfig(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        enable_pruning=args.enable_pruning,
        primary_metric=args.primary_metric,
        max_epochs_per_trial=args.max_epochs_per_trial,
        min_accuracy_threshold=args.min_accuracy_threshold,
        study_name=args.study_name,
        output_dir=args.output_dir,
        save_best_params=True,
        save_study_results=True
    )


def run_optimization(args):
    """Run hyperparameter optimization"""
    logger.info("Starting hyperparameter optimization...")
    
    # Create base training config
    if args.base_config:
        logger.info(f"Loading base config from {args.base_config}")
        base_config = TrainingConfig.load_config(args.base_config)
    else:
        logger.info(f"Using preset config: {args.preset}")
        base_config = get_config_for_scenario(args.preset)
    
    # Override dataset path if provided
    if args.dataset_path:
        base_config.dataset_path = args.dataset_path
    
    # Create optimization config
    opt_config = create_optimization_config(args)
    
    # Setup experiment tracker
    experiment_tracker = None
    if args.use_mlflow:
        experiment_tracker = ExperimentLogger(
            experiment_name=f"hyperopt_{opt_config.study_name}",
            tracking_uri=args.mlflow_uri
        )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        optimization_config=opt_config,
        experiment_tracker=experiment_tracker
    )
    
    # Run optimization
    try:
        study = optimizer.optimize()
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"Study name: {study.study_name}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best {opt_config.primary_metric}: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save production config
        if args.save_production_config:
            prod_config = optimizer.create_production_config(
                extended_epochs=args.production_epochs
            )
            if prod_config:
                prod_config_path = Path(opt_config.output_dir) / f"{opt_config.study_name}_production_config.json"
                prod_config.save_config(str(prod_config_path))
                print(f"\nProduction config saved to: {prod_config_path}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        sys.exit(1)


def run_auto_optimization(args):
    """Run automatic optimization triggered by poor performance"""
    logger.info("Running automatic optimization for poor performance...")
    
    # Create base config
    if args.base_config:
        base_config = TrainingConfig.load_config(args.base_config)
    else:
        base_config = get_config_for_scenario(args.preset)
    
    if args.dataset_path:
        base_config.dataset_path = args.dataset_path
    
    # Setup experiment tracker
    experiment_tracker = None
    if args.use_mlflow:
        experiment_tracker = ExperimentLogger(
            experiment_name=f"auto_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tracking_uri=args.mlflow_uri
        )
    
    # Trigger optimization
    optimized_config = trigger_optimization_on_poor_performance(
        current_accuracy=args.current_accuracy,
        threshold=args.accuracy_threshold,
        base_config=base_config,
        experiment_tracker=experiment_tracker
    )
    
    if optimized_config:
        # Save optimized config
        output_path = Path(args.output_dir) / f"auto_optimized_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimized_config.save_config(str(output_path))
        
        print(f"\nOptimization completed! New config saved to: {output_path}")
        print("You can now use this config for retraining your model.")
    else:
        print("No optimization needed - current accuracy is above threshold.")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for Chest X-Ray Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Optimization command
    opt_parser = subparsers.add_parser('optimize', help='Run hyperparameter optimization')
    
    # Base configuration
    opt_parser.add_argument('--base-config', type=str, help='Path to base training config JSON')
    opt_parser.add_argument('--preset', type=str, default='development', 
                           choices=['quick_test', 'development', 'production', 'fine_tuning', 'high_accuracy'],
                           help='Preset configuration to use')
    opt_parser.add_argument('--dataset-path', type=str, help='Override dataset path')
    
    # Optimization settings
    opt_parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials')
    opt_parser.add_argument('--timeout', type=int, help='Optimization timeout in seconds')
    opt_parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    opt_parser.add_argument('--enable-pruning', action='store_true', default=True, help='Enable trial pruning')
    opt_parser.add_argument('--primary-metric', type=str, default='val_f1_score',
                           choices=['val_accuracy', 'val_f1_score', 'val_auc_roc'],
                           help='Primary metric to optimize')
    opt_parser.add_argument('--max-epochs-per-trial', type=int, default=25, help='Max epochs per trial')
    opt_parser.add_argument('--min-accuracy-threshold', type=float, default=0.7, help='Minimum accuracy threshold')
    
    # Study settings
    opt_parser.add_argument('--study-name', type=str, help='Optuna study name')
    opt_parser.add_argument('--output-dir', type=str, default='outputs/optimization', help='Output directory')
    
    # MLflow settings
    opt_parser.add_argument('--use-mlflow', action='store_true', default=True, help='Use MLflow tracking')
    opt_parser.add_argument('--mlflow-uri', type=str, default='sqlite:///mlflow.db', help='MLflow tracking URI')
    
    # Production config
    opt_parser.add_argument('--save-production-config', action='store_true', default=True, 
                           help='Save production config with best parameters')
    opt_parser.add_argument('--production-epochs', type=int, default=100, 
                           help='Number of epochs for production config')
    
    # Auto optimization command
    auto_parser = subparsers.add_parser('auto', help='Run automatic optimization for poor performance')
    auto_parser.add_argument('--current-accuracy', type=float, required=True, 
                            help='Current model accuracy that triggered optimization')
    auto_parser.add_argument('--accuracy-threshold', type=float, default=0.85, 
                            help='Accuracy threshold for triggering optimization')
    auto_parser.add_argument('--base-config', type=str, help='Path to base training config JSON')
    auto_parser.add_argument('--preset', type=str, default='development', 
                           choices=['quick_test', 'development', 'production', 'fine_tuning', 'high_accuracy'],
                           help='Preset configuration to use')
    auto_parser.add_argument('--dataset-path', type=str, help='Override dataset path')
    auto_parser.add_argument('--output-dir', type=str, default='outputs/optimization', help='Output directory')
    auto_parser.add_argument('--use-mlflow', action='store_true', default=True, help='Use MLflow tracking')
    auto_parser.add_argument('--mlflow-uri', type=str, default='sqlite:///mlflow.db', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    if args.command == 'optimize':
        run_optimization(args)
    elif args.command == 'auto':
        run_auto_optimization(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()