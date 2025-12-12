#!/usr/bin/env python3
"""
Training Script for Chest X-Ray Pneumonia Detection
Simple script to train models using the medical training infrastructure
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training import (
    TrainingConfig, ModelTrainer, get_config_for_scenario,
    analyze_dataset, print_package_info
)

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Chest X-Ray Pneumonia Detection Model')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, choices=['quick_test', 'development', 'production', 'fine_tuning', 'high_accuracy'],
                       default='development', help='Training configuration preset')
    parser.add_argument('--dataset-path', type=str, default='data/chest_xray_final',
                       help='Path to the dataset')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--output-dir', type=str, default='outputs/training',
                       help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, choices=['efficientnet_b4', 'resnet50'],
                       default='efficientnet_b4', help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pre-trained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    # Utility arguments
    parser.add_argument('--analyze-dataset', action='store_true',
                       help='Analyze dataset before training')
    parser.add_argument('--dry-run', action='store_true',
                       help='Setup everything but don\'t start training')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--info', action='store_true',
                       help='Print package information and exit')
    
    # MLflow arguments
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='Disable MLflow experiment tracking')
    parser.add_argument('--mlflow-uri', type=str, default='sqlite:///mlflow.db',
                       help='MLflow tracking URI')
    parser.add_argument('--auto-promote', action='store_true', default=True,
                       help='Automatically promote models that meet criteria')
    
    args = parser.parse_args()
    
    # Print package info if requested
    if args.info:
        print_package_info()
        return
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Chest X-Ray Pneumonia Detection Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Analyze dataset if requested
    if args.analyze_dataset:
        logger.info("Analyzing dataset...")
        try:
            analysis = analyze_dataset(str(dataset_path))
            logger.info("Dataset analysis completed")
            
            # Print summary
            for split, stats in analysis.items():
                if 'error' not in stats:
                    logger.info(f"{split.upper()}: {stats['total_samples']} samples")
                    for class_name, count in stats['class_counts'].items():
                        percentage = stats['class_percentages'][class_name]
                        logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
    
    # Create training configuration
    logger.info(f"Creating {args.config} configuration...")
    config = get_config_for_scenario(args.config)
    
    # Override configuration with command line arguments
    config.dataset_path = str(dataset_path)
    config.output_dir = args.output_dir
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.architecture:
        config.model_architecture = args.architecture
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    config.pretrained = args.pretrained
    
    logger.info(f"Configuration created:")
    logger.info(f"  Experiment: {config.experiment_name}")
    logger.info(f"  Architecture: {config.model_architecture}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Device: {config.device}")
    
    # Save configuration
    config_path = Path(config.output_dir) / f"{config.experiment_name}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save_config(str(config_path))
    logger.info(f"Configuration saved to {config_path}")
    
    if args.dry_run:
        logger.info("Dry run completed. Exiting without training.")
        return
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    use_mlflow = not args.disable_mlflow
    trainer = ModelTrainer(config, use_mlflow=use_mlflow)
    
    if use_mlflow:
        logger.info(f"MLflow tracking enabled: {args.mlflow_uri}")
        logger.info("Launch MLflow UI with: python launch_mlflow_ui.py")
    else:
        logger.info("MLflow tracking disabled")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Start training
    logger.info("Starting training...")
    try:
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {results['best_validation_accuracy']:.4f}")
        logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
        logger.info(f"Test AUC-ROC: {results['test_metrics']['auc_roc']:.4f}")
        
        if 'sensitivity' in results['test_metrics']:
            logger.info(f"Test Sensitivity: {results['test_metrics']['sensitivity']:.4f}")
            logger.info(f"Test Specificity: {results['test_metrics']['specificity']:.4f}")
        
        logger.info(f"Results saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()