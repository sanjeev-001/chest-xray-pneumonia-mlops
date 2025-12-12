"""
Hyperparameter Optimization for Medical Chest X-Ray Models
Uses Optuna for automated hyperparameter tuning with medical-specific objectives
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import logging
from typing import Dict, Any, Optional, List, Callable
import json
import os
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from dataclasses import dataclass

from .config import TrainingConfig
from .trainer import MedicalImageTrainer
from .experiment_tracker import ExperimentLogger

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    
    # Optimization settings
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1  # parallel jobs
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5
    pruning_interval_steps: int = 1
    
    # Study settings
    study_name: Optional[str] = None
    storage_url: Optional[str] = None  # For distributed optimization
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    
    # Optimization target
    primary_metric: str = 'val_f1_score'  # Metric to optimize
    early_stopping_rounds: int = 10  # Stop if no improvement
    
    # Search space constraints
    max_epochs_per_trial: int = 25  # Limit epochs per trial
    min_accuracy_threshold: float = 0.7  # Minimum acceptable accuracy
    
    # Output settings
    output_dir: str = 'outputs/optimization'
    save_best_params: bool = True
    save_study_results: bool = True


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for medical imaging models
    """
    
    def __init__(self, 
                 base_config: TrainingConfig,
                 optimization_config: OptimizationConfig,
                 experiment_tracker: Optional[ExperimentLogger] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            base_config: Base training configuration
            optimization_config: Optimization settings
            experiment_tracker: MLflow experiment tracker
        """
        self.base_config = base_config
        self.opt_config = optimization_config
        self.experiment_tracker = experiment_tracker
        
        # Setup output directory
        Path(self.opt_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize study name
        if self.opt_config.study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.opt_config.study_name = f"chest_xray_optimization_{timestamp}"
        
        # Best trial tracking
        self.best_trial = None
        self.best_score = float('-inf') if self.opt_config.direction == 'maximize' else float('inf')
        
        logger.info(f"Initialized hyperparameter optimizer: {self.opt_config.study_name}")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for medical imaging
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters to try
        """
        # Model architecture (if multiple supported)
        model_architecture = trial.suggest_categorical(
            'model_architecture', 
            ['efficientnet_b4', 'resnet50', 'densenet121']
        )
        
        # Learning rate (log scale)
        learning_rate = trial.suggest_float(
            'learning_rate', 
            1e-5, 1e-2, 
            log=True
        )
        
        # Batch size (powers of 2)
        batch_size = trial.suggest_categorical(
            'batch_size', 
            [8, 16, 32, 64]
        )
        
        # Optimizer
        optimizer = trial.suggest_categorical(
            'optimizer', 
            ['adamw', 'adam', 'sgd']
        )
        
        # Weight decay
        weight_decay = trial.suggest_float(
            'weight_decay', 
            1e-6, 1e-2, 
            log=True
        )
        
        # Dropout rate
        dropout_rate = trial.suggest_float(
            'dropout_rate', 
            0.1, 0.6
        )
        
        # Learning rate scheduler
        scheduler = trial.suggest_categorical(
            'scheduler', 
            ['cosine', 'step', 'plateau']
        )
        
        # Loss function
        loss_function = trial.suggest_categorical(
            'loss_function',
            ['cross_entropy', 'focal', 'weighted_ce']
        )
        
        # Focal loss parameters (if focal loss selected)
        focal_alpha = 0.25
        focal_gamma = 2.0
        if loss_function == 'focal':
            focal_alpha = trial.suggest_float('focal_loss_alpha', 0.1, 0.9)
            focal_gamma = trial.suggest_float('focal_loss_gamma', 1.0, 5.0)
        
        # Data augmentation strength
        aug_rotation = trial.suggest_int('aug_rotation_range', 5, 20)
        aug_brightness = trial.suggest_float('aug_brightness_range', 0.05, 0.2)
        aug_contrast = trial.suggest_float('aug_contrast_range', 0.05, 0.2)
        
        # Fine-tuning strategy
        fine_tuning_strategy = trial.suggest_categorical(
            'fine_tuning_strategy',
            ['gradual', 'full']
        )
        
        freeze_epochs = 0
        if fine_tuning_strategy == 'gradual':
            freeze_epochs = trial.suggest_int('freeze_epochs', 3, 15)
        
        return {
            'model_architecture': model_architecture,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'scheduler': scheduler,
            'loss_function': loss_function,
            'focal_loss_alpha': focal_alpha,
            'focal_loss_gamma': focal_gamma,
            'fine_tuning_strategy': fine_tuning_strategy,
            'freeze_epochs': freeze_epochs,
            'augmentation_config': {
                'rotation_range': aug_rotation,
                'brightness_range': aug_brightness,
                'contrast_range': aug_contrast,
                'horizontal_flip': True,
                'vertical_flip': False,
                'zoom_range': 0.1,
                'shear_range': 5,
                'fill_mode': 'reflect'
            }
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score to optimize (higher is better for maximize)
        """
        try:
            # Get hyperparameters for this trial
            params = self.define_search_space(trial)
            
            # Create training config with suggested parameters
            trial_config = self._create_trial_config(params)
            
            # Initialize trainer
            trainer = MedicalImageTrainer(
                config=trial_config,
                experiment_tracker=self.experiment_tracker
            )
            
            # Train model with early stopping
            results = trainer.train_with_early_stopping(
                max_epochs=self.opt_config.max_epochs_per_trial,
                patience=self.opt_config.early_stopping_rounds,
                trial=trial  # Pass trial for pruning
            )
            
            # Extract target metric
            target_score = results.get(self.opt_config.primary_metric, 0.0)
            
            # Apply minimum accuracy constraint
            val_accuracy = results.get('val_accuracy', 0.0)
            if val_accuracy < self.opt_config.min_accuracy_threshold:
                logger.warning(f"Trial {trial.number}: Accuracy {val_accuracy:.3f} below threshold {self.opt_config.min_accuracy_threshold}")
                return 0.0  # Penalize low accuracy
            
            # Log trial results
            self._log_trial_results(trial, params, results, target_score)
            
            # Update best trial
            self._update_best_trial(trial, target_score, params, results)
            
            return target_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0  # Return poor score for failed trials
    
    def _create_trial_config(self, params: Dict[str, Any]) -> TrainingConfig:
        """Create training config for trial"""
        # Start with base config
        trial_config = TrainingConfig(
            # Copy base settings
            dataset_path=self.base_config.dataset_path,
            image_size=self.base_config.image_size,
            num_classes=self.base_config.num_classes,
            class_names=self.base_config.class_names,
            device=self.base_config.device,
            random_seed=self.base_config.random_seed,
            
            # Override with trial parameters
            model_architecture=params['model_architecture'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            optimizer=params['optimizer'],
            weight_decay=params['weight_decay'],
            dropout_rate=params['dropout_rate'],
            scheduler=params['scheduler'],
            loss_function=params['loss_function'],
            focal_loss_alpha=params['focal_loss_alpha'],
            focal_loss_gamma=params['focal_loss_gamma'],
            fine_tuning_strategy=params['fine_tuning_strategy'],
            freeze_epochs=params['freeze_epochs'],
            augmentation_config=params['augmentation_config'],
            
            # Optimization-specific settings
            epochs=self.opt_config.max_epochs_per_trial,
            early_stopping=True,
            early_stopping_patience=self.opt_config.early_stopping_rounds,
            validation_frequency=1,
            
            # Experiment naming
            experiment_name=f"{self.opt_config.study_name}_trial_{params.get('trial_number', 'unknown')}"
        )
        
        return trial_config
    
    def _log_trial_results(self, trial: optuna.Trial, params: Dict[str, Any], 
                          results: Dict[str, float], score: float):
        """Log trial results to experiment tracker"""
        if self.experiment_tracker is None:
            return
        
        try:
            # Start MLflow run for this trial
            with self.experiment_tracker.start_run(run_name=f"trial_{trial.number}"):
                # Log hyperparameters
                self.experiment_tracker.log_params(params)
                
                # Log results
                self.experiment_tracker.log_metrics(results)
                
                # Log optimization info
                self.experiment_tracker.log_metric("trial_number", trial.number)
                self.experiment_tracker.log_metric("optimization_score", score)
                
                # Log trial state
                self.experiment_tracker.log_param("trial_state", trial.state.name)
                
        except Exception as e:
            logger.warning(f"Failed to log trial {trial.number} to MLflow: {str(e)}")
    
    def _update_best_trial(self, trial: optuna.Trial, score: float, 
                          params: Dict[str, Any], results: Dict[str, float]):
        """Update best trial tracking"""
        is_better = False
        
        if self.opt_config.direction == 'maximize':
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score
        
        if is_better:
            self.best_score = score
            self.best_trial = {
                'trial_number': trial.number,
                'score': score,
                'params': params.copy(),
                'results': results.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"New best trial {trial.number}: {self.opt_config.primary_metric}={score:.4f}")
            
            # Save best parameters
            if self.opt_config.save_best_params:
                self._save_best_params()
    
    def _save_best_params(self):
        """Save best parameters to file"""
        if self.best_trial is None:
            return
        
        best_params_path = os.path.join(
            self.opt_config.output_dir, 
            f"{self.opt_config.study_name}_best_params.json"
        )
        
        with open(best_params_path, 'w') as f:
            json.dump(self.best_trial, f, indent=2)
        
        logger.info(f"Best parameters saved to {best_params_path}")
    
    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Returns:
            Completed Optuna study
        """
        logger.info(f"Starting hyperparameter optimization: {self.opt_config.study_name}")
        logger.info(f"Target metric: {self.opt_config.primary_metric}")
        logger.info(f"Number of trials: {self.opt_config.n_trials}")
        
        # Create study
        sampler = TPESampler(seed=self.base_config.random_seed)
        
        pruner = None
        if self.opt_config.enable_pruning:
            pruner = MedianPruner(
                n_startup_trials=self.opt_config.pruning_warmup_steps,
                n_warmup_steps=self.opt_config.pruning_warmup_steps,
                interval_steps=self.opt_config.pruning_interval_steps
            )
        
        study = optuna.create_study(
            study_name=self.opt_config.study_name,
            direction=self.opt_config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.opt_config.storage_url
        )
        
        # Run optimization
        try:
            study.optimize(
                self.objective,
                n_trials=self.opt_config.n_trials,
                timeout=self.opt_config.timeout,
                n_jobs=self.opt_config.n_jobs
            )
            
            # Log completion
            logger.info(f"Optimization completed: {len(study.trials)} trials")
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best {self.opt_config.primary_metric}: {study.best_value:.4f}")
            
            # Save study results
            if self.opt_config.save_study_results:
                self._save_study_results(study)
            
            return study
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _save_study_results(self, study: optuna.Study):
        """Save study results and analysis"""
        results_dir = os.path.join(self.opt_config.output_dir, "study_results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Save study summary
        summary_path = os.path.join(results_dir, f"{self.opt_config.study_name}_summary.json")
        summary = {
            'study_name': study.study_name,
            'direction': study.direction.name,
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'optimization_history': [
                {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Study results saved to {results_dir}")
    
    def get_best_config(self) -> Optional[TrainingConfig]:
        """
        Get training config with best hyperparameters
        
        Returns:
            TrainingConfig with optimized parameters
        """
        if self.best_trial is None:
            logger.warning("No best trial available. Run optimization first.")
            return None
        
        return self._create_trial_config(self.best_trial['params'])
    
    def create_production_config(self, extended_epochs: int = 100) -> Optional[TrainingConfig]:
        """
        Create production config with best hyperparameters and extended training
        
        Args:
            extended_epochs: Number of epochs for production training
            
        Returns:
            Production-ready TrainingConfig
        """
        best_config = self.get_best_config()
        if best_config is None:
            return None
        
        # Extend training for production
        best_config.epochs = extended_epochs
        best_config.early_stopping_patience = max(20, extended_epochs // 5)
        best_config.experiment_name = f"{self.opt_config.study_name}_production"
        
        return best_config


def trigger_optimization_on_poor_performance(
    current_accuracy: float,
    base_config: TrainingConfig,
    threshold: float = 0.85,
    experiment_tracker: Optional[ExperimentLogger] = None
) -> Optional[TrainingConfig]:
    """
    Trigger hyperparameter optimization when performance drops below threshold
    
    Args:
        current_accuracy: Current model accuracy
        threshold: Accuracy threshold to trigger optimization
        base_config: Base training configuration
        experiment_tracker: MLflow experiment tracker
        
    Returns:
        Optimized training config or None if optimization not needed
    """
    if current_accuracy >= threshold:
        logger.info(f"Current accuracy {current_accuracy:.3f} above threshold {threshold:.3f}. No optimization needed.")
        return None
    
    logger.warning(f"Current accuracy {current_accuracy:.3f} below threshold {threshold:.3f}. Triggering optimization...")
    
    # Create optimization config
    opt_config = OptimizationConfig(
        n_trials=30,  # Reduced for automatic trigger
        max_epochs_per_trial=15,
        primary_metric='val_accuracy',
        min_accuracy_threshold=threshold,
        study_name=f"auto_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        optimization_config=opt_config,
        experiment_tracker=experiment_tracker
    )
    
    study = optimizer.optimize()
    
    # Return production config with best parameters
    return optimizer.create_production_config()


if __name__ == "__main__":
    # Test hyperparameter optimization
    print("Testing Hyperparameter Optimization...")
    
    # Create base config
    base_config = TrainingConfig(
        dataset_path="data/chest_xray_final",
        epochs=5,  # Short for testing
        batch_size=16
    )
    
    # Create optimization config
    opt_config = OptimizationConfig(
        n_trials=3,  # Very few trials for testing
        max_epochs_per_trial=2,
        study_name="test_optimization"
    )
    
    # Test optimization (would need actual data to run)
    print(f"✅ Hyperparameter optimization system ready!")
    print(f"Base config: {base_config.model_architecture}, LR: {base_config.learning_rate}")
    print(f"Optimization: {opt_config.n_trials} trials, {opt_config.max_epochs_per_trial} epochs each")