"""
MLflow Experiment Tracking for Medical Model Training
Comprehensive experiment logging and tracking for chest X-ray models
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import torch
import numpy as np
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .config import TrainingConfig
from .metrics import MedicalMetrics

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    MLflow-based experiment logger for medical model training
    Handles comprehensive logging of experiments, metrics, and artifacts
    """
    
    def __init__(
        self, 
        experiment_name: str,
        tracking_uri: str = "sqlite:///mlflow.db",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location to store artifacts
        """
        
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Created new experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        # MLflow client for advanced operations
        self.client = MlflowClient()
        
        # Current run tracking
        self.current_run = None
        self.run_id = None
        
        logger.info(f"ExperimentLogger initialized for experiment: {experiment_name}")
        logger.info(f"Tracking URI: {tracking_uri}")
        logger.info(f"Experiment ID: {self.experiment_id}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        
        # End current run if exists
        if self.current_run is not None:
            self.end_run()
        
        # Start new run
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        
        self.run_id = self.current_run.info.run_id
        
        logger.info(f"Started MLflow run: {self.run_id}")
        if run_name:
            logger.info(f"Run name: {run_name}")
        
        return self.run_id
    
    def end_run(self):
        """End the current MLflow run"""
        if self.current_run is not None:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")
            self.current_run = None
            self.run_id = None
    
    def log_config(self, config: TrainingConfig):
        """
        Log training configuration
        
        Args:
            config: Training configuration object
        """
        
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Log hyperparameters
        config_dict = config.to_dict()
        
        # Log scalar parameters
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            elif isinstance(value, (list, tuple)) and len(value) <= 10:
                mlflow.log_param(key, str(value))
        
        # Log complex parameters as JSON artifact
        complex_params = {
            k: v for k, v in config_dict.items() 
            if not isinstance(v, (int, float, str, bool, type(None)))
        }
        
        if complex_params:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(complex_params, f, indent=2, default=str)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "config/complex_params.json")
            Path(temp_path).unlink()  # Clean up temp file
        
        # Log full config as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f, indent=2, default=str)
            temp_path = f.name
        
        mlflow.log_artifact(temp_path, "config/full_config.json")
        Path(temp_path).unlink()
        
        logger.info("Logged training configuration to MLflow")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (epoch, iteration, etc.)
        """
        
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                mlflow.log_metric(name, float(value), step=step)
        
        if step is not None:
            logger.debug(f"Logged metrics for step {step}: {list(metrics.keys())}")
        else:
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
    
    def log_medical_metrics(self, medical_metrics: Dict[str, Any], prefix: str = ""):
        """
        Log medical-specific metrics
        
        Args:
            medical_metrics: Medical metrics from MedicalMetrics.calculate_metrics()
            prefix: Optional prefix for metric names (e.g., "test_", "val_")
        """
        
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Log scalar metrics
        scalar_metrics = {}
        
        for key, value in medical_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_name = f"{prefix}{key}" if prefix else key
                scalar_metrics[metric_name] = float(value)
        
        self.log_metrics(scalar_metrics)
        
        # Log confusion matrix as artifact if present
        if 'confusion_matrix' in medical_metrics:
            self._log_confusion_matrix(medical_metrics['confusion_matrix'], prefix)
        
        # Log classification report as artifact if present
        if 'classification_report' in medical_metrics:
            self._log_classification_report(medical_metrics['classification_report'], prefix)
        
        logger.info(f"Logged medical metrics with prefix '{prefix}'")
    
    def _log_confusion_matrix(self, cm: List[List[int]], prefix: str = ""):
        """Log confusion matrix as artifact"""
        try:
            cm_array = np.array(cm)
            
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{prefix}Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Save and log
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=300, bbox_inches='tight')
                temp_path = f.name
            
            artifact_path = f"plots/{prefix}confusion_matrix.png"
            mlflow.log_artifact(temp_path, artifact_path)
            Path(temp_path).unlink()
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")
    
    def _log_classification_report(self, report: Dict[str, Any], prefix: str = ""):
        """Log classification report as artifact"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(report, f, indent=2, default=str)
                temp_path = f.name
            
            artifact_path = f"reports/{prefix}classification_report.json"
            mlflow.log_artifact(temp_path, artifact_path)
            Path(temp_path).unlink()
            
        except Exception as e:
            logger.warning(f"Failed to log classification report: {e}")
    
    def log_model(
        self, 
        model: torch.nn.Module, 
        model_name: str = "model",
        signature: Optional[mlflow.types.Schema] = None,
        input_example: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log PyTorch model to MLflow
        
        Args:
            model: PyTorch model to log
            model_name: Name for the model
            signature: MLflow model signature
            input_example: Example input for the model
            metadata: Additional metadata
        """
        
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        try:
            # Log the model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                metadata=metadata
            )
            
            logger.info(f"Logged PyTorch model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact to MLflow
        
        Args:
            local_path: Path to local file/directory
            artifact_path: Path within the artifact store
        """
        
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_training_plots(self, history: Dict[str, List[float]], save_dir: str = None):
        """
        Log training history plots
        
        Args:
            history: Training history dictionary
            save_dir: Optional directory to save plots locally
        """
        
        if not history or len(history.get('train_loss', [])) == 0:
            logger.warning("No training history to plot")
            return
        
        try:
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training History', fontsize=16)
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Loss plot
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            if 'val_loss' in history and history['val_loss']:
                val_epochs = range(1, len(history['val_loss']) + 1)
                axes[0, 0].plot(val_epochs, history['val_loss'], 'r-', label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy plot
            if 'train_acc' in history:
                axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
                if 'val_acc' in history and history['val_acc']:
                    axes[0, 1].plot(val_epochs, history['val_acc'], 'r-', label='Validation Accuracy')
                axes[0, 1].set_title('Model Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Learning rate plot
            if 'learning_rates' in history:
                axes[1, 0].plot(epochs, history['learning_rates'], 'g-')
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True)
            
            # Epoch time plot
            if 'epoch_times' in history:
                axes[1, 1].plot(epochs, history['epoch_times'], 'm-')
                axes[1, 1].set_title('Epoch Training Time')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save and log
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=300, bbox_inches='tight')
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "plots/training_history.png")
            
            # Save locally if requested
            if save_dir:
                save_path = Path(save_dir) / "training_history.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp_path, save_path)
            
            Path(temp_path).unlink()
            plt.close()
            
            logger.info("Logged training history plots")
            
        except Exception as e:
            logger.warning(f"Failed to log training plots: {e}")
    
    def get_experiment_runs(self, max_results: int = 100) -> List[mlflow.entities.Run]:
        """
        Get runs from the current experiment
        
        Args:
            max_results: Maximum number of runs to return
            
        Returns:
            List of MLflow runs
        """
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        return runs
    
    def get_best_run(self, metric_name: str = "val_accuracy", ascending: bool = False) -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether to sort in ascending order (True for loss, False for accuracy)
            
        Returns:
            Best run or None if no runs found
        """
        
        runs = self.get_experiment_runs()
        
        if not runs:
            return None
        
        # Filter runs that have the metric
        runs_with_metric = [
            run for run in runs 
            if metric_name in run.data.metrics
        ]
        
        if not runs_with_metric:
            logger.warning(f"No runs found with metric: {metric_name}")
            return None
        
        # Sort by metric
        best_run = sorted(
            runs_with_metric,
            key=lambda r: r.data.metrics[metric_name],
            reverse=not ascending
        )[0]
        
        logger.info(f"Best run by {metric_name}: {best_run.info.run_id}")
        logger.info(f"Best {metric_name}: {best_run.data.metrics[metric_name]:.4f}")
        
        return best_run
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple runs on specified metrics
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to compare
            
        Returns:
            Dictionary of run_id -> {metric_name: value}
        """
        
        comparison = {}
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                comparison[run_id] = {}
                
                for metric in metrics:
                    if metric in run.data.metrics:
                        comparison[run_id][metric] = run.data.metrics[metric]
                    else:
                        comparison[run_id][metric] = None
                        
            except Exception as e:
                logger.warning(f"Failed to get run {run_id}: {e}")
                comparison[run_id] = {metric: None for metric in metrics}
        
        return comparison
    
    def cleanup(self):
        """Cleanup resources"""
        if self.current_run is not None:
            self.end_run()


def setup_mlflow_tracking(
    experiment_name: str,
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_location: Optional[str] = None
) -> ExperimentLogger:
    """
    Setup MLflow tracking for medical model experiments
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking server URI
        artifact_location: Location to store artifacts
        
    Returns:
        Configured ExperimentLogger
    """
    
    logger.info(f"Setting up MLflow tracking for experiment: {experiment_name}")
    
    # Create experiment logger
    experiment_logger = ExperimentLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location
    )
    
    logger.info("MLflow tracking setup completed")
    
    return experiment_logger


if __name__ == "__main__":
    # Test experiment logger
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MLflow ExperimentLogger...")
    
    # Create test experiment
    logger_test = ExperimentLogger("test_experiment")
    
    # Start a run
    run_id = logger_test.start_run("test_run")
    print(f"Started run: {run_id}")
    
    # Log some test metrics
    test_metrics = {
        'accuracy': 0.85,
        'loss': 0.45,
        'auc_roc': 0.92
    }
    
    logger_test.log_metrics(test_metrics, step=1)
    print("Logged test metrics")
    
    # End run
    logger_test.end_run()
    print("Ended run")
    
    # Get experiment runs
    runs = logger_test.get_experiment_runs()
    print(f"Found {len(runs)} runs in experiment")
    
    print("✅ ExperimentLogger working correctly!")