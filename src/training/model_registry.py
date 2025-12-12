"""
MLflow Model Registry for Medical Models
Comprehensive model versioning and lifecycle management
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import torch
import numpy as np
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import tempfile
from datetime import datetime
from enum import Enum
import shutil

from .config import TrainingConfig
from .metrics import MedicalMetrics

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    NONE = "None"

class ModelRegistry:
    """
    MLflow-based model registry for medical models
    Handles model versioning, staging, and lifecycle management
    """
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """
        Initialize model registry
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        logger.info(f"ModelRegistry initialized with tracking URI: {tracking_uri}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        await_registration_for: int = 300
    ) -> ModelVersion:
        """
        Register a model in the MLflow Model Registry
        
        Args:
            model_uri: URI of the model (e.g., "runs:/<run_id>/model")
            model_name: Name to register the model under
            description: Optional description of the model
            tags: Optional tags for the model version
            await_registration_for: Time to wait for registration (seconds)
            
        Returns:
            ModelVersion object
        """
        
        try:
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                await_registration_for=await_registration_for
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )
            
            logger.info(f"Registered model '{model_name}' version {model_version.version}")
            logger.info(f"Model URI: {model_uri}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def create_registered_model(
        self,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Create a new registered model
        
        Args:
            model_name: Name of the model
            description: Optional description
            tags: Optional tags
        """
        
        try:
            # Check if model already exists
            try:
                existing_model = self.client.get_registered_model(model_name)
                logger.info(f"Model '{model_name}' already exists")
                return existing_model
            except mlflow.exceptions.RestException:
                pass  # Model doesn't exist, create it
            
            # Create the registered model
            registered_model = self.client.create_registered_model(
                name=model_name,
                description=description,
                tags=tags
            )
            
            logger.info(f"Created registered model: {model_name}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Failed to create registered model: {e}")
            raise
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get a specific model version
        
        Args:
            model_name: Name of the registered model
            version: Specific version number (e.g., "1", "2")
            stage: Stage name (e.g., "Production", "Staging")
            
        Returns:
            ModelVersion object or None if not found
        """
        
        try:
            if version:
                # Get specific version
                model_version = self.client.get_model_version(
                    name=model_name,
                    version=version
                )
                return model_version
            
            elif stage:
                # Get latest version in stage
                model_versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
                return model_versions[0] if model_versions else None
            
            else:
                # Get latest version
                model_versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=["None", "Staging", "Production"]
                )
                if model_versions:
                    # Return the highest version number
                    return max(model_versions, key=lambda v: int(v.version))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True
    ) -> ModelVersion:
        """
        Transition a model version to a new stage
        
        Args:
            model_name: Name of the registered model
            version: Version to transition
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing_versions: Whether to archive existing versions in target stage
            
        Returns:
            Updated ModelVersion
        """
        
        try:
            # Archive existing versions in target stage if requested
            if archive_existing_versions and stage in ["Staging", "Production"]:
                existing_versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
                
                for existing_version in existing_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=existing_version.version,
                        stage="Archived"
                    )
                    logger.info(f"Archived model version {existing_version.version}")
            
            # Transition to new stage
            model_version = self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Transitioned model '{model_name}' version {version} to {stage}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load
            stage: Stage to load from ("Production", "Staging")
            
        Returns:
            Loaded PyTorch model
        """
        
        try:
            # Construct model URI
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Load the model
            model = mlflow.pytorch.load_model(model_uri)
            
            logger.info(f"Loaded model from URI: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_metadata(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for a model version
        
        Args:
            model_name: Name of the registered model
            version: Specific version
            stage: Stage name
            
        Returns:
            Dictionary with model metadata
        """
        
        model_version = self.get_model_version(model_name, version, stage)
        
        if not model_version:
            return {}
        
        # Get run information
        run = self.client.get_run(model_version.run_id)
        
        metadata = {
            'model_name': model_version.name,
            'version': model_version.version,
            'stage': model_version.current_stage,
            'description': model_version.description,
            'creation_timestamp': model_version.creation_timestamp,
            'last_updated_timestamp': model_version.last_updated_timestamp,
            'run_id': model_version.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'source': model_version.source,
            'tags': model_version.tags,
            'metrics': run.data.metrics,
            'params': run.data.params
        }
        
        return metadata
    
    def compare_model_versions(
        self,
        model_name: str,
        versions: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple versions of a model
        
        Args:
            model_name: Name of the registered model
            versions: List of version numbers to compare
            metrics: List of metric names to compare
            
        Returns:
            Dictionary with comparison data
        """
        
        comparison = {}
        
        for version in versions:
            try:
                metadata = self.get_model_metadata(model_name, version=version)
                
                comparison[version] = {
                    'stage': metadata.get('stage', 'Unknown'),
                    'creation_timestamp': metadata.get('creation_timestamp'),
                    'run_id': metadata.get('run_id'),
                    'metrics': {}
                }
                
                # Extract requested metrics
                for metric in metrics:
                    comparison[version]['metrics'][metric] = metadata.get('metrics', {}).get(metric)
                
            except Exception as e:
                logger.warning(f"Failed to get metadata for version {version}: {e}")
                comparison[version] = {'error': str(e)}
        
        return comparison
    
    def get_production_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Get the current production model
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Production model or None if no production model exists
        """
        
        try:
            return self.load_model(model_name, stage="Production")
        except Exception as e:
            logger.warning(f"No production model found for {model_name}: {e}")
            return None
    
    def promote_to_production(
        self,
        model_name: str,
        version: str,
        validation_metrics: Optional[Dict[str, float]] = None,
        min_accuracy: float = 0.85
    ) -> bool:
        """
        Promote a model version to production with validation
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            validation_metrics: Metrics to validate before promotion
            min_accuracy: Minimum accuracy required for promotion
            
        Returns:
            True if promotion successful, False otherwise
        """
        
        try:
            # Validate metrics if provided
            if validation_metrics:
                accuracy = validation_metrics.get('accuracy', 0.0)
                if accuracy < min_accuracy:
                    logger.warning(f"Model accuracy {accuracy:.3f} below minimum {min_accuracy:.3f}")
                    return False
                
                # Log validation metrics as tags
                for metric_name, metric_value in validation_metrics.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=f"validation_{metric_name}",
                        value=str(metric_value)
                    )
            
            # Promote to production
            self.transition_model_stage(
                model_name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            
            # Add promotion timestamp
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_to_production_at",
                value=datetime.now().isoformat()
            )
            
            logger.info(f"Successfully promoted model '{model_name}' version {version} to Production")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model to production: {e}")
            return False
    
    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Returns:
            List of registered model information
        """
        
        try:
            registered_models = self.client.search_registered_models()
            
            models_info = []
            for model in registered_models:
                # Get latest versions
                latest_versions = self.client.get_latest_versions(
                    name=model.name,
                    stages=["None", "Staging", "Production", "Archived"]
                )
                
                model_info = {
                    'name': model.name,
                    'description': model.description,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'tags': model.tags,
                    'latest_versions': {
                        version.current_stage: {
                            'version': version.version,
                            'creation_timestamp': version.creation_timestamp,
                            'run_id': version.run_id
                        }
                        for version in latest_versions
                    }
                }
                
                models_info.append(model_info)
            
            return models_info
            
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []
    
    def delete_model_version(self, model_name: str, version: str):
        """
        Delete a specific model version
        
        Args:
            model_name: Name of the registered model
            version: Version to delete
        """
        
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted model '{model_name}' version {version}")
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise
    
    def delete_registered_model(self, model_name: str):
        """
        Delete an entire registered model
        
        Args:
            model_name: Name of the registered model to delete
        """
        
        try:
            self.client.delete_registered_model(name=model_name)
            logger.info(f"Deleted registered model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete registered model: {e}")
            raise


class AutoModelRegistry:
    """
    Automated model registry with intelligent promotion logic
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        model_name: str,
        promotion_criteria: Optional[Dict[str, float]] = None
    ):
        """
        Initialize auto model registry
        
        Args:
            model_registry: ModelRegistry instance
            model_name: Name of the model to manage
            promotion_criteria: Criteria for automatic promotion
        """
        
        self.registry = model_registry
        self.model_name = model_name
        self.promotion_criteria = promotion_criteria or {
            'accuracy': 0.85,
            'auc_roc': 0.80,
            'sensitivity': 0.80,
            'specificity': 0.80
        }
        
        # Ensure registered model exists
        self.registry.create_registered_model(
            model_name=model_name,
            description=f"Automated registry for {model_name}"
        )
        
        logger.info(f"AutoModelRegistry initialized for model: {model_name}")
    
    def register_and_evaluate(
        self,
        model_uri: str,
        validation_metrics: Dict[str, float],
        description: Optional[str] = None,
        auto_promote: bool = True
    ) -> Tuple[ModelVersion, bool]:
        """
        Register model and evaluate for promotion
        
        Args:
            model_uri: URI of the model to register
            validation_metrics: Validation metrics for the model
            description: Optional description
            auto_promote: Whether to automatically promote if criteria met
            
        Returns:
            Tuple of (ModelVersion, promoted_to_production)
        """
        
        # Register the model
        model_version = self.registry.register_model(
            model_uri=model_uri,
            model_name=self.model_name,
            description=description,
            tags={
                'auto_registered': 'true',
                'registration_timestamp': datetime.now().isoformat()
            }
        )
        
        # Evaluate for promotion
        promoted = False
        if auto_promote:
            promoted = self._evaluate_for_promotion(model_version.version, validation_metrics)
        
        return model_version, promoted
    
    def _evaluate_for_promotion(
        self,
        version: str,
        validation_metrics: Dict[str, float]
    ) -> bool:
        """
        Evaluate if model should be promoted to production
        
        Args:
            version: Model version to evaluate
            validation_metrics: Validation metrics
            
        Returns:
            True if promoted, False otherwise
        """
        
        # Check if all criteria are met
        criteria_met = True
        for metric_name, min_value in self.promotion_criteria.items():
            actual_value = validation_metrics.get(metric_name, 0.0)
            if actual_value < min_value:
                criteria_met = False
                logger.info(f"Promotion criteria not met: {metric_name} = {actual_value:.3f} < {min_value:.3f}")
        
        if not criteria_met:
            # Stage as staging for manual review
            self.registry.transition_model_stage(
                model_name=self.model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"Model staged for manual review (criteria not met)")
            return False
        
        # Compare with current production model
        current_production = self.registry.get_model_version(
            model_name=self.model_name,
            stage="Production"
        )
        
        if current_production:
            # Get current production metrics
            current_metadata = self.registry.get_model_metadata(
                model_name=self.model_name,
                stage="Production"
            )
            
            current_metrics = current_metadata.get('metrics', {})
            
            # Compare key metrics
            improvement_found = False
            for metric_name in ['accuracy', 'auc_roc', 'f1_score']:
                if metric_name in validation_metrics and metric_name in current_metrics:
                    new_value = validation_metrics[metric_name]
                    current_value = current_metrics[metric_name]
                    
                    if new_value > current_value + 0.01:  # At least 1% improvement
                        improvement_found = True
                        logger.info(f"Improvement found in {metric_name}: {new_value:.3f} > {current_value:.3f}")
                        break
            
            if not improvement_found:
                logger.info("No significant improvement over current production model")
                return False
        
        # Promote to production
        success = self.registry.promote_to_production(
            model_name=self.model_name,
            version=version,
            validation_metrics=validation_metrics
        )
        
        return success


if __name__ == "__main__":
    # Test model registry
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MLflow Model Registry...")
    
    # Create model registry
    registry = ModelRegistry()
    
    # Create a test registered model
    model_name = "test_chest_xray_model"
    registry.create_registered_model(
        model_name=model_name,
        description="Test model for chest X-ray classification"
    )
    
    # List registered models
    models = registry.list_registered_models()
    print(f"Found {len(models)} registered models")
    
    # Test auto registry
    auto_registry = AutoModelRegistry(
        model_registry=registry,
        model_name=model_name
    )
    
    print("✅ Model Registry working correctly!")