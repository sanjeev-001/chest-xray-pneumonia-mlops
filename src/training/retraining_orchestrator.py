"""
Automated Retraining Orchestrator for Chest X-Ray Pneumonia Detection
Handles automated model retraining triggered by performance degradation
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import time
import uuid

# Import training components (will be mocked for demo)
try:
    from .trainer import ModelTrainer
    from .config import TrainingConfig
    from .experiment_tracker import ExperimentTracker
    from .model_registry import ModelRegistry
except ImportError:
    try:
        from trainer import ModelTrainer
        from config import TrainingConfig
        from experiment_tracker import ExperimentTracker
        from model_registry import ModelRegistry
    except ImportError:
        # Create mock classes for demo purposes
        class ModelTrainer:
            def train(self, config): pass
        
        class TrainingConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ExperimentTracker:
            def log_event(self, event_type, data): pass
        
        class ModelRegistry:
            def register_model(self, *args, **kwargs): pass
            def get_model(self, *args, **kwargs): return {}
            def transition_model_stage(self, *args, **kwargs): pass
            def list_model_versions(self, *args, **kwargs): return []
            def delete_model_version(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    """Types of retraining triggers"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NEW_DATA_AVAILABLE = "new_data_available"

class RetrainingStatus(Enum):
    """Status of retraining workflow"""
    PENDING = "pending"
    TRIGGERED = "triggered"
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    COMPARISON = "comparison"
    PROMOTION = "promotion"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RetrainingRequest:
    """Retraining request data structure"""
    request_id: str
    trigger_type: RetrainingTrigger
    triggered_at: datetime
    triggered_by: str
    current_model_id: str
    current_model_version: str
    trigger_data: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RetrainingResult:
    """Result of retraining workflow"""
    request_id: str
    status: RetrainingStatus
    new_model_id: Optional[str]
    new_model_version: Optional[str]
    performance_comparison: Dict[str, float]
    promoted: bool
    started_at: datetime
    completed_at: Optional[datetime]
    duration_minutes: Optional[float]
    error_message: Optional[str] = None
    logs: List[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []

class ModelComparator:
    """Compares model performance for promotion decisions"""
    
    def __init__(self, improvement_threshold: float = 0.02):
        self.improvement_threshold = improvement_threshold
        logger.info(f"ModelComparator initialized with {improvement_threshold:.1%} improvement threshold")
    
    def compare_models(self, 
                      current_metrics: Dict[str, float],
                      new_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare two models and determine if new model should be promoted"""
        
        comparison = {
            "current_model": current_metrics,
            "new_model": new_metrics,
            "improvements": {},
            "degradations": {},
            "overall_improvement": 0.0,
            "should_promote": False,
            "promotion_reason": "",
            "comparison_timestamp": datetime.now().isoformat()
        }
        
        # Primary metrics for comparison (weighted)
        primary_metrics = {
            "accuracy": 0.4,
            "f1_score": 0.3,
            "auc_roc": 0.2,
            "precision": 0.05,
            "recall": 0.05
        }
        
        total_weighted_improvement = 0.0
        
        for metric, weight in primary_metrics.items():
            if metric in current_metrics and metric in new_metrics:
                current_value = current_metrics[metric]
                new_value = new_metrics[metric]
                improvement = new_value - current_value
                improvement_pct = improvement / current_value if current_value > 0 else 0
                
                if improvement > 0:
                    comparison["improvements"][metric] = {
                        "absolute": improvement,
                        "percentage": improvement_pct,
                        "current": current_value,
                        "new": new_value
                    }
                elif improvement < 0:
                    comparison["degradations"][metric] = {
                        "absolute": improvement,
                        "percentage": improvement_pct,
                        "current": current_value,
                        "new": new_value
                    }
                
                # Calculate weighted improvement
                total_weighted_improvement += improvement_pct * weight
        
        comparison["overall_improvement"] = total_weighted_improvement
        
        # Promotion decision logic
        if total_weighted_improvement >= self.improvement_threshold:
            comparison["should_promote"] = True
            comparison["promotion_reason"] = f"Overall improvement of {total_weighted_improvement:.1%} exceeds threshold of {self.improvement_threshold:.1%}"
        else:
            # Check for critical metric degradations
            critical_degradations = []
            for metric in ["accuracy", "f1_score"]:
                if metric in comparison["degradations"]:
                    degradation = comparison["degradations"][metric]
                    if abs(degradation["percentage"]) > 0.05:  # 5% degradation threshold
                        critical_degradations.append(f"{metric} degraded by {abs(degradation['percentage']):.1%}")
            
            if critical_degradations:
                comparison["should_promote"] = False
                comparison["promotion_reason"] = f"Critical degradations detected: {', '.join(critical_degradations)}"
            else:
                comparison["should_promote"] = False
                comparison["promotion_reason"] = f"Improvement of {total_weighted_improvement:.1%} below threshold of {self.improvement_threshold:.1%}"
        
        return comparison

class ModelVersionManager:
    """Manages model versions and cleanup"""
    
    def __init__(self, max_versions: int = 3):
        self.max_versions = max_versions
        logger.info(f"ModelVersionManager initialized to keep {max_versions} versions")
    
    def cleanup_old_versions(self, model_registry: ModelRegistry, model_name: str) -> List[str]:
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            # Get all versions for the model
            versions = model_registry.list_model_versions(model_name)
            
            if len(versions) <= self.max_versions:
                logger.info(f"Model {model_name} has {len(versions)} versions, no cleanup needed")
                return []
            
            # Sort by creation time (newest first)
            versions.sort(key=lambda v: v.get('created_at', ''), reverse=True)
            
            # Keep the most recent versions
            versions_to_keep = versions[:self.max_versions]
            versions_to_delete = versions[self.max_versions:]
            
            deleted_versions = []
            for version in versions_to_delete:
                try:
                    version_id = version.get('version', version.get('id'))
                    model_registry.delete_model_version(model_name, version_id)
                    deleted_versions.append(version_id)
                    logger.info(f"Deleted old model version: {model_name}:{version_id}")
                except Exception as e:
                    logger.error(f"Failed to delete model version {model_name}:{version_id}: {e}")
            
            return deleted_versions
            
        except Exception as e:
            logger.error(f"Failed to cleanup old versions for {model_name}: {e}")
            return []

class RetrainingOrchestrator:
    """Main orchestrator for automated retraining workflows"""
    
    def __init__(self,
                 model_trainer: ModelTrainer,
                 experiment_tracker: ExperimentTracker,
                 model_registry: ModelRegistry,
                 improvement_threshold: float = 0.02,
                 max_concurrent_retraining: int = 1):
        
        self.model_trainer = model_trainer
        self.experiment_tracker = experiment_tracker
        self.model_registry = model_registry
        self.improvement_threshold = improvement_threshold
        self.max_concurrent_retraining = max_concurrent_retraining
        
        # Components
        self.model_comparator = ModelComparator(improvement_threshold)
        self.version_manager = ModelVersionManager()
        
        # State management
        self.active_requests: Dict[str, RetrainingRequest] = {}
        self.completed_requests: Dict[str, RetrainingResult] = {}
        self.request_queue: List[RetrainingRequest] = []
        self.running_workflows = 0
        
        # Threading
        self.lock = threading.Lock()
        self.orchestrator_thread = None
        self.running = False
        
        logger.info("RetrainingOrchestrator initialized")
    
    def trigger_retraining(self, 
                          trigger_type: RetrainingTrigger,
                          current_model_id: str,
                          current_model_version: str,
                          trigger_data: Dict[str, Any],
                          triggered_by: str = "system",
                          priority: int = 1) -> str:
        """Trigger a retraining workflow"""
        
        request = RetrainingRequest(
            request_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            triggered_at=datetime.now(),
            triggered_by=triggered_by,
            current_model_id=current_model_id,
            current_model_version=current_model_version,
            trigger_data=trigger_data,
            priority=priority
        )
        
        with self.lock:
            self.request_queue.append(request)
            self.active_requests[request.request_id] = request
        
        logger.info(f"Retraining triggered: {request.request_id} ({trigger_type.value})")
        
        # Start orchestrator if not running
        if not self.running:
            self.start_orchestrator()
        
        return request.request_id
    
    def start_orchestrator(self):
        """Start the orchestrator background thread"""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.orchestrator_thread = threading.Thread(target=self._orchestrator_loop, daemon=True)
        self.orchestrator_thread.start()
        logger.info("Retraining orchestrator started")
    
    def stop_orchestrator(self):
        """Stop the orchestrator"""
        self.running = False
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=30)
        logger.info("Retraining orchestrator stopped")
    
    def _orchestrator_loop(self):
        """Main orchestrator loop"""
        while self.running:
            try:
                # Process pending requests
                self._process_request_queue()
                
                # Sleep before next iteration
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def _process_request_queue(self):
        """Process pending retraining requests"""
        with self.lock:
            if not self.request_queue or self.running_workflows >= self.max_concurrent_retraining:
                return
            
            # Sort by priority (lower number = higher priority)
            self.request_queue.sort(key=lambda r: (r.priority, r.triggered_at))
            
            # Process highest priority request
            request = self.request_queue.pop(0)
        
        # Execute retraining workflow
        self._execute_retraining_workflow(request)
    
    def _execute_retraining_workflow(self, request: RetrainingRequest):
        """Execute complete retraining workflow"""
        
        with self.lock:
            self.running_workflows += 1
        
        result = RetrainingResult(
            request_id=request.request_id,
            status=RetrainingStatus.TRIGGERED,
            new_model_id=None,
            new_model_version=None,
            performance_comparison={},
            promoted=False,
            started_at=datetime.now(),
            completed_at=None,
            duration_minutes=None
        )
        
        try:
            logger.info(f"Starting retraining workflow: {request.request_id}")
            
            # Step 1: Data Preparation
            result.status = RetrainingStatus.DATA_PREPARATION
            result.logs.append(f"[{datetime.now()}] Starting data preparation")
            
            training_data = self._prepare_training_data(request)
            if not training_data:
                raise Exception("Failed to prepare training data")
            
            # Step 2: Model Training
            result.status = RetrainingStatus.TRAINING
            result.logs.append(f"[{datetime.now()}] Starting model training")
            
            new_model_info = self._train_new_model(request, training_data)
            result.new_model_id = new_model_info["model_id"]
            result.new_model_version = new_model_info["version"]
            
            # Step 3: Model Evaluation
            result.status = RetrainingStatus.EVALUATION
            result.logs.append(f"[{datetime.now()}] Evaluating new model")
            
            new_model_metrics = self._evaluate_new_model(new_model_info)
            
            # Step 4: Model Comparison
            result.status = RetrainingStatus.COMPARISON
            result.logs.append(f"[{datetime.now()}] Comparing models")
            
            current_model_metrics = self._get_current_model_metrics(
                request.current_model_id, 
                request.current_model_version
            )
            
            comparison = self.model_comparator.compare_models(
                current_model_metrics, 
                new_model_metrics
            )
            result.performance_comparison = comparison
            
            # Step 5: Promotion Decision
            if comparison["should_promote"]:
                result.status = RetrainingStatus.PROMOTION
                result.logs.append(f"[{datetime.now()}] Promoting new model")
                
                self._promote_model(new_model_info, comparison)
                result.promoted = True
                
                # Clean up old versions
                self.version_manager.cleanup_old_versions(
                    self.model_registry, 
                    request.current_model_id
                )
            else:
                result.logs.append(f"[{datetime.now()}] Model not promoted: {comparison['promotion_reason']}")
                result.promoted = False
            
            # Step 6: Complete
            result.status = RetrainingStatus.COMPLETED
            result.completed_at = datetime.now()
            result.duration_minutes = (result.completed_at - result.started_at).total_seconds() / 60
            
            logger.info(f"Retraining workflow completed: {request.request_id} (promoted: {result.promoted})")
            
        except Exception as e:
            result.status = RetrainingStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.duration_minutes = (result.completed_at - result.started_at).total_seconds() / 60
            
            logger.error(f"Retraining workflow failed: {request.request_id}: {e}")
        
        finally:
            # Update state
            with self.lock:
                self.running_workflows -= 1
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                self.completed_requests[request.request_id] = result
    
    def _prepare_training_data(self, request: RetrainingRequest) -> Dict[str, Any]:
        """Prepare training data for retraining"""
        try:
            # This would integrate with your data pipeline
            # For now, return mock data structure
            return {
                "train_path": "data/train",
                "val_path": "data/val",
                "test_path": "data/test",
                "data_version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "sample_count": {
                    "train": 5000,
                    "val": 1000,
                    "test": 1000
                }
            }
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None
    
    def _train_new_model(self, request: RetrainingRequest, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train new model with updated data"""
        
        # Create training configuration
        config = TrainingConfig(
            model_name="chest_xray_pneumonia",
            architecture="resnet50",
            num_epochs=20,
            batch_size=32,
            learning_rate=0.001,
            data_path=training_data["train_path"],
            val_data_path=training_data["val_path"],
            experiment_name=f"retraining_{request.request_id[:8]}"
        )
        
        # Start training
        training_result = self.model_trainer.train(config)
        
        # Register new model
        model_info = {
            "model_id": f"chest_xray_pneumonia_retrained",
            "version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "training_result": training_result,
            "config": config,
            "parent_model": f"{request.current_model_id}:{request.current_model_version}",
            "trigger_type": request.trigger_type.value
        }
        
        # Register with model registry
        self.model_registry.register_model(
            model_info["model_id"],
            model_info["version"],
            training_result["model_path"],
            {
                "metrics": training_result["metrics"],
                "config": asdict(config),
                "parent_model": model_info["parent_model"],
                "trigger_type": model_info["trigger_type"],
                "retraining_request_id": request.request_id
            }
        )
        
        return model_info
    
    def _evaluate_new_model(self, model_info: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the newly trained model"""
        # Extract metrics from training result
        training_result = model_info["training_result"]
        
        return {
            "accuracy": training_result["metrics"]["test_accuracy"],
            "precision": training_result["metrics"]["test_precision"],
            "recall": training_result["metrics"]["test_recall"],
            "f1_score": training_result["metrics"]["test_f1"],
            "auc_roc": training_result["metrics"]["test_auc"]
        }
    
    def _get_current_model_metrics(self, model_id: str, model_version: str) -> Dict[str, float]:
        """Get metrics for current production model"""
        try:
            model_info = self.model_registry.get_model(model_id, model_version)
            metrics = model_info.get("metadata", {}).get("metrics", {})
            
            return {
                "accuracy": metrics.get("test_accuracy", 0.0),
                "precision": metrics.get("test_precision", 0.0),
                "recall": metrics.get("test_recall", 0.0),
                "f1_score": metrics.get("test_f1", 0.0),
                "auc_roc": metrics.get("test_auc", 0.0)
            }
        except Exception as e:
            logger.error(f"Failed to get current model metrics: {e}")
            return {}
    
    def _promote_model(self, model_info: Dict[str, Any], comparison: Dict[str, Any]):
        """Promote new model to production"""
        try:
            # Update model stage to production
            self.model_registry.transition_model_stage(
                model_info["model_id"],
                model_info["version"],
                "Production"
            )
            
            # Log promotion event
            self.experiment_tracker.log_event(
                "model_promotion",
                {
                    "model_id": model_info["model_id"],
                    "version": model_info["version"],
                    "comparison": comparison,
                    "promoted_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Model promoted to production: {model_info['model_id']}:{model_info['version']}")
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def get_retraining_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of retraining request"""
        with self.lock:
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                return {
                    "request_id": request_id,
                    "status": "active",
                    "trigger_type": request.trigger_type.value,
                    "triggered_at": request.triggered_at.isoformat(),
                    "triggered_by": request.triggered_by
                }
            
            if request_id in self.completed_requests:
                result = self.completed_requests[request_id]
                return {
                    "request_id": request_id,
                    "status": result.status.value,
                    "promoted": result.promoted,
                    "started_at": result.started_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "duration_minutes": result.duration_minutes,
                    "error_message": result.error_message,
                    "performance_comparison": result.performance_comparison
                }
        
        return None
    
    def list_retraining_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent retraining history"""
        with self.lock:
            results = list(self.completed_requests.values())
        
        # Sort by completion time (newest first)
        results.sort(key=lambda r: r.completed_at or datetime.min, reverse=True)
        
        return [
            {
                "request_id": r.request_id,
                "status": r.status.value,
                "promoted": r.promoted,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_minutes": r.duration_minutes,
                "new_model": f"{r.new_model_id}:{r.new_model_version}" if r.new_model_id else None
            }
            for r in results[:limit]
        ]
    
    def cancel_retraining(self, request_id: str) -> bool:
        """Cancel pending retraining request"""
        with self.lock:
            # Remove from queue if pending
            self.request_queue = [r for r in self.request_queue if r.request_id != request_id]
            
            # Mark as cancelled if active (note: can't stop running training)
            if request_id in self.active_requests:
                # For now, we can only cancel queued requests
                # Running training cannot be easily cancelled
                return True
        
        return False

# Global orchestrator instance
_retraining_orchestrator = None

def get_retraining_orchestrator(
    model_trainer: ModelTrainer = None,
    experiment_tracker: ExperimentTracker = None,
    model_registry: ModelRegistry = None,
    improvement_threshold: float = 0.02
) -> RetrainingOrchestrator:
    """Get global retraining orchestrator instance"""
    global _retraining_orchestrator
    
    if _retraining_orchestrator is None:
        if not all([model_trainer, experiment_tracker, model_registry]):
            raise ValueError("Must provide all components for first initialization")
        
        _retraining_orchestrator = RetrainingOrchestrator(
            model_trainer=model_trainer,
            experiment_tracker=experiment_tracker,
            model_registry=model_registry,
            improvement_threshold=improvement_threshold
        )
    
    return _retraining_orchestrator

# Convenience functions for triggering retraining
def trigger_performance_retraining(
    current_model_id: str,
    current_model_version: str,
    performance_metrics: Dict[str, float],
    threshold_violations: List[str]
) -> str:
    """Trigger retraining due to performance degradation"""
    
    orchestrator = get_retraining_orchestrator()
    
    return orchestrator.trigger_retraining(
        trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
        current_model_id=current_model_id,
        current_model_version=current_model_version,
        trigger_data={
            "performance_metrics": performance_metrics,
            "threshold_violations": threshold_violations,
            "trigger_timestamp": datetime.now().isoformat()
        },
        triggered_by="performance_monitor",
        priority=1  # High priority
    )

def trigger_drift_retraining(
    current_model_id: str,
    current_model_version: str,
    drift_report: Dict[str, Any]
) -> str:
    """Trigger retraining due to data/concept drift"""
    
    orchestrator = get_retraining_orchestrator()
    
    drift_type = RetrainingTrigger.DATA_DRIFT
    if drift_report.get("concept_drift_score", 0) > drift_report.get("data_drift_score", 0):
        drift_type = RetrainingTrigger.CONCEPT_DRIFT
    
    return orchestrator.trigger_retraining(
        trigger_type=drift_type,
        current_model_id=current_model_id,
        current_model_version=current_model_version,
        trigger_data={
            "drift_report": drift_report,
            "trigger_timestamp": datetime.now().isoformat()
        },
        triggered_by="drift_detector",
        priority=2  # Medium priority
    )

def trigger_scheduled_retraining(
    current_model_id: str,
    current_model_version: str,
    schedule_info: Dict[str, Any]
) -> str:
    """Trigger scheduled retraining"""
    
    orchestrator = get_retraining_orchestrator()
    
    return orchestrator.trigger_retraining(
        trigger_type=RetrainingTrigger.SCHEDULED,
        current_model_id=current_model_id,
        current_model_version=current_model_version,
        trigger_data={
            "schedule_info": schedule_info,
            "trigger_timestamp": datetime.now().isoformat()
        },
        triggered_by="scheduler",
        priority=3  # Low priority
    )