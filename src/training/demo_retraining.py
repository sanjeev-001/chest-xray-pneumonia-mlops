"""
Demo script for Automated Retraining System
Shows how the retraining orchestrator and integration service work
"""

import logging
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import retraining components
from retraining_orchestrator import (
    RetrainingOrchestrator, RetrainingTrigger, ModelComparator,
    ModelVersionManager, trigger_performance_retraining, trigger_drift_retraining
)
from retraining_integration import (
    RetrainingIntegrationService, RetrainingPolicy
)

def create_mock_components():
    """Create mock components for demonstration"""
    
    # Mock ModelTrainer
    mock_trainer = Mock()
    mock_trainer.train.return_value = {
        "model_path": "/models/retrained_model.pth",
        "metrics": {
            "test_accuracy": 0.92,
            "test_precision": 0.90,
            "test_recall": 0.89,
            "test_f1": 0.895,
            "test_auc": 0.94
        },
        "training_time": 3600,
        "epochs_completed": 20
    }
    
    # Mock ExperimentTracker
    mock_experiment_tracker = Mock()
    mock_experiment_tracker.log_event.return_value = True
    
    # Mock ModelRegistry
    mock_model_registry = Mock()
    mock_model_registry.register_model.return_value = "model_registered"
    mock_model_registry.get_model.return_value = {
        "metadata": {
            "metrics": {
                "test_accuracy": 0.85,
                "test_precision": 0.83,
                "test_recall": 0.82,
                "test_f1": 0.825,
                "test_auc": 0.87
            }
        }
    }
    mock_model_registry.transition_model_stage.return_value = True
    mock_model_registry.list_model_versions.return_value = [
        {"version": "v1.0", "created_at": "2023-01-01T00:00:00"},
        {"version": "v1.1", "created_at": "2023-02-01T00:00:00"},
        {"version": "v1.2", "created_at": "2023-03-01T00:00:00"},
        {"version": "v1.3", "created_at": "2023-04-01T00:00:00"}
    ]
    mock_model_registry.delete_model_version.return_value = True
    
    return mock_trainer, mock_experiment_tracker, mock_model_registry

def demo_model_comparison():
    """Demonstrate model comparison functionality"""
    print("\n" + "="*60)
    print("DEMO: Model Comparison")
    print("="*60)
    
    comparator = ModelComparator(improvement_threshold=0.02)
    
    # Current model metrics (baseline)
    current_metrics = {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.82,
        "f1_score": 0.825,
        "auc_roc": 0.87
    }
    
    # Test Case 1: Significant improvement
    print("\nTest Case 1: Significant Improvement")
    new_metrics_good = {
        "accuracy": 0.92,  # +7% improvement
        "precision": 0.90,
        "recall": 0.89,
        "f1_score": 0.895,
        "auc_roc": 0.94
    }
    
    comparison = comparator.compare_models(current_metrics, new_metrics_good)
    print(f"Should promote: {comparison['should_promote']}")
    print(f"Overall improvement: {comparison['overall_improvement']:.1%}")
    print(f"Reason: {comparison['promotion_reason']}")
    
    # Test Case 2: Minor improvement (below threshold)
    print("\nTest Case 2: Minor Improvement")
    new_metrics_minor = {
        "accuracy": 0.86,  # +1% improvement
        "precision": 0.84,
        "recall": 0.83,
        "f1_score": 0.835,
        "auc_roc": 0.88
    }
    
    comparison = comparator.compare_models(current_metrics, new_metrics_minor)
    print(f"Should promote: {comparison['should_promote']}")
    print(f"Overall improvement: {comparison['overall_improvement']:.1%}")
    print(f"Reason: {comparison['promotion_reason']}")
    
    # Test Case 3: Performance degradation
    print("\nTest Case 3: Performance Degradation")
    new_metrics_worse = {
        "accuracy": 0.78,  # -7% degradation
        "precision": 0.76,
        "recall": 0.75,
        "f1_score": 0.755,
        "auc_roc": 0.80
    }
    
    comparison = comparator.compare_models(current_metrics, new_metrics_worse)
    print(f"Should promote: {comparison['should_promote']}")
    print(f"Overall improvement: {comparison['overall_improvement']:.1%}")
    print(f"Reason: {comparison['promotion_reason']}")

def demo_version_management():
    """Demonstrate model version management"""
    print("\n" + "="*60)
    print("DEMO: Model Version Management")
    print("="*60)
    
    # Create mock model registry
    _, _, mock_model_registry = create_mock_components()
    
    version_manager = ModelVersionManager(max_versions=3)
    
    print("Current model versions:")
    versions = mock_model_registry.list_model_versions("chest_xray_model")
    for version in versions:
        print(f"  - {version['version']} (created: {version['created_at']})")
    
    print(f"\nCleaning up old versions (keeping {version_manager.max_versions} most recent)...")
    deleted_versions = version_manager.cleanup_old_versions(
        mock_model_registry, 
        "chest_xray_model"
    )
    
    print(f"Deleted versions: {deleted_versions}")

def demo_retraining_orchestrator():
    """Demonstrate retraining orchestrator functionality"""
    print("\n" + "="*60)
    print("DEMO: Retraining Orchestrator")
    print("="*60)
    
    # Create mock components
    mock_trainer, mock_experiment_tracker, mock_model_registry = create_mock_components()
    
    # Create orchestrator
    orchestrator = RetrainingOrchestrator(
        model_trainer=mock_trainer,
        experiment_tracker=mock_experiment_tracker,
        model_registry=mock_model_registry,
        improvement_threshold=0.02,
        max_concurrent_retraining=1
    )
    
    print("Retraining orchestrator created")
    
    # Trigger performance-based retraining
    print("\nTriggering performance-based retraining...")
    request_id = orchestrator.trigger_retraining(
        trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
        current_model_id="chest_xray_model",
        current_model_version="v1.0",
        trigger_data={
            "performance_metrics": {"accuracy": 0.75, "f1_score": 0.72},
            "threshold_violations": ["Accuracy: 0.75 < 0.80", "F1: 0.72 < 0.75"]
        },
        triggered_by="performance_monitor"
    )
    
    print(f"Retraining request created: {request_id}")
    
    # Check status
    print("\nChecking retraining status...")
    status = orchestrator.get_retraining_status(request_id)
    if status:
        print(f"Status: {status['status']}")
        print(f"Trigger type: {status['trigger_type']}")
        print(f"Triggered by: {status['triggered_by']}")
    
    # Start orchestrator (in demo, we'll simulate completion)
    print("\nStarting orchestrator...")
    orchestrator.start_orchestrator()
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Simulate completion by manually updating the result
    print("\nSimulating retraining completion...")
    
    # Stop orchestrator
    orchestrator.stop_orchestrator()
    
    # Show retraining history
    print("\nRetraining history:")
    history = orchestrator.list_retraining_history(limit=5)
    for entry in history:
        print(f"  - {entry['request_id'][:8]}... ({entry['status']}) - {entry.get('new_model', 'N/A')}")

def demo_retraining_integration():
    """Demonstrate retraining integration with monitoring"""
    print("\n" + "="*60)
    print("DEMO: Retraining Integration")
    print("="*60)
    
    # Create mock components
    mock_trainer, mock_experiment_tracker, mock_model_registry = create_mock_components()
    
    # Create orchestrator
    orchestrator = RetrainingOrchestrator(
        model_trainer=mock_trainer,
        experiment_tracker=mock_experiment_tracker,
        model_registry=mock_model_registry
    )
    
    # Create integration service with custom policy
    policy = RetrainingPolicy(
        accuracy_threshold=0.80,
        min_alerts_for_retraining=2,
        alert_window_hours=1,
        max_retraining_per_day=3
    )
    
    integration_service = RetrainingIntegrationService(
        retraining_orchestrator=orchestrator,
        policy=policy
    )
    
    print("Integration service created with policy:")
    print(f"  - Accuracy threshold: {policy.accuracy_threshold}")
    print(f"  - Min alerts for retraining: {policy.min_alerts_for_retraining}")
    print(f"  - Alert window: {policy.alert_window_hours} hours")
    
    # Create mock performance alerts
    from retraining_integration import PerformanceAlert, AlertType, AlertSeverity
    
    # Simulate accuracy drop alerts
    print("\nSimulating accuracy drop alerts...")
    
    alert1 = PerformanceAlert(
        alert_type=AlertType.ACCURACY_DROP,
        severity=AlertSeverity.CRITICAL,
        timestamp=datetime.now(),
        current_value=0.75,
        threshold=0.80,
        metric_name="accuracy"
    )
    
    alert2 = PerformanceAlert(
        alert_type=AlertType.ACCURACY_DROP,
        severity=AlertSeverity.CRITICAL,
        timestamp=datetime.now(),
        current_value=0.73,
        threshold=0.80,
        metric_name="accuracy"
    )
    
    # Handle alerts
    request_id1 = integration_service.handle_performance_alert(alert1)
    print(f"Alert 1 handled: {request_id1 or 'No retraining triggered'}")
    
    request_id2 = integration_service.handle_performance_alert(alert2)
    print(f"Alert 2 handled: {request_id2 or 'No retraining triggered'}")
    
    # Show statistics
    print("\nRetraining statistics:")
    stats = integration_service.get_retraining_statistics()
    print(f"  - Current model: {stats['current_model']['model_id']}:{stats['current_model']['model_version']}")
    print(f"  - Alerts last 24h: {stats['recent_activity']['alerts_last_24h']}")
    print(f"  - Retraining requests today: {stats['recent_activity']['retraining_count_today']}")
    print(f"  - Total alerts: {stats['alert_summary']['total_alerts']}")
    print(f"  - Critical alerts: {stats['alert_summary']['critical_alerts']}")

def demo_retraining_scenarios():
    """Demonstrate different retraining scenarios"""
    print("\n" + "="*60)
    print("DEMO: Retraining Scenarios")
    print("="*60)
    
    # Create integration service
    mock_trainer, mock_experiment_tracker, mock_model_registry = create_mock_components()
    orchestrator = RetrainingOrchestrator(mock_trainer, mock_experiment_tracker, mock_model_registry)
    integration_service = RetrainingIntegrationService(orchestrator)
    
    # Scenario 1: Critical accuracy drop
    print("\nScenario 1: Critical accuracy drop")
    scenario1 = {
        "accuracy_drop": 0.65,  # Well below threshold
        "last_retraining_time": None,
        "retraining_count_today": 0
    }
    
    result1 = integration_service.simulate_retraining_scenario(scenario1)
    print(f"  Should trigger: {result1['decision']['should_trigger']}")
    print(f"  Reason: {result1['decision']['reason']}")
    print(f"  Priority: {result1['decision']['priority']}")
    
    # Scenario 2: Recent retraining (cooldown)
    print("\nScenario 2: Recent retraining (cooldown)")
    scenario2 = {
        "accuracy_drop": 0.70,
        "last_retraining_time": datetime.now() - timedelta(hours=12),  # 12 hours ago
        "retraining_count_today": 1
    }
    
    result2 = integration_service.simulate_retraining_scenario(scenario2)
    print(f"  Should trigger: {result2['decision']['should_trigger']}")
    print(f"  Reason: {result2['decision']['reason']}")
    print(f"  Blocking factors: {result2['decision']['blocking_factors']}")
    
    # Scenario 3: Daily limit reached
    print("\nScenario 3: Daily limit reached")
    scenario3 = {
        "accuracy_drop": 0.60,
        "last_retraining_time": datetime.now() - timedelta(hours=25),  # Yesterday
        "retraining_count_today": 2  # At daily limit
    }
    
    result3 = integration_service.simulate_retraining_scenario(scenario3)
    print(f"  Should trigger: {result3['decision']['should_trigger']}")
    print(f"  Reason: {result3['decision']['reason']}")
    print(f"  Blocking factors: {result3['decision']['blocking_factors']}")

def demo_convenience_functions():
    """Demonstrate convenience functions for triggering retraining"""
    print("\n" + "="*60)
    print("DEMO: Convenience Functions")
    print("="*60)
    
    # Set up global orchestrator (normally done at startup)
    mock_trainer, mock_experiment_tracker, mock_model_registry = create_mock_components()
    
    # This would normally be done during application initialization
    from retraining_orchestrator import _retraining_orchestrator
    global _retraining_orchestrator
    _retraining_orchestrator = RetrainingOrchestrator(
        mock_trainer, mock_experiment_tracker, mock_model_registry
    )
    
    # Test performance-based retraining trigger
    print("\nTriggering performance-based retraining...")
    request_id1 = trigger_performance_retraining(
        current_model_id="chest_xray_model",
        current_model_version="v1.0",
        performance_metrics={"accuracy": 0.72, "f1_score": 0.70},
        threshold_violations=["Accuracy below 80%", "F1 score below 75%"]
    )
    print(f"Performance retraining triggered: {request_id1}")
    
    # Test drift-based retraining trigger
    print("\nTriggering drift-based retraining...")
    drift_report = {
        "overall_drift_score": 0.45,
        "data_drift_score": 0.35,
        "concept_drift_score": 0.55,
        "alerts": [
            {"drift_type": "concept_drift", "severity": "high", "metric_name": "accuracy"}
        ],
        "recommendations": ["Consider model retraining due to concept drift"]
    }
    
    request_id2 = trigger_drift_retraining(
        current_model_id="chest_xray_model",
        current_model_version="v1.0",
        drift_report=drift_report
    )
    print(f"Drift retraining triggered: {request_id2}")

def main():
    """Run all retraining demos"""
    print("CHEST X-RAY PNEUMONIA DETECTION")
    print("Automated Retraining System Demo")
    print("="*60)
    
    try:
        # Run all demos
        demo_model_comparison()
        demo_version_management()
        demo_retraining_orchestrator()
        demo_retraining_integration()
        demo_retraining_scenarios()
        demo_convenience_functions()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Model performance comparison with promotion logic")
        print("✓ Automated model version management and cleanup")
        print("✓ Retraining orchestration with queue management")
        print("✓ Integration with monitoring alerts")
        print("✓ Policy-based retraining decisions")
        print("✓ Multiple retraining trigger scenarios")
        print("✓ Convenience functions for easy integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()