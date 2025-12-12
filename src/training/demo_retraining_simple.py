"""
Simple Demo for Automated Retraining System
Shows how the retraining orchestrator works
"""

import logging
import time
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import retraining components
from retraining_system import (
    RetrainingOrchestrator, RetrainingTrigger, ModelComparator,
    RetrainingPolicy, trigger_performance_retraining, trigger_drift_retraining
)

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
    print(f"✓ Should promote: {comparison['should_promote']}")
    print(f"✓ Overall improvement: {comparison['overall_improvement']:.1%}")
    print(f"✓ Reason: {comparison['promotion_reason']}")
    
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
    print(f"✓ Should promote: {comparison['should_promote']}")
    print(f"✓ Overall improvement: {comparison['overall_improvement']:.1%}")
    print(f"✓ Reason: {comparison['promotion_reason']}")

def demo_retraining_orchestrator():
    """Demonstrate retraining orchestrator functionality"""
    print("\n" + "="*60)
    print("DEMO: Retraining Orchestrator")
    print("="*60)
    
    # Create orchestrator
    orchestrator = RetrainingOrchestrator(improvement_threshold=0.02)
    
    print("✓ Retraining orchestrator created")
    
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
        priority=1
    )
    
    print(f"✓ Retraining request created: {request_id}")
    
    # Check status
    print("\nChecking retraining status...")
    status = orchestrator.get_retraining_status(request_id)
    if status:
        print(f"✓ Status: {status['status']}")
        print(f"✓ Promoted: {status.get('promoted', 'N/A')}")
        if 'performance_comparison' in status:
            comparison = status['performance_comparison']
            print(f"✓ Overall improvement: {comparison.get('overall_improvement', 0):.1%}")
    
    # Show retraining history
    print("\nRetraining history:")
    history = orchestrator.list_retraining_history(limit=5)
    for i, entry in enumerate(history, 1):
        print(f"  {i}. {entry['request_id'][:8]}... ({entry['status']}) - Promoted: {entry['promoted']}")

def demo_retraining_policy():
    """Demonstrate retraining policy decisions"""
    print("\n" + "="*60)
    print("DEMO: Retraining Policy")
    print("="*60)
    
    policy = RetrainingPolicy(
        accuracy_threshold=0.80,
        min_alerts_for_retraining=3,
        alert_window_hours=2,
        max_retraining_per_day=2
    )
    
    print("Policy configuration:")
    print(f"  - Accuracy threshold: {policy.accuracy_threshold}")
    print(f"  - Min alerts for retraining: {policy.min_alerts_for_retraining}")
    print(f"  - Alert window: {policy.alert_window_hours} hours")
    print(f"  - Max retraining per day: {policy.max_retraining_per_day}")
    
    # Test scenarios
    scenarios = [
        {
            "name": "Critical Performance Issues",
            "alert_count": 5,
            "critical_alerts": 2,
            "retraining_count_today": 0
        },
        {
            "name": "Multiple Alerts",
            "alert_count": 8,
            "critical_alerts": 0,
            "retraining_count_today": 1
        },
        {
            "name": "Daily Limit Reached",
            "alert_count": 10,
            "critical_alerts": 3,
            "retraining_count_today": 2
        },
        {
            "name": "Insufficient Alerts",
            "alert_count": 2,
            "critical_alerts": 0,
            "retraining_count_today": 0
        }
    ]
    
    print("\nTesting policy scenarios:")
    for scenario in scenarios:
        decision = policy.should_trigger_retraining(
            alert_count=scenario["alert_count"],
            critical_alerts=scenario["critical_alerts"],
            retraining_count_today=scenario["retraining_count_today"]
        )
        
        print(f"\n  Scenario: {scenario['name']}")
        print(f"    Alerts: {scenario['alert_count']} (critical: {scenario['critical_alerts']})")
        print(f"    Should trigger: {decision['should_trigger']}")
        print(f"    Priority: {decision['priority']}")
        print(f"    Reason: {decision['reason']}")

def demo_convenience_functions():
    """Demonstrate convenience functions for triggering retraining"""
    print("\n" + "="*60)
    print("DEMO: Convenience Functions")
    print("="*60)
    
    # Test performance-based retraining trigger
    print("Triggering performance-based retraining...")
    request_id1 = trigger_performance_retraining(
        current_model_id="chest_xray_model",
        current_model_version="v1.0",
        performance_metrics={"accuracy": 0.72, "f1_score": 0.70},
        threshold_violations=["Accuracy below 80%", "F1 score below 75%"]
    )
    print(f"✓ Performance retraining triggered: {request_id1}")
    
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
    print(f"✓ Drift retraining triggered: {request_id2}")

def demo_integration_scenario():
    """Demonstrate a complete integration scenario"""
    print("\n" + "="*60)
    print("DEMO: Complete Integration Scenario")
    print("="*60)
    
    # Simulate a production scenario
    print("Simulating production monitoring scenario...")
    
    # 1. Performance monitoring detects issues
    print("\n1. Performance monitoring detects accuracy drop")
    performance_metrics = {
        "accuracy": 0.72,  # Below 80% threshold
        "f1_score": 0.69,  # Below 75% threshold
        "latency_ms": 2500  # Above 2000ms threshold
    }
    
    threshold_violations = [
        "Accuracy: 0.72 < 0.80",
        "F1 Score: 0.69 < 0.75",
        "Latency: 2500ms > 2000ms"
    ]
    
    print(f"   Current metrics: {performance_metrics}")
    print(f"   Violations: {len(threshold_violations)}")
    
    # 2. Policy evaluation
    print("\n2. Evaluating retraining policy")
    policy = RetrainingPolicy()
    decision = policy.should_trigger_retraining(
        alert_count=5,  # Multiple alerts
        critical_alerts=2,  # Critical accuracy issues
        retraining_count_today=0  # No retraining today yet
    )
    
    print(f"   Should trigger: {decision['should_trigger']}")
    print(f"   Priority: {decision['priority']}")
    print(f"   Reason: {decision['reason']}")
    
    # 3. Trigger retraining if policy allows
    if decision["should_trigger"]:
        print("\n3. Triggering automated retraining")
        request_id = trigger_performance_retraining(
            current_model_id="chest_xray_pneumonia",
            current_model_version="v1.2.0",
            performance_metrics=performance_metrics,
            threshold_violations=threshold_violations
        )
        
        print(f"   Retraining request: {request_id}")
        
        # 4. Monitor retraining progress (simulated)
        print("\n4. Monitoring retraining progress")
        orchestrator = RetrainingOrchestrator()
        status = orchestrator.get_retraining_status(request_id)
        
        if status:
            print(f"   Status: {status['status']}")
            print(f"   Model promoted: {status.get('promoted', False)}")
            
            if status.get('performance_comparison'):
                comparison = status['performance_comparison']
                print(f"   Performance improvement: {comparison.get('overall_improvement', 0):.1%}")
        
        print("\n✓ Integration scenario completed successfully!")
    else:
        print("\n3. Retraining not triggered due to policy constraints")

def main():
    """Run all retraining demos"""
    print("CHEST X-RAY PNEUMONIA DETECTION")
    print("Automated Retraining System Demo")
    print("="*60)
    
    try:
        # Run all demos
        demo_model_comparison()
        demo_retraining_orchestrator()
        demo_retraining_policy()
        demo_convenience_functions()
        demo_integration_scenario()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Model performance comparison with promotion logic")
        print("✓ Automated retraining orchestration")
        print("✓ Policy-based retraining decisions")
        print("✓ Convenience functions for easy integration")
        print("✓ Complete monitoring-to-retraining workflow")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()