"""
Comprehensive Tests for Retraining Workflows
Tests retraining trigger conditions, model comparison, and notifications
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import threading

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.retraining_system import (
    RetrainingOrchestrator, RetrainingTrigger, RetrainingStatus,
    ModelComparator, RetrainingPolicy, RetrainingRequest, RetrainingResult,
    trigger_performance_retraining, trigger_drift_retraining
)
from training.notification_system import (
    NotificationManager, NotificationRecipient, NotificationChannel,
    NotificationType, NotificationPriority, send_retraining_notification
)

class TestRetrainingTriggerConditions:
    """Test retraining trigger conditions and logic"""
    
    def test_performance_degradation_trigger(self):
        """Test triggering retraining due to performance degradation"""
        orchestrator = RetrainingOrchestrator(improvement_threshold=0.02)
        
        # Test performance degradation trigger
        request_id = orchestrator.trigger_retraining(
            trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            current_model_id="test_model",
            current_model_version="v1.0",
            trigger_data={
                "performance_metrics": {"accuracy": 0.72, "f1_score": 0.68},
                "threshold_violations": ["Accuracy: 0.72 < 0.80"]
            },
            priority=1
        )
        
        assert request_id is not None
        assert len(request_id) > 0
        
        # Check status
        status = orchestrator.get_retraining_status(request_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["promoted"] is True  # Mock should promote
    
    def test_drift_detection_trigger(self):
        """Test triggering retraining due to drift detection"""
        orchestrator = RetrainingOrchestrator()
        
        request_id = orchestrator.trigger_retraining(
            trigger_type=RetrainingTrigger.DATA_DRIFT,
            current_model_id="test_model",
            current_model_version="v1.0",
            trigger_data={
                "drift_report": {
                    "overall_drift_score": 0.45,
                    "data_drift_score": 0.38,
                    "concept_drift_score": 0.52
                }
            },
            priority=2
        )
        
        assert request_id is not None
        status = orchestrator.get_retraining_status(request_id)
        assert status["status"] == "completed"    
    
    def test_manual_trigger(self):
        """Test manual retraining trigger"""
        orchestrator = RetrainingOrchestrator()
        
        request_id = orchestrator.trigger_retraining(
            trigger_type=RetrainingTrigger.MANUAL,
            current_model_id="test_model",
            current_model_version="v1.0",
            trigger_data={"reason": "Manual testing", "user": "test_user"},
            priority=3
        )
        
        assert request_id is not None
        status = orchestrator.get_retraining_status(request_id)
        assert status is not None
    
    def test_scheduled_trigger(self):
        """Test scheduled retraining trigger"""
        orchestrator = RetrainingOrchestrator()
        
        request_id = orchestrator.trigger_retraining(
            trigger_type=RetrainingTrigger.SCHEDULED,
            current_model_id="test_model",
            current_model_version="v1.0",
            trigger_data={
                "schedule_info": {"frequency": "weekly", "day": "sunday"}
            },
            priority=3
        )
        
        assert request_id is not None
        status = orchestrator.get_retraining_status(request_id)
        assert status["status"] == "completed"
    
    def test_convenience_functions(self):
        """Test convenience functions for triggering retraining"""
        
        # Test performance retraining convenience function
        request_id1 = trigger_performance_retraining(
            current_model_id="test_model",
            current_model_version="v1.0",
            performance_metrics={"accuracy": 0.70, "f1_score": 0.65},
            threshold_violations=["Accuracy below threshold"]
        )
        assert request_id1 is not None
        
        # Test drift retraining convenience function
        request_id2 = trigger_drift_retraining(
            current_model_id="test_model",
            current_model_version="v1.0",
            drift_report={
                "overall_drift_score": 0.5,
                "recommendations": ["Consider retraining"]
            }
        )
        assert request_id2 is not None

class TestModelComparison:
    """Test model comparison and promotion logic"""
    
    def test_significant_improvement_promotion(self):
        """Test model promotion with significant improvement"""
        comparator = ModelComparator(improvement_threshold=0.02)
        
        current_metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "auc_roc": 0.87,
            "precision": 0.83,
            "recall": 0.81
        }
        
        new_metrics = {
            "accuracy": 0.92,  # 7% improvement
            "f1_score": 0.89,  # 7% improvement
            "auc_roc": 0.94,   # 7% improvement
            "precision": 0.90,
            "recall": 0.88
        }
        
        comparison = comparator.compare_models(current_metrics, new_metrics)
        
        assert comparison["should_promote"] is True
        assert comparison["overall_improvement"] > 0.02
        assert "exceeds threshold" in comparison["promotion_reason"]
        assert len(comparison["improvements"]) > 0
    
    def test_minor_improvement_no_promotion(self):
        """Test no promotion with minor improvement"""
        comparator = ModelComparator(improvement_threshold=0.02)
        
        current_metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "auc_roc": 0.87
        }
        
        new_metrics = {
            "accuracy": 0.86,  # 1% improvement
            "f1_score": 0.83,  # 1% improvement
            "auc_roc": 0.88    # 1% improvement
        }
        
        comparison = comparator.compare_models(current_metrics, new_metrics)
        
        assert comparison["should_promote"] is False
        assert comparison["overall_improvement"] < 0.02
        assert "below threshold" in comparison["promotion_reason"]
    
    def test_performance_degradation_no_promotion(self):
        """Test no promotion with performance degradation"""
        comparator = ModelComparator(improvement_threshold=0.02)
        
        current_metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82
        }
        
        new_metrics = {
            "accuracy": 0.78,  # 7% degradation
            "f1_score": 0.75   # 7% degradation
        }
        
        comparison = comparator.compare_models(current_metrics, new_metrics)
        
        assert comparison["should_promote"] is False
        assert comparison["overall_improvement"] < 0
    
    def test_weighted_metric_calculation(self):
        """Test weighted metric calculation in comparison"""
        comparator = ModelComparator(improvement_threshold=0.02)
        
        # Test that accuracy has higher weight than precision/recall
        current_metrics = {
            "accuracy": 0.80,
            "f1_score": 0.75,
            "auc_roc": 0.82,
            "precision": 0.90,  # High precision
            "recall": 0.90      # High recall
        }
        
        new_metrics = {
            "accuracy": 0.85,  # 5% improvement (high weight)
            "f1_score": 0.80,  # 5% improvement (medium weight)
            "auc_roc": 0.87,   # 5% improvement (medium weight)
            "precision": 0.70, # 20% degradation (low weight)
            "recall": 0.70     # 20% degradation (low weight)
        }
        
        comparison = comparator.compare_models(current_metrics, new_metrics)
        
        # Should still promote due to high-weight metric improvements
        assert comparison["should_promote"] is True
        assert comparison["overall_improvement"] > 0.02

class TestRetrainingPolicy:
    """Test retraining policy decisions"""
    
    def test_policy_critical_alerts_trigger(self):
        """Test policy triggers retraining for critical alerts"""
        policy = RetrainingPolicy(
            min_alerts_for_retraining=3,
            max_retraining_per_day=2
        )
        
        decision = policy.should_trigger_retraining(
            alert_count=5,
            critical_alerts=2,
            retraining_count_today=0
        )
        
        assert decision["should_trigger"] is True
        assert decision["priority"] == 1  # High priority
        assert "Critical performance issues" in decision["reason"]
    
    def test_policy_multiple_alerts_trigger(self):
        """Test policy triggers for multiple alerts"""
        policy = RetrainingPolicy(
            min_alerts_for_retraining=3,
            max_retraining_per_day=2
        )
        
        decision = policy.should_trigger_retraining(
            alert_count=8,  # 2x minimum
            critical_alerts=0,
            retraining_count_today=1
        )
        
        assert decision["should_trigger"] is True
        assert decision["priority"] == 2  # Medium priority
        assert "Multiple performance issues" in decision["reason"]
    
    def test_policy_daily_limit_blocks(self):
        """Test policy blocks retraining when daily limit reached"""
        policy = RetrainingPolicy(
            min_alerts_for_retraining=3,
            max_retraining_per_day=2
        )
        
        decision = policy.should_trigger_retraining(
            alert_count=10,
            critical_alerts=5,
            retraining_count_today=2  # At limit
        )
        
        assert decision["should_trigger"] is False
        assert "Daily limit reached" in decision["reason"]
    
    def test_policy_insufficient_alerts(self):
        """Test policy blocks retraining with insufficient alerts"""
        policy = RetrainingPolicy(
            min_alerts_for_retraining=5,
            max_retraining_per_day=2
        )
        
        decision = policy.should_trigger_retraining(
            alert_count=3,  # Below minimum
            critical_alerts=1,
            retraining_count_today=0
        )
        
        assert decision["should_trigger"] is False
        assert "Insufficient alerts" in decision["reason"]

class TestNotificationSystem:
    """Test notification system for retraining workflows"""
    
    def test_notification_manager_initialization(self):
        """Test notification manager initialization"""
        manager = NotificationManager()
        
        # Add test recipient
        recipient = NotificationRecipient(
            name="Test User",
            email="test@example.com",
            notification_types=[NotificationType.MODEL_PROMOTED],
            channels=[NotificationChannel.EMAIL]
        )
        
        manager.add_recipient(recipient)
        assert "Test User" in manager.recipients
    
    def test_retraining_success_notification(self):
        """Test notification for successful retraining"""
        manager = NotificationManager()
        
        # Add mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = {"test@example.com": "sent"}
        manager.add_notifier(NotificationChannel.EMAIL, mock_notifier)
        
        # Add recipient
        recipient = NotificationRecipient(
            name="ML Engineer",
            email="ml@example.com",
            notification_types=[NotificationType.MODEL_PROMOTED],
            channels=[NotificationChannel.EMAIL],
            priority_threshold=NotificationPriority.MEDIUM
        )
        manager.add_recipient(recipient)
        
        # Create successful retraining result
        result = RetrainingResult(
            request_id="test_001",
            status=RetrainingStatus.COMPLETED,
            new_model_id="test_model_v2",
            new_model_version="v2.0",
            performance_comparison={"should_promote": True, "overall_improvement": 0.05},
            promoted=True,
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now()
        )
        
        notification_id = manager.send_retraining_notification(result)
        
        assert notification_id is not None
        assert mock_notifier.send_notification.called
    
    def test_retraining_failure_notification(self):
        """Test notification for failed retraining"""
        manager = NotificationManager()
        
        # Add mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = {"devops@example.com": "sent"}
        manager.add_notifier(NotificationChannel.EMAIL, mock_notifier)
        
        # Add recipient interested in failures
        recipient = NotificationRecipient(
            name="DevOps",
            email="devops@example.com",
            notification_types=[NotificationType.RETRAINING_FAILED],
            channels=[NotificationChannel.EMAIL],
            priority_threshold=NotificationPriority.LOW
        )
        manager.add_recipient(recipient)
        
        # Create failed retraining result
        result = RetrainingResult(
            request_id="test_002",
            status=RetrainingStatus.FAILED,
            new_model_id=None,
            new_model_version=None,
            performance_comparison={},
            promoted=False,
            started_at=datetime.now() - timedelta(minutes=30),
            completed_at=datetime.now(),
            error_message="Training data corruption detected"
        )
        
        notification_id = manager.send_retraining_notification(result)
        
        assert notification_id is not None
        assert mock_notifier.send_notification.called
    
    def test_notification_filtering_by_priority(self):
        """Test notification filtering by priority threshold"""
        manager = NotificationManager()
        
        # Add mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = {}
        manager.add_notifier(NotificationChannel.EMAIL, mock_notifier)
        
        # Add recipient with HIGH priority threshold
        recipient = NotificationRecipient(
            name="Manager",
            email="manager@example.com",
            notification_types=[NotificationType.PERFORMANCE_ALERT],
            channels=[NotificationChannel.EMAIL],
            priority_threshold=NotificationPriority.HIGH
        )
        manager.add_recipient(recipient)
        
        # Send LOW priority notification (should be filtered out)
        notification_id = manager.send_notification(
            notification_type=NotificationType.PERFORMANCE_ALERT,
            subject="Low Priority Alert",
            message="Minor performance issue",
            priority=NotificationPriority.LOW
        )
        
        # Should not call notifier due to priority filtering
        assert not mock_notifier.send_notification.called
        
        # Send HIGH priority notification (should go through)
        notification_id = manager.send_notification(
            notification_type=NotificationType.PERFORMANCE_ALERT,
            subject="High Priority Alert",
            message="Critical performance issue",
            priority=NotificationPriority.HIGH
        )
        
        # Should call notifier for high priority
        assert mock_notifier.send_notification.called
    
    def test_notification_filtering_by_type(self):
        """Test notification filtering by notification type"""
        manager = NotificationManager()
        
        # Add mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = {}
        manager.add_notifier(NotificationChannel.EMAIL, mock_notifier)
        
        # Add recipient only interested in model promotions
        recipient = NotificationRecipient(
            name="Product Owner",
            email="po@example.com",
            notification_types=[NotificationType.MODEL_PROMOTED],
            channels=[NotificationChannel.EMAIL]
        )
        manager.add_recipient(recipient)
        
        # Send system error notification (should be filtered out)
        manager.send_notification(
            notification_type=NotificationType.SYSTEM_ERROR,
            subject="System Error",
            message="Database connection lost",
            priority=NotificationPriority.CRITICAL
        )
        
        # Should not call notifier due to type filtering
        assert not mock_notifier.send_notification.called
        
        # Send model promotion notification (should go through)
        manager.send_notification(
            notification_type=NotificationType.MODEL_PROMOTED,
            subject="Model Promoted",
            message="New model deployed",
            priority=NotificationPriority.HIGH
        )
        
        # Should call notifier for matching type
        assert mock_notifier.send_notification.called
    
    def test_notification_history_tracking(self):
        """Test notification history and statistics"""
        manager = NotificationManager()
        
        # Add mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = {"test@example.com": "sent"}
        manager.add_notifier(NotificationChannel.EMAIL, mock_notifier)
        
        # Add recipient
        recipient = NotificationRecipient(
            name="Test User",
            email="test@example.com",
            notification_types=list(NotificationType),
            channels=[NotificationChannel.EMAIL]
        )
        manager.add_recipient(recipient)
        
        # Send multiple notifications
        for i in range(5):
            manager.send_notification(
                notification_type=NotificationType.PERFORMANCE_ALERT,
                subject=f"Alert {i}",
                message=f"Test alert {i}",
                priority=NotificationPriority.MEDIUM
            )
        
        # Check history
        history = manager.get_notification_history(limit=10)
        assert len(history) == 5
        
        # Check statistics
        stats = manager.get_notification_stats()
        assert stats["total_notifications"] == 5
        assert stats["by_type"]["performance_alert"] == 5
        assert stats["success_rate"] == 1.0  # All successful

class TestRetrainingWorkflowIntegration:
    """Test complete retraining workflow integration"""
    
    def test_end_to_end_retraining_workflow(self):
        """Test complete retraining workflow from trigger to notification"""
        
        # Initialize components
        orchestrator = RetrainingOrchestrator(improvement_threshold=0.02)
        notification_manager = NotificationManager()
        
        # Add mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = {"ml@example.com": "sent"}
        notification_manager.add_notifier(NotificationChannel.EMAIL, mock_notifier)
        
        # Add recipient
        recipient = NotificationRecipient(
            name="ML Engineer",
            email="ml@example.com",
            notification_types=[
                NotificationType.RETRAINING_STARTED,
                NotificationType.MODEL_PROMOTED,
                NotificationType.RETRAINING_FAILED
            ],
            channels=[NotificationChannel.EMAIL]
        )
        notification_manager.add_recipient(recipient)
        
        # Trigger retraining
        request_id = orchestrator.trigger_retraining(
            trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            current_model_id="integration_test_model",
            current_model_version="v1.0",
            trigger_data={
                "performance_metrics": {"accuracy": 0.70},
                "threshold_violations": ["Accuracy below 80%"]
            },
            priority=1
        )
        
        # Get retraining result
        status = orchestrator.get_retraining_status(request_id)
        assert status is not None
        assert status["status"] == "completed"
        
        # Create result object for notification
        result = RetrainingResult(
            request_id=request_id,
            status=RetrainingStatus.COMPLETED,
            new_model_id="integration_test_model_v2",
            new_model_version="v2.0",
            performance_comparison={"should_promote": True, "overall_improvement": 0.05},
            promoted=True,
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now()
        )
        
        # Send notification
        notification_id = notification_manager.send_retraining_notification(result)
        
        assert notification_id is not None
        assert mock_notifier.send_notification.called
        
        # Verify notification was sent with correct data
        call_args = mock_notifier.send_notification.call_args
        notification_message = call_args[0][0]  # First argument
        
        assert notification_message.notification_type == NotificationType.MODEL_PROMOTED
        assert notification_message.priority == NotificationPriority.HIGH
        assert request_id in notification_message.data["request_id"]
    
    def test_concurrent_retraining_requests(self):
        """Test handling of concurrent retraining requests"""
        orchestrator = RetrainingOrchestrator()
        
        # Submit multiple requests quickly
        request_ids = []
        for i in range(3):
            request_id = orchestrator.trigger_retraining(
                trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                current_model_id=f"concurrent_test_model_{i}",
                current_model_version="v1.0",
                trigger_data={"test": f"concurrent_{i}"},
                priority=1
            )
            request_ids.append(request_id)
        
        # All requests should be created
        assert len(request_ids) == 3
        assert len(set(request_ids)) == 3  # All unique
        
        # Check that all requests were processed
        for request_id in request_ids:
            status = orchestrator.get_retraining_status(request_id)
            assert status is not None
    
    def test_retraining_history_tracking(self):
        """Test retraining history tracking"""
        orchestrator = RetrainingOrchestrator()
        
        # Submit multiple retraining requests
        request_ids = []
        for i in range(5):
            request_id = orchestrator.trigger_retraining(
                trigger_type=RetrainingTrigger.MANUAL,
                current_model_id="history_test_model",
                current_model_version=f"v1.{i}",
                trigger_data={"test": f"history_{i}"},
                priority=2
            )
            request_ids.append(request_id)
        
        # Get history
        history = orchestrator.list_retraining_history(limit=10)
        
        assert len(history) == 5
        
        # Verify history contains our requests
        history_request_ids = [h["request_id"] for h in history]
        for request_id in request_ids:
            assert request_id in history_request_ids
        
        # Verify history is sorted by completion time (newest first)
        timestamps = [h["completed_at"] for h in history if h["completed_at"]]
        assert timestamps == sorted(timestamps, reverse=True)

class TestErrorHandling:
    """Test error handling in retraining workflows"""
    
    def test_invalid_trigger_data(self):
        """Test handling of invalid trigger data"""
        orchestrator = RetrainingOrchestrator()
        
        # Should handle missing or invalid trigger data gracefully
        request_id = orchestrator.trigger_retraining(
            trigger_type=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            current_model_id="error_test_model",
            current_model_version="v1.0",
            trigger_data={},  # Empty trigger data
            priority=1
        )
        
        assert request_id is not None
        status = orchestrator.get_retraining_status(request_id)
        assert status is not None
    
    def test_notification_delivery_failure(self):
        """Test handling of notification delivery failures"""
        manager = NotificationManager()
        
        # Add failing notifier
        failing_notifier = Mock()
        failing_notifier.send_notification.side_effect = Exception("Network error")
        manager.add_notifier(NotificationChannel.EMAIL, failing_notifier)
        
        # Add recipient
        recipient = NotificationRecipient(
            name="Test User",
            email="test@example.com",
            notification_types=[NotificationType.SYSTEM_ERROR],
            channels=[NotificationChannel.EMAIL]
        )
        manager.add_recipient(recipient)
        
        # Send notification (should handle failure gracefully)
        notification_id = manager.send_notification(
            notification_type=NotificationType.SYSTEM_ERROR,
            subject="Test Error",
            message="Test error message",
            priority=NotificationPriority.HIGH
        )
        
        # Should still return notification ID even if delivery fails
        assert notification_id is not None
        
        # Check that failure was recorded
        history = manager.get_notification_history(limit=1)
        assert len(history) == 1
        assert "error" in str(history[0]["delivery_status"]).lower()
    
    def test_model_comparison_edge_cases(self):
        """Test model comparison with edge cases"""
        comparator = ModelComparator(improvement_threshold=0.02)
        
        # Test with missing metrics
        current_metrics = {"accuracy": 0.85}
        new_metrics = {"f1_score": 0.90}  # Different metrics
        
        comparison = comparator.compare_models(current_metrics, new_metrics)
        
        # Should handle gracefully
        assert "should_promote" in comparison
        assert "overall_improvement" in comparison
        
        # Test with zero values
        current_metrics = {"accuracy": 0.0}
        new_metrics = {"accuracy": 0.85}
        
        comparison = comparator.compare_models(current_metrics, new_metrics)
        
        # Should handle division by zero - when current is 0, improvement_pct is 0
        # So overall improvement will be 0, which means no promotion
        assert comparison["should_promote"] is False  # Division by zero results in 0 improvement

if __name__ == "__main__":
    pytest.main([__file__, "-v"])