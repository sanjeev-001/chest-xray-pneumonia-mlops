"""
Demo script for Stakeholder Notification System
Shows how notifications work for retraining workflows
"""

import logging
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import notification components
from notification_system import (
    NotificationManager, NotificationRecipient, NotificationChannel,
    NotificationType, NotificationPriority, EmailNotifier, SlackNotifier,
    WebhookNotifier, send_retraining_notification, send_alert_notification
)

# Import retraining components for demo
from retraining_system import RetrainingResult, RetrainingStatus

def demo_notification_recipients():
    """Demonstrate notification recipient configuration"""
    print("\n" + "="*60)
    print("DEMO: Notification Recipients")
    print("="*60)
    
    # Create different types of recipients
    recipients = [
        NotificationRecipient(
            name="ML Engineer",
            email="ml-engineer@company.com",
            slack_user_id="U123456789",
            notification_types=[
                NotificationType.RETRAINING_STARTED,
                NotificationType.RETRAINING_COMPLETED,
                NotificationType.RETRAINING_FAILED,
                NotificationType.MODEL_PROMOTED
            ],
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            priority_threshold=NotificationPriority.MEDIUM
        ),
        NotificationRecipient(
            name="Data Science Manager",
            email="ds-manager@company.com",
            notification_types=[
                NotificationType.MODEL_PROMOTED,
                NotificationType.MODEL_NOT_PROMOTED,
                NotificationType.RETRAINING_FAILED
            ],
            channels=[NotificationChannel.EMAIL],
            priority_threshold=NotificationPriority.HIGH
        ),
        NotificationRecipient(
            name="DevOps Team",
            email="devops@company.com",
            slack_user_id="U987654321",
            notification_types=[
                NotificationType.SYSTEM_ERROR,
                NotificationType.RETRAINING_FAILED,
                NotificationType.PERFORMANCE_ALERT
            ],
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            priority_threshold=NotificationPriority.MEDIUM
        ),
        NotificationRecipient(
            name="Product Owner",
            email="product@company.com",
            notification_types=[
                NotificationType.MODEL_PROMOTED,
                NotificationType.PERFORMANCE_ALERT
            ],
            channels=[NotificationChannel.EMAIL],
            priority_threshold=NotificationPriority.HIGH
        )
    ]
    
    print("Configured recipients:")
    for recipient in recipients:
        print(f"\n  📧 {recipient.name}")
        print(f"     Email: {recipient.email}")
        print(f"     Channels: {[c.value for c in recipient.channels]}")
        print(f"     Priority threshold: {recipient.priority_threshold.value}")
        print(f"     Notification types: {len(recipient.notification_types)} types")
    
    return recipients

def demo_notification_channels():
    """Demonstrate notification channel setup"""
    print("\n" + "="*60)
    print("DEMO: Notification Channels")
    print("="*60)
    
    # Create mock notifiers (in production, use real credentials)
    print("Setting up notification channels...")
    
    # Email notifier
    email_notifier = EmailNotifier(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="mlops-system@company.com",
        password="app_password",  # Use app password in production
        from_email="MLOps System <mlops-system@company.com>"
    )
    print("✓ Email notifier configured")
    
    # Slack notifier
    slack_notifier = SlackNotifier(
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
bot_token="xoxb-your-bot-token",
        bot_token="xoxb-your-bot-token"
    )
    print("✓ Slack notifier configured")
    
    # Webhook notifier
    webhook_notifier = WebhookNotifier(
        webhook_urls=[
            "https://api.company.com/webhooks/mlops",
            "https://monitoring.company.com/alerts"
        ],
        headers={
            "Authorization": "Bearer your-api-token",
            "Content-Type": "application/json"
        }
    )
    print("✓ Webhook notifier configured")
    
    return {
        NotificationChannel.EMAIL: email_notifier,
        NotificationChannel.SLACK: slack_notifier,
        NotificationChannel.WEBHOOK: webhook_notifier
    }

def demo_notification_manager():
    """Demonstrate notification manager functionality"""
    print("\n" + "="*60)
    print("DEMO: Notification Manager")
    print("="*60)
    
    # Create notification manager
    manager = NotificationManager()
    
    # Add recipients
    recipients = demo_notification_recipients()
    for recipient in recipients:
        manager.add_recipient(recipient)
    
    print(f"\n✓ Added {len(recipients)} recipients to manager")
    
    # Add notifiers (mock for demo)
    notifiers = {
        NotificationChannel.EMAIL: Mock(),
        NotificationChannel.SLACK: Mock(),
        NotificationChannel.WEBHOOK: Mock()
    }
    
    # Configure mock responses
    for notifier in notifiers.values():
        notifier.send_notification.return_value = {"recipient1": "sent", "recipient2": "sent"}
    
    for channel, notifier in notifiers.items():
        manager.add_notifier(channel, notifier)
    
    print(f"✓ Added {len(notifiers)} notification channels")
    
    return manager

def demo_retraining_notifications():
    """Demonstrate retraining-specific notifications"""
    print("\n" + "="*60)
    print("DEMO: Retraining Notifications")
    print("="*60)
    
    manager = demo_notification_manager()
    
    # Test Case 1: Successful retraining with promotion
    print("\nTest Case 1: Successful Retraining with Model Promotion")
    
    successful_result = RetrainingResult(
        request_id="retraining_001",
        status=RetrainingStatus.COMPLETED,
        new_model_id="chest_xray_model_v2",
        new_model_version="v2.1.0",
        performance_comparison={
            "overall_improvement": 0.05,  # 5% improvement
            "should_promote": True,
            "promotion_reason": "Overall improvement of 5.0% exceeds threshold"
        },
        promoted=True,
        started_at=datetime.now() - timedelta(hours=2),
        completed_at=datetime.now()
    )
    
    notification_id = manager.send_retraining_notification(successful_result)
    print(f"✓ Promotion notification sent: {notification_id}")
    
    # Test Case 2: Retraining completed but not promoted
    print("\nTest Case 2: Retraining Completed but Not Promoted")
    
    not_promoted_result = RetrainingResult(
        request_id="retraining_002",
        status=RetrainingStatus.COMPLETED,
        new_model_id="chest_xray_model_v2",
        new_model_version="v2.1.1",
        performance_comparison={
            "overall_improvement": 0.01,  # 1% improvement
            "should_promote": False,
            "promotion_reason": "Improvement of 1.0% below threshold of 2.0%"
        },
        promoted=False,
        started_at=datetime.now() - timedelta(hours=1),
        completed_at=datetime.now()
    )
    
    notification_id = manager.send_retraining_notification(not_promoted_result)
    print(f"✓ Non-promotion notification sent: {notification_id}")
    
    # Test Case 3: Failed retraining
    print("\nTest Case 3: Failed Retraining")
    
    failed_result = RetrainingResult(
        request_id="retraining_003",
        status=RetrainingStatus.FAILED,
        new_model_id=None,
        new_model_version=None,
        performance_comparison={},
        promoted=False,
        started_at=datetime.now() - timedelta(minutes=30),
        completed_at=datetime.now(),
        error_message="Training data corruption detected during preprocessing"
    )
    
    notification_id = manager.send_retraining_notification(failed_result)
    print(f"✓ Failure notification sent: {notification_id}")

def demo_alert_notifications():
    """Demonstrate alert notifications"""
    print("\n" + "="*60)
    print("DEMO: Alert Notifications")
    print("="*60)
    
    manager = demo_notification_manager()
    
    # Performance alert
    print("Sending performance alert...")
    perf_notification_id = manager.send_notification(
        notification_type=NotificationType.PERFORMANCE_ALERT,
        subject="🚨 Model Performance Degradation Detected",
        message="""
Critical performance degradation has been detected in the production model.

Current Metrics:
- Accuracy: 72.5% (below 80% threshold)
- F1 Score: 69.8% (below 75% threshold)
- Average Latency: 2.8s (above 2.0s threshold)

Automated retraining has been triggered to address this issue.
        """.strip(),
        data={
            "current_accuracy": 0.725,
            "accuracy_threshold": 0.80,
            "current_f1": 0.698,
            "f1_threshold": 0.75,
            "current_latency_ms": 2800,
            "latency_threshold_ms": 2000,
            "retraining_triggered": True,
            "alert_timestamp": datetime.now().isoformat()
        },
        priority=NotificationPriority.CRITICAL
    )
    print(f"✓ Performance alert sent: {perf_notification_id}")
    
    # Drift alert
    print("\nSending drift alert...")
    drift_notification_id = manager.send_notification(
        notification_type=NotificationType.DRIFT_ALERT,
        subject="⚠️ Data Drift Detected",
        message="""
Significant data drift has been detected in the input data distribution.

Drift Analysis:
- Overall drift score: 0.45 (above 0.3 threshold)
- Data drift score: 0.38
- Concept drift score: 0.52

This may indicate changes in the underlying data patterns that could affect model performance.
        """.strip(),
        data={
            "overall_drift_score": 0.45,
            "data_drift_score": 0.38,
            "concept_drift_score": 0.52,
            "drift_threshold": 0.3,
            "detection_timestamp": datetime.now().isoformat(),
            "recommendations": [
                "Monitor model performance closely",
                "Consider retraining if performance degrades",
                "Investigate data source changes"
            ]
        },
        priority=NotificationPriority.HIGH
    )
    print(f"✓ Drift alert sent: {drift_notification_id}")

def demo_notification_filtering():
    """Demonstrate notification filtering by recipient preferences"""
    print("\n" + "="*60)
    print("DEMO: Notification Filtering")
    print("="*60)
    
    manager = demo_notification_manager()
    
    # Test different priority levels
    priorities = [
        (NotificationPriority.LOW, "Low priority system update"),
        (NotificationPriority.MEDIUM, "Medium priority performance warning"),
        (NotificationPriority.HIGH, "High priority accuracy drop"),
        (NotificationPriority.CRITICAL, "Critical system failure")
    ]
    
    print("Testing notification filtering by priority and preferences...")
    
    for priority, message in priorities:
        notification_id = manager.send_notification(
            notification_type=NotificationType.PERFORMANCE_ALERT,
            subject=f"{priority.value.upper()} Alert",
            message=message,
            priority=priority
        )
        
        print(f"  {priority.value.upper()}: {notification_id[:8]}... - {message}")
    
    # Test recipient-specific filtering
    print("\nTesting recipient-specific notification types...")
    
    # This should only go to DevOps team (they subscribe to system errors)
    system_error_id = manager.send_notification(
        notification_type=NotificationType.SYSTEM_ERROR,
        subject="System Error: Database Connection Lost",
        message="Critical system error requires immediate attention",
        priority=NotificationPriority.CRITICAL
    )
    print(f"  System Error (DevOps only): {system_error_id[:8]}...")
    
    # This should go to ML Engineer and Data Science Manager
    model_promotion_id = manager.send_notification(
        notification_type=NotificationType.MODEL_PROMOTED,
        subject="New Model Promoted to Production",
        message="Model v2.1.0 has been successfully promoted",
        priority=NotificationPriority.HIGH
    )
    print(f"  Model Promotion (ML + Manager): {model_promotion_id[:8]}...")

def demo_notification_history():
    """Demonstrate notification history and statistics"""
    print("\n" + "="*60)
    print("DEMO: Notification History & Statistics")
    print("="*60)
    
    manager = demo_notification_manager()
    
    # Send several notifications to build history
    notifications = [
        (NotificationType.RETRAINING_STARTED, NotificationPriority.LOW),
        (NotificationType.PERFORMANCE_ALERT, NotificationPriority.HIGH),
        (NotificationType.MODEL_PROMOTED, NotificationPriority.HIGH),
        (NotificationType.DRIFT_ALERT, NotificationPriority.MEDIUM),
        (NotificationType.RETRAINING_FAILED, NotificationPriority.CRITICAL)
    ]
    
    print("Sending sample notifications...")
    for notification_type, priority in notifications:
        manager.send_notification(
            notification_type=notification_type,
            subject=f"Demo {notification_type.value}",
            message=f"This is a demo {notification_type.value} notification",
            priority=priority
        )
    
    # Get notification history
    print(f"\n✓ Sent {len(notifications)} demo notifications")
    
    history = manager.get_notification_history(limit=10)
    print(f"\nRecent notification history ({len(history)} notifications):")
    
    for i, notification in enumerate(history[:5], 1):
        print(f"  {i}. {notification['type']} ({notification['priority']}) - {notification['subject']}")
        print(f"     Sent: {notification['created_at']}")
        print(f"     Recipients: {len(notification['recipients'])}")
    
    # Get statistics
    stats = manager.get_notification_stats()
    print(f"\nNotification Statistics:")
    print(f"  Total notifications: {stats['total_notifications']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Total recipients: {stats['total_recipients']}")
    
    print(f"\n  By Type:")
    for notification_type, count in stats['by_type'].items():
        print(f"    {notification_type}: {count}")
    
    print(f"\n  By Priority:")
    for priority, count in stats['by_priority'].items():
        print(f"    {priority}: {count}")

def demo_convenience_functions():
    """Demonstrate convenience functions"""
    print("\n" + "="*60)
    print("DEMO: Convenience Functions")
    print("="*60)
    
    # Set up global manager
    manager = demo_notification_manager()
    
    # Test retraining notification convenience function
    print("Testing retraining notification convenience function...")
    
    sample_result = RetrainingResult(
        request_id="convenience_test",
        status=RetrainingStatus.COMPLETED,
        new_model_id="test_model",
        new_model_version="v1.0",
        performance_comparison={"overall_improvement": 0.03, "should_promote": True},
        promoted=True,
        started_at=datetime.now() - timedelta(hours=1),
        completed_at=datetime.now()
    )
    
    notification_id = send_retraining_notification(sample_result)
    print(f"✓ Retraining notification sent: {notification_id}")
    
    # Test alert notification convenience function
    print("\nTesting alert notification convenience function...")
    
    alert_id = send_alert_notification(
        alert_type="Performance Degradation",
        message="Model accuracy has dropped below acceptable threshold",
        data={
            "current_accuracy": 0.75,
            "threshold": 0.80,
            "drop_percentage": 0.05
        },
        priority=NotificationPriority.HIGH
    )
    print(f"✓ Alert notification sent: {alert_id}")

def main():
    """Run all notification demos"""
    print("CHEST X-RAY PNEUMONIA DETECTION")
    print("Stakeholder Notification System Demo")
    print("="*60)
    
    try:
        # Run all demos
        demo_notification_recipients()
        demo_notification_channels()
        demo_notification_manager()
        demo_retraining_notifications()
        demo_alert_notifications()
        demo_notification_filtering()
        demo_notification_history()
        demo_convenience_functions()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Multi-channel notifications (Email, Slack, Webhook)")
        print("✓ Recipient-specific filtering and preferences")
        print("✓ Priority-based notification routing")
        print("✓ Retraining workflow notifications")
        print("✓ Performance and drift alert notifications")
        print("✓ Notification history and statistics")
        print("✓ Convenience functions for easy integration")
        print("\nProduction Setup Notes:")
        print("• Configure real SMTP settings for email notifications")
        print("• Set up Slack webhook URLs and bot tokens")
        print("• Configure webhook endpoints for external integrations")
        print("• Add recipient contact information")
        print("• Test notification delivery in staging environment")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()