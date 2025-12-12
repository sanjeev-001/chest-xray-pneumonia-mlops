"""
Performance Monitoring for Chest X-Ray Pneumonia Detection System
Monitors model performance metrics and triggers alerts
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path
import threading
import time

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from .metrics_collector import MetricsDatabase, get_metrics_collector
    from .drift_detector import DriftDetector, DriftSeverity
except ImportError:
    from metrics_collector import MetricsDatabase, get_metrics_collector
    from drift_detector import DriftDetector, DriftSeverity

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Types of performance alerts"""
    ACCURACY_DROP = "accuracy_drop"
    HIGH_LATENCY = "high_latency"
    LOW_THROUGHPUT = "low_throughput"
    HIGH_ERROR_RATE = "high_error_rate"
    DRIFT_DETECTED = "drift_detected"
    SYSTEM_RESOURCE = "system_resource"
    MODEL_FAILURE = "model_failure"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    message: str
    metadata: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    accuracy: Optional[float]
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_mb: Optional[float]
    active_requests: int
    cache_hit_rate: float
    model_version: str

class NotificationChannel:
    """Base class for notification channels"""
    
    def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert notification"""
        raise NotImplementedError

class EmailNotifier(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_email: str, to_emails: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert via email"""
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available")
            return False
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] Chest X-Ray Model Alert: {alert.alert_type.value}"
            
            body = f"""
Performance Alert Detected

Alert Type: {alert.alert_type.value}
Severity: {alert.severity.value}
Timestamp: {alert.timestamp}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}

Message: {alert.message}

Metadata: {json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}

Please investigate and take appropriate action.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

class SlackNotifier(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert via Slack webhook"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, cannot send Slack notification")
            return False
        
        try:
            # Color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500", 
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#8B0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#ff0000"),
                    "title": f"Chest X-Ray Model Alert: {alert.alert_type.value}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.alert_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

class WebhookNotifier(NotificationChannel):
    """Generic webhook notification channel"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    def send_alert(self, alert: PerformanceAlert) -> bool:
        """Send alert via webhook"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, cannot send webhook notification")
            return False
        
        try:
            payload = {
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "message": alert.message,
                "metadata": alert.metadata
            }
            
            response = requests.post(self.webhook_url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.alert_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

class PerformanceMonitor:
    """
    Main performance monitoring service
    """
    
    def __init__(self, database: MetricsDatabase = None, drift_detector: DriftDetector = None):
        self.database = database or MetricsDatabase()
        self.drift_detector = drift_detector or DriftDetector(self.database)
        
        # Performance thresholds (from requirements)
        self.accuracy_threshold = 0.8  # 80% minimum accuracy
        self.latency_threshold_ms = 2000  # 2 seconds maximum response time
        self.error_rate_threshold = 0.05  # 5% maximum error rate
        self.cpu_threshold = 90.0  # 90% CPU usage
        self.memory_threshold = 90.0  # 90% memory usage
        self.throughput_threshold_rps = 1.0  # Minimum 1 request per second
        
        # Alert management
        self.active_alerts = {}  # alert_key -> PerformanceAlert
        self.alert_history = []
        self.notification_channels = []
        
        # Monitoring configuration
        self.monitoring_interval = 300  # 5 minutes
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Alert suppression (prevent spam)
        self.alert_cooldown = 1800  # 30 minutes
        self.last_alert_times = {}  # alert_key -> timestamp
        
        logger.info("PerformanceMonitor initialized")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel"""
        self.notification_channels.append(channel)
        logger.info(f"Added notification channel: {type(channel).__name__}")
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                self._check_performance()
            except Exception as e:
                logger.error(f"Performance monitoring check failed: {e}")
    
    def _check_performance(self):
        """Check current performance metrics and trigger alerts if needed"""
        logger.debug("Checking performance metrics...")
        
        # Get current metrics
        metrics = self._collect_current_metrics()
        
        if not metrics:
            logger.warning("No metrics available for performance check")
            return
        
        # Check each performance threshold
        alerts = []
        
        # 1. Accuracy check (requirement: >80%)
        if metrics.accuracy is not None and metrics.accuracy < self.accuracy_threshold:
            alert = PerformanceAlert(
                alert_type=AlertType.ACCURACY_DROP,
                severity=AlertSeverity.CRITICAL if metrics.accuracy < 0.7 else AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metric_name="accuracy",
                current_value=metrics.accuracy,
                threshold=self.accuracy_threshold,
                message=f"Model accuracy dropped to {metrics.accuracy:.3f}, below threshold of {self.accuracy_threshold}",
                metadata={"model_version": metrics.model_version}
            )
            alerts.append(alert)
        
        # 2. Latency check (requirement: <2 seconds)
        if metrics.avg_latency_ms > self.latency_threshold_ms:
            alert = PerformanceAlert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.CRITICAL if metrics.avg_latency_ms > 5000 else AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metric_name="avg_latency_ms",
                current_value=metrics.avg_latency_ms,
                threshold=self.latency_threshold_ms,
                message=f"Average response time is {metrics.avg_latency_ms:.1f}ms, exceeding {self.latency_threshold_ms}ms threshold",
                metadata={
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "p99_latency_ms": metrics.p99_latency_ms
                }
            )
            alerts.append(alert)
        
        # 3. Error rate check
        if metrics.error_rate > self.error_rate_threshold:
            alert = PerformanceAlert(
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.CRITICAL if metrics.error_rate > 0.1 else AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metric_name="error_rate",
                current_value=metrics.error_rate,
                threshold=self.error_rate_threshold,
                message=f"Error rate is {metrics.error_rate:.3f}, exceeding {self.error_rate_threshold} threshold",
                metadata={"active_requests": metrics.active_requests}
            )
            alerts.append(alert)
        
        # 4. Throughput check
        if metrics.throughput_rps < self.throughput_threshold_rps:
            alert = PerformanceAlert(
                alert_type=AlertType.LOW_THROUGHPUT,
                severity=AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metric_name="throughput_rps",
                current_value=metrics.throughput_rps,
                threshold=self.throughput_threshold_rps,
                message=f"Throughput is {metrics.throughput_rps:.2f} RPS, below {self.throughput_threshold_rps} RPS threshold"
            )
            alerts.append(alert)
        
        # 5. System resource checks
        if metrics.cpu_usage > self.cpu_threshold:
            alert = PerformanceAlert(
                alert_type=AlertType.SYSTEM_RESOURCE,
                severity=AlertSeverity.CRITICAL if metrics.cpu_usage > 95 else AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metric_name="cpu_usage",
                current_value=metrics.cpu_usage,
                threshold=self.cpu_threshold,
                message=f"CPU usage is {metrics.cpu_usage:.1f}%, exceeding {self.cpu_threshold}% threshold"
            )
            alerts.append(alert)
        
        if metrics.memory_usage > self.memory_threshold:
            alert = PerformanceAlert(
                alert_type=AlertType.SYSTEM_RESOURCE,
                severity=AlertSeverity.CRITICAL if metrics.memory_usage > 95 else AlertSeverity.WARNING,
                timestamp=datetime.now(),
                metric_name="memory_usage",
                current_value=metrics.memory_usage,
                threshold=self.memory_threshold,
                message=f"Memory usage is {metrics.memory_usage:.1f}%, exceeding {self.memory_threshold}% threshold"
            )
            alerts.append(alert)
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        # Check for drift (less frequent)
        if datetime.now().minute % 15 == 0:  # Every 15 minutes
            self._check_drift()
    
    def _collect_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect current performance metrics"""
        try:
            collector = get_metrics_collector()
            
            # Get recent predictions (last hour)
            recent_predictions = collector.get_recent_predictions(limit=1000)
            if not recent_predictions:
                return None
            
            # Filter to last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_predictions = [
                p for p in recent_predictions
                if datetime.fromisoformat(p['timestamp']) >= one_hour_ago
            ]
            
            if not recent_predictions:
                return None
            
            # Calculate metrics
            successful_preds = [p for p in recent_predictions if p.get('status') == 'success']
            
            # Accuracy (if actual labels available)
            accuracy = None
            labeled_preds = [p for p in successful_preds if p.get('actual_label')]
            if labeled_preds:
                correct = sum(1 for p in labeled_preds if p['prediction'] == p['actual_label'])
                accuracy = correct / len(labeled_preds)
            
            # Latency metrics
            processing_times = [p.get('processing_time_ms', 0) for p in successful_preds]
            if processing_times:
                avg_latency = sum(processing_times) / len(processing_times)
                sorted_times = sorted(processing_times)
                p95_latency = sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0
                p99_latency = sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            # Error rate
            total_requests = len(recent_predictions)
            error_count = total_requests - len(successful_preds)
            error_rate = error_count / total_requests if total_requests > 0 else 0
            
            # Throughput (requests per second)
            time_span_hours = 1.0  # We're looking at 1 hour of data
            throughput_rps = total_requests / (time_span_hours * 3600)
            
            # System metrics (get latest)
            system_metrics = collector.get_recent_system_metrics(limit=1)
            if system_metrics:
                latest_system = system_metrics[0]
                cpu_usage = latest_system.get('cpu_percent', 0)
                memory_usage = latest_system.get('memory_percent', 0)
                gpu_memory_mb = latest_system.get('gpu_memory_mb')
                cache_hit_rate = latest_system.get('cache_hit_rate', 0)
            else:
                cpu_usage = memory_usage = cache_hit_rate = 0
                gpu_memory_mb = None
            
            # Model version (from most recent prediction)
            model_version = recent_predictions[0].get('model_version', 'unknown') if recent_predictions else 'unknown'
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                accuracy=accuracy,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                throughput_rps=throughput_rps,
                error_rate=error_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_memory_mb=gpu_memory_mb,
                active_requests=0,  # Would need to track this separately
                cache_hit_rate=cache_hit_rate,
                model_version=model_version
            )
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return None
    
    def _check_drift(self):
        """Check for drift and create alerts if detected"""
        try:
            drift_report = self.drift_detector.detect_drift(hours=24)
            
            # Create alerts for significant drift
            for drift_alert in drift_report.alerts:
                if drift_alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    alert = PerformanceAlert(
                        alert_type=AlertType.DRIFT_DETECTED,
                        severity=AlertSeverity.CRITICAL if drift_alert.severity == DriftSeverity.CRITICAL else AlertSeverity.WARNING,
                        timestamp=datetime.now(),
                        metric_name=drift_alert.metric_name,
                        current_value=drift_alert.current_value,
                        threshold=drift_alert.threshold,
                        message=f"Drift detected: {drift_alert.description}",
                        metadata={
                            "drift_type": drift_alert.drift_type.value,
                            "drift_severity": drift_alert.severity.value,
                            "overall_drift_score": drift_report.overall_drift_score
                        }
                    )
                    self._process_alert(alert)
                    
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process and potentially send an alert"""
        alert_key = f"{alert.alert_type.value}_{alert.metric_name}"
        
        # Check alert cooldown to prevent spam
        now = datetime.now()
        last_alert_time = self.last_alert_times.get(alert_key)
        
        if last_alert_time and (now - last_alert_time).total_seconds() < self.alert_cooldown:
            logger.debug(f"Alert {alert_key} suppressed due to cooldown")
            return
        
        # Update active alerts
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = now
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {type(channel).__name__}: {e}")
        
        logger.warning(f"Performance alert triggered: {alert.message}")
    
    def resolve_alert(self, alert_key: str) -> bool:
        """Manually resolve an alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_key]
            logger.info(f"Alert resolved: {alert_key}")
            return True
        return False
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        metrics = self._collect_current_metrics()
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(hours)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "time_period_hours": hours,
            "current_metrics": asdict(metrics) if metrics else None,
            "active_alerts_count": len(active_alerts),
            "recent_alerts_count": len(recent_alerts),
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "alert_summary": {
                alert_type.value: len([a for a in recent_alerts if a.alert_type == alert_type])
                for alert_type in AlertType
            }
        }

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def setup_email_notifications(smtp_host: str, smtp_port: int, username: str, 
                            password: str, from_email: str, to_emails: List[str]):
    """Setup email notifications for performance alerts"""
    if not EMAIL_AVAILABLE:
        logger.warning("Email functionality not available, skipping email notifications setup")
        return
        
    monitor = get_performance_monitor()
    email_notifier = EmailNotifier(smtp_host, smtp_port, username, password, from_email, to_emails)
    monitor.add_notification_channel(email_notifier)

def setup_slack_notifications(webhook_url: str):
    """Setup Slack notifications for performance alerts"""
    monitor = get_performance_monitor()
    slack_notifier = SlackNotifier(webhook_url)
    monitor.add_notification_channel(slack_notifier)

if __name__ == "__main__":
    # Test performance monitoring
    print("Testing Performance Monitoring System...")
    
    monitor = PerformanceMonitor()
    
    # Test metrics collection
    metrics = monitor._collect_current_metrics()
    if metrics:
        print(f"✅ Collected metrics: accuracy={metrics.accuracy}, latency={metrics.avg_latency_ms}ms")
    else:
        print("⚠️  No metrics available (expected without data)")
    
    # Test alert creation
    test_alert = PerformanceAlert(
        alert_type=AlertType.ACCURACY_DROP,
        severity=AlertSeverity.WARNING,
        timestamp=datetime.now(),
        metric_name="accuracy",
        current_value=0.75,
        threshold=0.8,
        message="Test alert"
    )
    
    monitor._process_alert(test_alert)
    print(f"✅ Test alert processed: {len(monitor.get_active_alerts())} active alerts")
    
    print("✅ Performance monitoring system initialized!")