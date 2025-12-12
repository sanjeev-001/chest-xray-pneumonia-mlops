"""
Comprehensive Tests for Monitoring Components
Tests metrics collection, drift detection, performance monitoring, and alerting
"""

import pytest
import numpy as np
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import os
import sqlite3

# Import monitoring components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.metrics_collector import (
    MetricsCollector, MetricsDatabase, 
    PredictionMetric, PredictionStatus, SystemMetric
)
from monitoring.performance_monitor import (
    PerformanceMonitor, AlertType, AlertSeverity, PerformanceAlert,
    PerformanceMetrics, EmailNotifier, SlackNotifier, WebhookNotifier
)
from monitoring.drift_detector import (
    DriftDetector, DriftType, DriftSeverity, DriftAlert, DriftReport,
    ImageStatistics
)

class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def test_system_metric_creation(self):
        """Test system metric data structure creation"""
        timestamp = datetime.now()
        metric = SystemMetric(
            timestamp=timestamp,
            cpu_percent=45.2,
            memory_percent=60.5,
            gpu_memory_mb=2048.0,
            disk_usage_percent=75.0,
            request_count=100,
            error_count=2,
            active_connections=5,
            response_time_avg_ms=150.5,
            cache_hit_rate=0.85
        )
        
        assert metric.timestamp == timestamp
        assert metric.cpu_percent == 45.2
        assert metric.memory_percent == 60.5
        assert metric.gpu_memory_mb == 2048.0
        assert metric.request_count == 100
        assert metric.cache_hit_rate == 0.85
    
    def test_prediction_metric_creation(self):
        """Test prediction metric structure"""
        timestamp = datetime.now()
        pred_metric = PredictionMetric(
            prediction_id="pred_123",
            timestamp=timestamp,
            model_version="v1.0",
            prediction="PNEUMONIA",
            confidence=0.85,
            processing_time_ms=150.5,
            image_hash="abc123",
            status=PredictionStatus.SUCCESS,
            actual_label="PNEUMONIA",
            metadata={"source": "test"}
        )
        
        assert pred_metric.prediction_id == "pred_123"
        assert pred_metric.prediction == "PNEUMONIA"
        assert pred_metric.confidence == 0.85
        assert pred_metric.status == PredictionStatus.SUCCESS
        assert pred_metric.actual_label == "PNEUMONIA"
    
    def test_metrics_database_initialization(self):
        """Test metrics database initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_metrics.db")
            
            # Test with PostgreSQL unavailable (should handle gracefully)
            with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
                db = MetricsDatabase(db_path)
                
                # Should initialize without error even if PostgreSQL is unavailable
                assert db is not None
                assert db.connection is None  # No connection when PostgreSQL unavailable
    
    def test_metrics_database_storage(self):
        """Test storing metrics in database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_metrics.db")
            
            with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
                db = MetricsDatabase(db_path)
                
                # Test storing prediction metric (should return False when PostgreSQL unavailable)
                pred_metric = PredictionMetric(
                    prediction_id="test_pred",
                    timestamp=datetime.now(),
                    model_version="v1.0",
                    prediction="NORMAL",
                    confidence=0.9,
                    processing_time_ms=100.0,
                    image_hash="test_hash",
                    status=PredictionStatus.SUCCESS
                )
                
                success = db.store_prediction_metric(pred_metric)
                # Should return False when database is not available
                assert success is False
    
    def test_system_metrics_collection(self):
        """Test system metrics collection using available classes"""
        # Test creating a system metric directly
        timestamp = datetime.now()
        
        system_metric = SystemMetric(
            timestamp=timestamp,
            cpu_percent=45.2,
            memory_percent=60.5,
            gpu_memory_mb=2048.0,
            disk_usage_percent=75.0,
            request_count=100,
            error_count=2,
            active_connections=5,
            response_time_avg_ms=150.5,
            cache_hit_rate=0.85
        )
        
        # Verify the metric was created correctly
        assert system_metric.cpu_percent == 45.2
        assert system_metric.memory_percent == 60.5
        assert system_metric.gpu_memory_mb == 2048.0
        assert system_metric.timestamp == timestamp
    
    def test_prediction_metric_validation(self):
        """Test prediction metric validation and processing"""
        timestamp = datetime.now()
        
        # Test valid prediction metric
        pred_metric = PredictionMetric(
            prediction_id="test_pred_001",
            timestamp=timestamp,
            model_version="v1.0",
            prediction="PNEUMONIA",
            confidence=0.87,
            processing_time_ms=150.5,
            image_hash="abc123def456",
            status=PredictionStatus.SUCCESS,
            actual_label="PNEUMONIA",
            metadata={"source": "test"}
        )
        
        assert pred_metric.prediction_id == "test_pred_001"
        assert pred_metric.confidence == 0.87
        assert pred_metric.status == PredictionStatus.SUCCESS
        assert pred_metric.metadata["source"] == "test"
        
        # Test error status
        error_metric = PredictionMetric(
            prediction_id="test_pred_002",
            timestamp=timestamp,
            model_version="v1.0",
            prediction="NORMAL",
            confidence=0.0,
            processing_time_ms=0.0,
            image_hash="error_hash",
            status=PredictionStatus.ERROR,
            error_message="Model loading failed"
        )
        
        assert error_metric.status == PredictionStatus.ERROR
        assert error_metric.error_message == "Model loading failed"
    
    def test_metrics_collector_integration(self):
        """Test integrated metrics collector"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_metrics.db")
            
            with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
                db = MetricsDatabase(db_path)
                
                # Test storing prediction metric (should handle gracefully when DB unavailable)
                pred_metric = PredictionMetric(
                    prediction_id="integration_test",
                    timestamp=datetime.now(),
                    model_version="v1.0",
                    prediction="NORMAL",
                    confidence=0.9,
                    processing_time_ms=150.0,
                    image_hash="test_hash",
                    status=PredictionStatus.SUCCESS
                )
                
                success = db.store_prediction_metric(pred_metric)
                # Should return False when database is not available
                assert success is False
                
                # Test storing system metric (should also handle gracefully)
                system_metric = SystemMetric(
                    timestamp=datetime.now(),
                    cpu_percent=45.2,
                    memory_percent=60.5,
                    gpu_memory_mb=2048.0,
                    disk_usage_percent=75.0,
                    request_count=100,
                    error_count=2,
                    active_connections=5,
                    response_time_avg_ms=150.5,
                    cache_hit_rate=0.85
                )
                
                success = db.store_system_metric(system_metric)
                # Should return False when database is not available
                assert success is False

class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def test_performance_alert_creation(self):
        """Test performance alert structure"""
        alert = PerformanceAlert(
            alert_type=AlertType.ACCURACY_DROP,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            metric_name="accuracy",
            current_value=0.75,
            threshold=0.8,
            message="Model accuracy dropped below threshold",
            metadata={"model_version": "v1.0"}
        )
        
        assert alert.alert_type == AlertType.ACCURACY_DROP
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.current_value == 0.75
        assert alert.threshold == 0.8
        assert alert.resolved is False
    
    def test_performance_metrics_structure(self):
        """Test performance metrics data structure"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=0.92,
            avg_latency_ms=150.5,
            p95_latency_ms=250.0,
            p99_latency_ms=400.0,
            throughput_rps=10.5,
            error_rate=0.02,
            cpu_usage=45.2,
            memory_usage=60.8,
            gpu_memory_mb=2048.0,
            active_requests=5,
            cache_hit_rate=0.85,
            model_version="v1.0"
        )
        
        assert metrics.accuracy == 0.92
        assert metrics.avg_latency_ms == 150.5
        assert metrics.throughput_rps == 10.5
        assert metrics.error_rate == 0.02
    
    def test_email_notifier(self):
        """Test email notification functionality"""
        with patch('monitoring.performance_monitor.EMAIL_AVAILABLE', True):
            with patch('monitoring.performance_monitor.smtplib.SMTP') as mock_smtp:
                mock_server = Mock()
                mock_smtp.return_value = mock_server
                
                notifier = EmailNotifier(
                    smtp_host="smtp.test.com",
                    smtp_port=587,
                    username="test@test.com",
                    password="password",
                    from_email="alerts@test.com",
                    to_emails=["admin@test.com"]
                )
                
                alert = PerformanceAlert(
                    alert_type=AlertType.HIGH_LATENCY,
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    metric_name="latency",
                    current_value=3000,
                    threshold=2000,
                    message="High latency detected"
                )
                
                success = notifier.send_alert(alert)
                assert success
                
                # Verify SMTP calls
                mock_server.starttls.assert_called_once()
                mock_server.login.assert_called_once()
                mock_server.send_message.assert_called_once()
                mock_server.quit.assert_called_once()
    
    def test_slack_notifier(self):
        """Test Slack notification functionality"""
        with patch('monitoring.performance_monitor.REQUESTS_AVAILABLE', True):
            with patch('monitoring.performance_monitor.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                
                notifier = SlackNotifier("https://hooks.slack.com/test")
                
                alert = PerformanceAlert(
                    alert_type=AlertType.DRIFT_DETECTED,
                    severity=AlertSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    metric_name="drift_score",
                    current_value=0.8,
                    threshold=0.5,
                    message="Data drift detected"
                )
                
                success = notifier.send_alert(alert)
                assert success
                
                # Verify request was made
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert "attachments" in call_args[1]["json"]
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        mock_database = Mock()
        mock_drift_detector = Mock()
        
        monitor = PerformanceMonitor(
            database=mock_database,
            drift_detector=mock_drift_detector
        )
        
        assert monitor.database == mock_database
        assert monitor.drift_detector == mock_drift_detector
        assert monitor.accuracy_threshold == 0.8
        assert monitor.latency_threshold_ms == 2000
        assert monitor.error_rate_threshold == 0.05
    
    def test_performance_monitor_alert_processing(self):
        """Test alert processing and suppression"""
        mock_database = Mock()
        monitor = PerformanceMonitor(database=mock_database)
        
        # Add mock notification channel
        mock_channel = Mock()
        mock_channel.send_alert.return_value = True
        monitor.add_notification_channel(mock_channel)
        
        # Create test alert
        alert = PerformanceAlert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            metric_name="error_rate",
            current_value=0.08,
            threshold=0.05,
            message="Error rate exceeded threshold"
        )
        
        # Process alert
        monitor._process_alert(alert)
        
        # Verify alert was processed
        alert_key = f"{alert.alert_type.value}_{alert.metric_name}"
        assert alert_key in monitor.active_alerts
        assert len(monitor.alert_history) == 1
        
        # Verify notification was sent
        mock_channel.send_alert.assert_called_once_with(alert)
        
        # Test alert suppression (cooldown)
        monitor._process_alert(alert)  # Same alert again
        
        # Should still be only one call due to cooldown
        assert mock_channel.send_alert.call_count == 1
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection"""
        mock_database = Mock()
        monitor = PerformanceMonitor(database=mock_database)
        
        # Mock metrics collector
        with patch('monitoring.performance_monitor.get_metrics_collector') as mock_get_collector:
            mock_collector = Mock()
            
            # Mock recent predictions
            mock_collector.get_recent_predictions.return_value = [
                {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success',
                    'processing_time_ms': 150.0,
                    'actual_label': 'NORMAL',
                    'prediction': 'NORMAL'
                },
                {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success',
                    'processing_time_ms': 200.0,
                    'actual_label': 'PNEUMONIA',
                    'prediction': 'PNEUMONIA'
                }
            ]
            
            # Mock system metrics
            mock_collector.get_recent_system_metrics.return_value = [
                {
                    'cpu_percent': 45.2,
                    'memory_percent': 60.8,
                    'gpu_memory_mb': 2048.0,
                    'cache_hit_rate': 0.85
                }
            ]
            
            mock_get_collector.return_value = mock_collector
            
            metrics = monitor._collect_current_metrics()
            
            assert metrics is not None
            assert metrics.accuracy == 1.0  # Both predictions correct
            assert metrics.avg_latency_ms == 175.0  # Average of 150 and 200
            assert metrics.error_rate == 0.0  # No errors
            assert metrics.cpu_usage == 45.2
            assert metrics.memory_usage == 60.8

class TestDriftDetector:
    """Test drift detection functionality"""
    
    def test_drift_alert_creation(self):
        """Test drift alert structure"""
        alert = DriftAlert(
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.HIGH,
            timestamp=datetime.now(),
            metric_name="image_statistics",
            current_value=0.8,
            baseline_value=0.5,
            threshold=0.6,
            p_value=0.01,
            description="Significant data drift detected"
        )
        
        assert alert.drift_type == DriftType.DATA_DRIFT
        assert alert.severity == DriftSeverity.HIGH
        assert alert.current_value == 0.8
        assert alert.baseline_value == 0.5
        assert alert.p_value == 0.01
    
    def test_drift_report_structure(self):
        """Test drift report structure"""
        alerts = [
            DriftAlert(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.MEDIUM,
                timestamp=datetime.now(),
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                threshold=0.8,
                description="Model accuracy drift"
            )
        ]
        
        report = DriftReport(
            timestamp=datetime.now(),
            time_window_hours=24,
            alerts=alerts,
            data_drift_score=0.3,
            concept_drift_score=0.6,
            prediction_drift_score=0.2,
            overall_drift_score=0.6,
            recommendations=["Consider model retraining"]
        )
        
        assert len(report.alerts) == 1
        assert report.overall_drift_score == 0.6
        assert "Consider model retraining" in report.recommendations
    
    def test_image_statistics_calculation(self):
        """Test image statistics calculation"""
        # Test with sample hash values
        image_hashes = ["abc123", "def456", "789ghi", "jkl012"]
        
        stats = ImageStatistics.calculate_image_stats(image_hashes)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        
        # Test with empty list
        empty_stats = ImageStatistics.calculate_image_stats([])
        assert empty_stats == {}
    
    def test_distribution_comparison(self):
        """Test statistical distribution comparison"""
        baseline_stats = {
            "mean": 100.0,
            "std": 15.0,
            "median": 98.0,
            "skewness": 0.1,
            "kurtosis": -0.2
        }
        
        # Similar distribution (no drift)
        current_stats_similar = {
            "mean": 102.0,
            "std": 16.0,
            "median": 99.0,
            "skewness": 0.12,
            "kurtosis": -0.18
        }
        
        is_drift, drift_score, differences = ImageStatistics.compare_distributions(
            baseline_stats, current_stats_similar, threshold=0.05
        )
        
        assert not is_drift  # Should not detect drift for similar distributions
        assert drift_score < 0.3
        
        # Very different distribution (drift)
        current_stats_different = {
            "mean": 150.0,  # 50% increase
            "std": 25.0,    # 67% increase
            "median": 145.0,
            "skewness": 0.5,
            "kurtosis": 1.0
        }
        
        is_drift, drift_score, differences = ImageStatistics.compare_distributions(
            baseline_stats, current_stats_different, threshold=0.05
        )
        
        assert is_drift  # Should detect drift for very different distributions
        assert drift_score > 0.3
    
    def test_drift_detector_initialization(self):
        """Test drift detector initialization"""
        mock_database = Mock()
        
        detector = DriftDetector(database=mock_database)
        
        assert detector.database == mock_database
        assert detector.data_drift_threshold == 0.05
        assert detector.concept_drift_threshold == 0.1
        assert detector.prediction_drift_threshold == 0.15
    
    def test_baseline_establishment(self):
        """Test baseline statistics establishment"""
        mock_database = Mock()
        
        # Mock historical predictions
        mock_predictions = [
            {
                'status': 'success',
                'image_hash': 'hash1',
                'confidence': 0.9,
                'processing_time': 150,
                'prediction': 'NORMAL',
                'actual_label': 'NORMAL'
            },
            {
                'status': 'success',
                'image_hash': 'hash2',
                'confidence': 0.85,
                'processing_time': 200,
                'prediction': 'PNEUMONIA',
                'actual_label': 'PNEUMONIA'
            }
        ] * 50  # 100 predictions total
        
        mock_database.get_prediction_metrics.return_value = mock_predictions
        
        detector = DriftDetector(database=mock_database)
        
        with patch.object(detector, '_save_baseline_stats'):
            result = detector.establish_baseline(hours=168)
            
            assert result["status"] == "baseline_established"
            assert result["predictions_count"] == 100
            assert "baseline_stats" in result
    
    def test_drift_detection_without_baseline(self):
        """Test drift detection when no baseline exists"""
        mock_database = Mock()
        detector = DriftDetector(database=mock_database)
        
        # No baseline stats
        detector.baseline_stats = {}
        
        report = detector.detect_drift(hours=24)
        
        assert len(report.alerts) == 0
        assert report.overall_drift_score == 0.0
        assert "Establish baseline statistics" in report.recommendations[0]
    
    def test_concept_drift_detection(self):
        """Test concept drift detection"""
        mock_database = Mock()
        detector = DriftDetector(database=mock_database)
        
        # Set baseline with high accuracy
        detector.baseline_stats = {
            'accuracy': 0.95,
            'image_statistics': {'mean': 100, 'std': 15},
            'prediction_distribution': {'NORMAL': 0.6, 'PNEUMONIA': 0.4}
        }
        
        # Mock recent predictions with lower accuracy
        mock_recent_predictions = [
            {
                'status': 'success',
                'image_hash': 'hash1',
                'prediction': 'NORMAL',
                'actual_label': 'PNEUMONIA'  # Incorrect
            },
            {
                'status': 'success',
                'image_hash': 'hash2',
                'prediction': 'PNEUMONIA',
                'actual_label': 'PNEUMONIA'  # Correct
            }
        ] * 10  # 20 predictions, 50% accuracy
        
        mock_database.get_prediction_metrics.return_value = mock_recent_predictions
        
        with patch.object(detector, '_calculate_baseline_stats') as mock_calc:
            mock_calc.return_value = {
                'accuracy': 0.5,  # Dropped from 0.95 to 0.5
                'image_statistics': {'mean': 100, 'std': 15},
                'prediction_distribution': {'NORMAL': 0.6, 'PNEUMONIA': 0.4}
            }
            
            report = detector.detect_drift(hours=24)
            
            # Should detect concept drift due to accuracy drop
            concept_alerts = [a for a in report.alerts if a.drift_type == DriftType.CONCEPT_DRIFT]
            assert len(concept_alerts) > 0
            
            concept_alert = concept_alerts[0]
            assert concept_alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            assert concept_alert.current_value == 0.5
            assert concept_alert.baseline_value == 0.95

class TestAlertingSystem:
    """Test alerting and notification systems"""
    
    def test_webhook_notifier(self):
        """Test webhook notification functionality"""
        with patch('monitoring.performance_monitor.REQUESTS_AVAILABLE', True):
            with patch('monitoring.performance_monitor.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response
                
                notifier = WebhookNotifier(
                    webhook_url="https://api.test.com/webhook",
                    headers={"Authorization": "Bearer token123"}
                )
                
                alert = PerformanceAlert(
                    alert_type=AlertType.SYSTEM_RESOURCE,
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    metric_name="cpu_usage",
                    current_value=95.0,
                    threshold=90.0,
                    message="High CPU usage detected"
                )
                
                success = notifier.send_alert(alert)
                assert success
                
                # Verify webhook call
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                
                assert call_args[0][0] == "https://api.test.com/webhook"
                assert "Authorization" in call_args[1]["headers"]
                
                payload = call_args[1]["json"]
                assert payload["alert_type"] == "system_resource"
                assert payload["severity"] == "warning"
                assert payload["current_value"] == 95.0
    
    def test_notification_failure_handling(self):
        """Test notification failure handling"""
        with patch('monitoring.performance_monitor.REQUESTS_AVAILABLE', True):
            with patch('monitoring.performance_monitor.requests.post') as mock_post:
                # Simulate network error
                mock_post.side_effect = Exception("Network error")
                
                notifier = SlackNotifier("https://hooks.slack.com/test")
                
                alert = PerformanceAlert(
                    alert_type=AlertType.MODEL_FAILURE,
                    severity=AlertSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    metric_name="model_status",
                    current_value=0,
                    threshold=1,
                    message="Model failed to load"
                )
                
                success = notifier.send_alert(alert)
                assert not success  # Should return False on failure
    
    def test_alert_cooldown_mechanism(self):
        """Test alert cooldown to prevent spam"""
        mock_database = Mock()
        monitor = PerformanceMonitor(database=mock_database)
        monitor.alert_cooldown = 1  # 1 second cooldown for testing
        
        mock_channel = Mock()
        mock_channel.send_alert.return_value = True
        monitor.add_notification_channel(mock_channel)
        
        alert = PerformanceAlert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            metric_name="latency",
            current_value=3000,
            threshold=2000,
            message="High latency"
        )
        
        # Send first alert
        monitor._process_alert(alert)
        assert mock_channel.send_alert.call_count == 1
        
        # Send same alert immediately (should be suppressed)
        monitor._process_alert(alert)
        assert mock_channel.send_alert.call_count == 1
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # Send alert again (should go through)
        monitor._process_alert(alert)
        assert mock_channel.send_alert.call_count == 2
    
    def test_alert_resolution(self):
        """Test manual alert resolution"""
        mock_database = Mock()
        monitor = PerformanceMonitor(database=mock_database)
        
        alert = PerformanceAlert(
            alert_type=AlertType.ACCURACY_DROP,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            metric_name="accuracy",
            current_value=0.75,
            threshold=0.8,
            message="Accuracy dropped"
        )
        
        # Process alert
        monitor._process_alert(alert)
        alert_key = f"{alert.alert_type.value}_{alert.metric_name}"
        
        assert alert_key in monitor.active_alerts
        assert not monitor.active_alerts[alert_key].resolved
        
        # Resolve alert
        success = monitor.resolve_alert(alert_key)
        assert success
        assert alert_key not in monitor.active_alerts
        
        # Try to resolve non-existent alert
        success = monitor.resolve_alert("non_existent_alert")
        assert not success

class TestIntegratedMonitoring:
    """Test integrated monitoring system functionality"""
    
    def test_monitoring_system_startup(self):
        """Test monitoring system startup and shutdown"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_metrics.db")
            
            with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
                db = MetricsDatabase(db_path)
                collector = MetricsCollector(db, collection_interval=0.1)  # Fast interval for testing
                
                # Start collection
                collector.start_collection()
                assert collector.running
                
                # Let it run briefly
                time.sleep(0.2)
                
                # Stop collection
                collector.stop_collection()
                assert not collector.running
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring with real metrics"""
        mock_database = Mock()
        mock_drift_detector = Mock()
        
        # Mock drift detection
        mock_drift_report = DriftReport(
            timestamp=datetime.now(),
            time_window_hours=24,
            alerts=[],
            data_drift_score=0.2,
            concept_drift_score=0.1,
            prediction_drift_score=0.15,
            overall_drift_score=0.2,
            recommendations=[]
        )
        mock_drift_detector.detect_drift.return_value = mock_drift_report
        
        monitor = PerformanceMonitor(
            database=mock_database,
            drift_detector=mock_drift_detector
        )
        
        # Mock metrics collection
        with patch.object(monitor, '_collect_current_metrics') as mock_collect:
            mock_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                accuracy=0.75,  # Below threshold
                avg_latency_ms=2500,  # Above threshold
                p95_latency_ms=3000,
                p99_latency_ms=4000,
                throughput_rps=0.5,  # Below threshold
                error_rate=0.08,  # Above threshold
                cpu_usage=95,  # Above threshold
                memory_usage=92,  # Above threshold
                gpu_memory_mb=4096,
                active_requests=10,
                cache_hit_rate=0.7,
                model_version="v1.0"
            )
            mock_collect.return_value = mock_metrics
            
            # Add notification channel
            mock_channel = Mock()
            mock_channel.send_alert.return_value = True
            monitor.add_notification_channel(mock_channel)
            
            # Run performance check
            monitor._check_performance()
            
            # Should generate multiple alerts
            assert len(monitor.active_alerts) > 0
            assert mock_channel.send_alert.call_count > 0
            
            # Check for specific alert types
            alert_types = [alert.alert_type for alert in monitor.active_alerts.values()]
            assert AlertType.ACCURACY_DROP in alert_types
            assert AlertType.HIGH_LATENCY in alert_types
            assert AlertType.HIGH_ERROR_RATE in alert_types
            assert AlertType.SYSTEM_RESOURCE in alert_types
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_metrics.db")
            
            with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
                # Initialize components
                db = MetricsDatabase(db_path)
                collector = MetricsCollector(db, collection_interval=0.1)
                
                # Record some prediction metrics
                pred_metric = PredictionMetric(
                    prediction_id="test_pred_1",
                    timestamp=datetime.now(),
                    model_version="v1.0",
                    prediction="PNEUMONIA",
                    confidence=0.65,  # Low confidence
                    processing_time_ms=2500,  # High latency
                    image_hash="test_hash_1",
                    status=PredictionStatus.SUCCESS,
                    actual_label="NORMAL"  # Incorrect prediction
                )
                
                success = db.store_prediction_metric(pred_metric)
                assert success
                
                # Initialize performance monitor
                drift_detector = DriftDetector(database=db)
                performance_monitor = PerformanceMonitor(
                    database=db,
                    drift_detector=drift_detector
                )
                
                # Add mock notification
                mock_channel = Mock()
                mock_channel.send_alert.return_value = True
                performance_monitor.add_notification_channel(mock_channel)
                
                # Mock metrics collection to return our test data
                with patch.object(performance_monitor, '_collect_current_metrics') as mock_collect:
                    mock_metrics = PerformanceMetrics(
                        timestamp=datetime.now(),
                        accuracy=0.0,  # 0% accuracy (incorrect prediction)
                        avg_latency_ms=2500,  # High latency
                        p95_latency_ms=3000,
                        p99_latency_ms=4000,
                        throughput_rps=0.1,
                        error_rate=0.0,
                        cpu_usage=50,
                        memory_usage=60,
                        gpu_memory_mb=2048,
                        active_requests=1,
                        cache_hit_rate=0.8,
                        model_version="v1.0"
                    )
                    mock_collect.return_value = mock_metrics
                    
                    # Run performance check
                    performance_monitor._check_performance()
                    
                    # Verify alerts were generated
                    assert len(performance_monitor.active_alerts) > 0
                    
                    # Check for accuracy and latency alerts
                    alert_types = [alert.alert_type for alert in performance_monitor.active_alerts.values()]
                    assert AlertType.ACCURACY_DROP in alert_types
                    assert AlertType.HIGH_LATENCY in alert_types
                    
                    # Verify notifications were sent
                    assert mock_channel.send_alert.call_count > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])