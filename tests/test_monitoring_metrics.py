"""
Tests for monitoring metrics collection, drift detection, and performance monitoring
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from monitoring.metrics_collector import (
    MetricsCollector, MetricsDatabase, PredictionMetric, SystemMetric,
    PredictionStatus, get_metrics_collector, record_prediction_metric
)
from monitoring.drift_detector import (
    DriftDetector, DriftType, DriftSeverity, ImageStatistics
)
from monitoring.performance_monitor import (
    PerformanceMonitor, AlertType, AlertSeverity, PerformanceAlert
)


class TestMetricsDatabase:
    """Test MetricsDatabase functionality"""
    
    def test_database_initialization_without_connection(self):
        """Test database initialization when connection fails"""
        # Test with invalid connection string
        db = MetricsDatabase(connection_string="invalid://connection")
        assert db.connection is None
    
    def test_prediction_metric_storage_without_db(self):
        """Test prediction metric storage when database is not available"""
        db = MetricsDatabase(connection_string="invalid://connection")
        
        metric = PredictionMetric(
            prediction_id="test-123",
            timestamp=datetime.now(),
            model_version="v1.0.0",
            prediction="PNEUMONIA",
            confidence=0.95,
            processing_time_ms=45.2,
            image_hash="abc123",
            status=PredictionStatus.SUCCESS
        )
        
        # Should return False when database is not available
        result = db.store_prediction_metric(metric)
        assert result is False
    
    def test_system_metric_storage_without_db(self):
        """Test system metric storage when database is not available"""
        db = MetricsDatabase(connection_string="invalid://connection")
        
        metric = SystemMetric(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            gpu_memory_mb=1024.0,
            disk_usage_percent=70.0,
            request_count=100,
            error_count=5,
            active_connections=10,
            response_time_avg_ms=150.0,
            cache_hit_rate=0.8
        )
        
        # Should return False when database is not available
        result = db.store_system_metric(metric)
        assert result is False
    
    def test_get_metrics_without_db(self):
        """Test getting metrics when database is not available"""
        db = MetricsDatabase(connection_string="invalid://connection")
        
        # Should return empty lists when database is not available
        predictions = db.get_prediction_metrics()
        assert predictions == []
        
        system_metrics = db.get_system_metrics()
        assert system_metrics == []
        
        accuracy = db.get_accuracy_over_time()
        assert accuracy == []


class TestMetricsCollector:
    """Test MetricsCollector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create collector with mock database
        self.mock_db = Mock(spec=MetricsDatabase)
        self.mock_db.connection = None  # Simulate no database connection
        self.mock_db.store_prediction_metric.return_value = True
        self.mock_db.store_system_metric.return_value = True
        self.mock_db.get_model_id.return_value = "test-model-id"
        
        self.collector = MetricsCollector(
            database=self.mock_db,
            enable_file_backup=False
        )
    
    def test_record_prediction_basic(self):
        """Test basic prediction recording"""
        prediction_id = self.collector.record_prediction(
            prediction="PNEUMONIA",
            confidence=0.95,
            processing_time_ms=45.2,
            model_version="v1.0.0"
        )
        
        assert prediction_id is not None
        assert len(self.collector.recent_predictions) == 1
        assert self.collector.request_count == 1
        assert self.collector.predictions_by_class["PNEUMONIA"] == 1
    
    def test_record_prediction_with_image_data(self):
        """Test prediction recording with image data"""
        image_data = b"fake_image_data"
        
        prediction_id = self.collector.record_prediction(
            prediction="NORMAL",
            confidence=0.88,
            processing_time_ms=32.1,
            model_version="v1.0.0",
            image_data=image_data,
            image_id="test-image-123"
        )
        
        assert prediction_id == "test-image-123"
        
        # Check that image hash was generated
        recorded_prediction = self.collector.recent_predictions[0]
        assert recorded_prediction['image_hash'] != "unknown"
    
    def test_record_prediction_with_error(self):
        """Test prediction recording with error status"""
        prediction_id = self.collector.record_prediction(
            prediction="PNEUMONIA",
            confidence=0.0,
            processing_time_ms=100.0,
            model_version="v1.0.0",
            status=PredictionStatus.ERROR,
            error_message="Model loading failed"
        )
        
        assert prediction_id is not None
        assert self.collector.error_count == 1
        
        recorded_prediction = self.collector.recent_predictions[0]
        assert recorded_prediction['status'] == PredictionStatus.ERROR
        assert recorded_prediction['error_message'] == "Model loading failed"
    
    def test_record_prediction_with_actual_label(self):
        """Test prediction recording with actual label for accuracy tracking"""
        prediction_id = self.collector.record_prediction(
            prediction="PNEUMONIA",
            confidence=0.92,
            processing_time_ms=40.0,
            model_version="v1.0.0",
            actual_label="PNEUMONIA"
        )
        
        recorded_prediction = self.collector.recent_predictions[0]
        assert recorded_prediction['actual_label'] == "PNEUMONIA"
    
    def test_record_prediction_with_metadata(self):
        """Test prediction recording with metadata"""
        metadata = {
            "patient_id": "P123456",
            "hospital": "General Hospital",
            "radiologist": "Dr. Smith"
        }
        
        prediction_id = self.collector.record_prediction(
            prediction="NORMAL",
            confidence=0.85,
            processing_time_ms=38.5,
            model_version="v1.0.0",
            metadata=metadata
        )
        
        recorded_prediction = self.collector.recent_predictions[0]
        assert recorded_prediction['metadata'] == metadata
    
    def test_memory_limit_enforcement(self):
        """Test that memory storage respects limits"""
        # Set a small limit for testing
        self.collector.max_memory_metrics = 3
        
        # Record more predictions than the limit
        for i in range(5):
            self.collector.record_prediction(
                prediction="NORMAL",
                confidence=0.8,
                processing_time_ms=30.0,
                model_version="v1.0.0"
            )
        
        # Should only keep the last 3
        assert len(self.collector.recent_predictions) == 3
    
    def test_get_prediction_summary_empty(self):
        """Test prediction summary with no data"""
        # Mock database to return empty list
        self.mock_db.get_prediction_metrics.return_value = []
        
        summary = self.collector.get_prediction_summary(hours=24)
        
        assert summary["total_predictions"] == 0
        assert summary["error_rate"] == 0
        assert summary["avg_confidence"] == 0
        assert summary["avg_processing_time_ms"] == 0
        assert summary["predictions_by_class"]["NORMAL"] == 0
        assert summary["predictions_by_class"]["PNEUMONIA"] == 0
    
    def test_get_prediction_summary_with_data(self):
        """Test prediction summary with recorded data"""
        # Record some predictions
        self.collector.record_prediction("PNEUMONIA", 0.95, 45.0, "v1.0.0")
        self.collector.record_prediction("NORMAL", 0.88, 32.0, "v1.0.0")
        self.collector.record_prediction("PNEUMONIA", 0.0, 100.0, "v1.0.0", 
                                       status=PredictionStatus.ERROR)
        
        # Mock database to return our in-memory data
        self.mock_db.get_prediction_metrics.return_value = []
        
        summary = self.collector.get_prediction_summary(hours=1)
        
        assert summary["total_predictions"] == 3
        assert summary["successful_predictions"] == 2
        assert summary["error_count"] == 1
        assert summary["error_rate"] == 1/3
        assert summary["predictions_by_class"]["PNEUMONIA"] == 1
        assert summary["predictions_by_class"]["NORMAL"] == 1
    
    def test_register_model(self):
        """Test model registration"""
        self.mock_db.register_model.return_value = "model-123"
        
        model_id = self.collector.register_model(
            name="chest-xray-classifier",
            version="v1.0.0",
            architecture="ResNet50",
            metrics={"accuracy": 0.92}
        )
        
        assert model_id == "model-123"
        self.mock_db.register_model.assert_called_once()
    
    def test_register_experiment(self):
        """Test experiment registration"""
        self.mock_db.register_experiment.return_value = "exp-123"
        
        experiment_id = self.collector.register_experiment(
            name="hyperparameter-tuning-1",
            parameters={"learning_rate": 0.001, "batch_size": 32},
            model_name="chest-xray-classifier",
            model_version="v1.0.0"
        )
        
        assert experiment_id == "exp-123"
        self.mock_db.register_experiment.assert_called_once()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    def test_system_metrics_collection(self, mock_connections, mock_disk, 
                                     mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock system calls
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)
        mock_connections.return_value = [Mock()] * 5  # 5 connections
        
        # Collect system metrics
        self.collector._collect_system_metrics()
        
        # Verify database was called
        self.mock_db.store_system_metric.assert_called_once()
        
        # Verify in-memory storage
        assert len(self.collector.recent_system_metrics) == 1
        
        metric = self.collector.recent_system_metrics[0]
        assert metric['cpu_percent'] == 50.0
        assert metric['memory_percent'] == 60.0
        assert metric['disk_usage_percent'] == 70.0
        assert metric['active_connections'] == 5


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('monitoring.metrics_collector.get_metrics_collector')
    def test_record_prediction_metric_function(self, mock_get_collector):
        """Test the convenience function for recording predictions"""
        mock_collector = Mock()
        mock_collector.record_prediction.return_value = "pred-123"
        mock_get_collector.return_value = mock_collector
        
        prediction_id = record_prediction_metric(
            prediction="PNEUMONIA",
            confidence=0.95,
            processing_time_ms=45.0,
            model_version="v1.0.0"
        )
        
        assert prediction_id == "pred-123"
        mock_collector.record_prediction.assert_called_once()


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow_without_database(self):
        """Test full workflow when database is not available"""
        # This simulates the real-world scenario where PostgreSQL might not be available
        collector = MetricsCollector(enable_file_backup=False)
        
        # Should still work without database
        prediction_id = collector.record_prediction(
            prediction="PNEUMONIA",
            confidence=0.95,
            processing_time_ms=45.0,
            model_version="v1.0.0",
            image_data=b"test_image"
        )
        
        assert prediction_id is not None
        
        # Should be able to get summary from memory
        summary = collector.get_prediction_summary(hours=1)
        assert summary["total_predictions"] == 1
        
        # Should be able to get recent predictions from memory
        predictions = collector.get_recent_predictions(limit=10)
        assert len(predictions) == 1
        assert predictions[0]["prediction"] == "PNEUMONIA"


class TestDriftDetector:
    """Test DriftDetector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_db = Mock(spec=MetricsDatabase)
        self.mock_db.connection = None
        self.drift_detector = DriftDetector(database=self.mock_db)
    
    def test_image_statistics_calculation(self):
        """Test image statistics calculation"""
        image_hashes = ["abc123def456", "789ghi012jkl", "345mno678pqr"]
        stats = ImageStatistics.calculate_image_stats(image_hashes)
        
        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert isinstance(stats["mean"], float)
    
    def test_image_statistics_empty_input(self):
        """Test image statistics with empty input"""
        stats = ImageStatistics.calculate_image_stats([])
        assert stats == {}
    
    def test_distribution_comparison(self):
        """Test distribution comparison"""
        baseline_stats = {"mean": 100.0, "std": 10.0, "median": 95.0}
        current_stats = {"mean": 120.0, "std": 15.0, "median": 115.0}
        
        is_drift, drift_score, differences = ImageStatistics.compare_distributions(
            baseline_stats, current_stats, threshold=0.1
        )
        
        assert isinstance(is_drift, bool)
        assert isinstance(drift_score, float)
        assert isinstance(differences, dict)
    
    def test_establish_baseline_insufficient_data(self):
        """Test baseline establishment with insufficient data"""
        self.mock_db.get_prediction_metrics.return_value = []  # No data
        
        result = self.drift_detector.establish_baseline(hours=24)
        
        assert "error" in result
        assert "Insufficient data" in result["error"]
    
    def test_detect_drift_no_baseline(self):
        """Test drift detection without baseline"""
        # Clear baseline
        self.drift_detector.baseline_stats = {}
        
        report = self.drift_detector.detect_drift(hours=24)
        
        assert len(report.alerts) == 0
        assert report.overall_drift_score == 0.0
        assert "Establish baseline" in report.recommendations[0]
    
    def test_detect_drift_insufficient_recent_data(self):
        """Test drift detection with insufficient recent data"""
        # Set up baseline
        self.drift_detector.baseline_stats = {"sample_size": 100}
        
        # Mock insufficient recent data
        self.mock_db.get_prediction_metrics.return_value = []
        
        report = self.drift_detector.detect_drift(hours=24)
        
        assert len(report.alerts) == 0
        assert "Insufficient recent data" in report.recommendations[0]


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_db = Mock(spec=MetricsDatabase)
        self.mock_drift_detector = Mock(spec=DriftDetector)
        self.performance_monitor = PerformanceMonitor(
            database=self.mock_db,
            drift_detector=self.mock_drift_detector
        )
    
    def test_alert_creation(self):
        """Test performance alert creation"""
        alert = PerformanceAlert(
            alert_type=AlertType.ACCURACY_DROP,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            metric_name="accuracy",
            current_value=0.75,
            threshold=0.8,
            message="Accuracy dropped below threshold"
        )
        
        assert alert.alert_type == AlertType.ACCURACY_DROP
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.current_value == 0.75
        assert not alert.resolved
    
    def test_process_alert(self):
        """Test alert processing"""
        alert = PerformanceAlert(
            alert_type=AlertType.HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            metric_name="avg_latency_ms",
            current_value=2500.0,
            threshold=2000.0,
            message="High latency detected"
        )
        
        self.performance_monitor._process_alert(alert)
        
        # Check that alert was added to active alerts
        active_alerts = self.performance_monitor.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_type == AlertType.HIGH_LATENCY
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism"""
        alert1 = PerformanceAlert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            metric_name="error_rate",
            current_value=0.1,
            threshold=0.05,
            message="High error rate"
        )
        
        alert2 = PerformanceAlert(
            alert_type=AlertType.HIGH_ERROR_RATE,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            metric_name="error_rate",
            current_value=0.12,
            threshold=0.05,
            message="High error rate continues"
        )
        
        # Process first alert
        self.performance_monitor._process_alert(alert1)
        assert len(self.performance_monitor.get_active_alerts()) == 1
        
        # Process second alert immediately (should be suppressed)
        self.performance_monitor._process_alert(alert2)
        assert len(self.performance_monitor.get_active_alerts()) == 1  # Still only one
    
    def test_resolve_alert(self):
        """Test manual alert resolution"""
        alert = PerformanceAlert(
            alert_type=AlertType.SYSTEM_RESOURCE,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(),
            metric_name="cpu_usage",
            current_value=95.0,
            threshold=90.0,
            message="High CPU usage"
        )
        
        self.performance_monitor._process_alert(alert)
        assert len(self.performance_monitor.get_active_alerts()) == 1
        
        # Resolve alert
        alert_key = f"{alert.alert_type.value}_{alert.metric_name}"
        success = self.performance_monitor.resolve_alert(alert_key)
        
        assert success
        assert len(self.performance_monitor.get_active_alerts()) == 0
    
    def test_get_performance_summary(self):
        """Test performance summary generation"""
        summary = self.performance_monitor.get_performance_summary(hours=24)
        
        assert "timestamp" in summary
        assert "time_period_hours" in summary
        assert "active_alerts_count" in summary
        assert "recent_alerts_count" in summary
        assert summary["time_period_hours"] == 24
    
    @patch('monitoring.performance_monitor.get_metrics_collector')
    def test_collect_current_metrics_no_data(self, mock_get_collector):
        """Test metrics collection with no data"""
        mock_collector = Mock()
        mock_collector.get_recent_predictions.return_value = []
        mock_get_collector.return_value = mock_collector
        
        metrics = self.performance_monitor._collect_current_metrics()
        
        assert metrics is None
    
    @patch('monitoring.performance_monitor.get_metrics_collector')
    def test_collect_current_metrics_with_data(self, mock_get_collector):
        """Test metrics collection with sample data"""
        mock_collector = Mock()
        
        # Mock recent predictions
        sample_predictions = [
            {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "prediction": "PNEUMONIA",
                "confidence": 0.95,
                "processing_time_ms": 45.0,
                "model_version": "v1.0.0",
                "actual_label": "PNEUMONIA"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "prediction": "NORMAL",
                "confidence": 0.88,
                "processing_time_ms": 32.0,
                "model_version": "v1.0.0",
                "actual_label": "NORMAL"
            }
        ]
        
        mock_collector.get_recent_predictions.return_value = sample_predictions
        mock_collector.get_recent_system_metrics.return_value = [
            {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "cache_hit_rate": 0.8
            }
        ]
        
        mock_get_collector.return_value = mock_collector
        
        metrics = self.performance_monitor._collect_current_metrics()
        
        assert metrics is not None
        assert metrics.accuracy == 1.0  # Both predictions correct
        assert metrics.throughput_rps > 0
        assert metrics.error_rate == 0.0
        assert metrics.cpu_usage == 50.0


class TestNotificationChannels:
    """Test notification channel functionality"""
    
    def test_email_notifier_creation(self):
        """Test email notifier creation"""
        from monitoring.performance_monitor import EmailNotifier
        
        notifier = EmailNotifier(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            from_email="alerts@example.com",
            to_emails=["admin@example.com"]
        )
        
        assert notifier.smtp_host == "smtp.example.com"
        assert notifier.smtp_port == 587
        assert len(notifier.to_emails) == 1
    
    def test_slack_notifier_creation(self):
        """Test Slack notifier creation"""
        from monitoring.performance_monitor import SlackNotifier
        
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        
        assert notifier.webhook_url == "https://hooks.slack.com/test"


class TestPrometheusExporter:
    """Test Prometheus exporter functionality"""
    
    def test_prometheus_exporter_creation(self):
        """Test Prometheus exporter creation"""
        from monitoring.prometheus_exporter import PrometheusExporter
        
        exporter = PrometheusExporter(port=8080, update_interval=30)
        
        assert exporter.port == 8080
        assert exporter.update_interval == 30
        assert not exporter.running
    
    def test_record_prediction_metrics(self):
        """Test recording prediction metrics"""
        from monitoring.prometheus_exporter import PrometheusExporter
        
        exporter = PrometheusExporter(port=8080)
        
        # This should not raise an error even if Prometheus is not available
        exporter.record_prediction(
            prediction="PNEUMONIA",
            model_version="v1.0.0",
            processing_time_seconds=0.045,
            success=True
        )


class TestIntegrationWithNewComponents:
    """Integration tests with drift detection and performance monitoring"""
    
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # Create components
        collector = MetricsCollector(enable_file_backup=False)
        drift_detector = DriftDetector(database=collector.database)
        performance_monitor = PerformanceMonitor(
            database=collector.database,
            drift_detector=drift_detector
        )
        
        # Record some predictions
        collector.record_prediction("PNEUMONIA", 0.95, 45.0, "v1.0.0")
        collector.record_prediction("NORMAL", 0.88, 32.0, "v1.0.0")
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary(hours=1)
        
        assert summary["time_period_hours"] == 1
        assert "active_alerts_count" in summary
        
        # Test drift detection (will have no baseline, but should not crash)
        report = drift_detector.detect_drift(hours=1)
        
        assert report.overall_drift_score == 0.0
        assert len(report.recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__])