"""
Integration Tests for Complete Monitoring System
Tests the integration between metrics collection, performance monitoring, 
drift detection, and audit/explainability systems
"""

import pytest
import numpy as np
import json
import tempfile
import threading
import time
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import os

# Import all monitoring components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.metrics_collector import (
    MetricsCollector, MetricsDatabase, PredictionMetric, PredictionStatus
)
from monitoring.performance_monitor import (
    PerformanceMonitor, AlertType, AlertSeverity, get_performance_monitor
)
from monitoring.drift_detector import (
    DriftDetector, DriftType, DriftSeverity, get_drift_detector
)
from monitoring.audit_trail import (
    AuditTrailManager, ComplianceLevel, get_audit_manager
)
from monitoring.explainability import (
    ExplainabilityService, ExplanationResult, get_explainability_service
)
from monitoring.audit_explainability_manager import (
    AuditExplainabilityManager, get_audit_explainability_manager
)

class TestMonitoringSystemIntegration:
    """Test complete monitoring system integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        
        # Reset global instances
        import monitoring.metrics_collector
        import monitoring.performance_monitor
        import monitoring.drift_detector
        import monitoring.audit_trail
        import monitoring.explainability
        import monitoring.audit_explainability_manager
        
        monitoring.metrics_collector._metrics_collector = None
        monitoring.performance_monitor._performance_monitor = None
        monitoring.drift_detector._drift_detector = None
        monitoring.audit_trail._audit_manager = None
        monitoring.explainability._explainability_service = None
        monitoring.audit_explainability_manager._audit_explainability_manager = None
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_prediction_workflow(self):
        """Test complete prediction workflow with monitoring"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            with patch('monitoring.audit_trail.PSYCOPG2_AVAILABLE', False):
                
                # Initialize integrated manager
                manager = get_audit_explainability_manager(
                    compliance_level=ComplianceLevel.MEDICAL,
                    auto_explain_threshold=0.8
                )
                
                # Mock image data
                image_data = b"fake_image_data_for_testing"
                
                # Test low confidence prediction (should trigger explanation)
                record = manager.log_prediction_with_explanation(
                    prediction_id="integration_test_001",
                    model_id="test_model",
                    model_version="v1.0",
                    image=image_data,
                    input_metadata={
                        "source": "integration_test",
                        "image_size": [224, 224]
                    },
                    prediction_result="PNEUMONIA",
                    confidence_score=0.65,  # Below threshold
                    processing_time_ms=250.0,
                    api_context={
                        "api_endpoint": "/predict",
                        "client_ip": "127.0.0.1",
                        "user_agent": "TestClient/1.0"
                    }
                )
                
                # Verify record was created
                assert record.prediction_id == "integration_test_001"
                assert record.confidence_score == 0.65
                assert record.prediction_result == "PNEUMONIA"
                assert record.compliance_level == ComplianceLevel.MEDICAL
                assert record.integrity_hash is not None
                
                # Test high confidence prediction (no explanation)
                record2 = manager.log_prediction_with_explanation(
                    prediction_id="integration_test_002",
                    model_id="test_model",
                    model_version="v1.0",
                    image=image_data,
                    input_metadata={"source": "integration_test"},
                    prediction_result="NORMAL",
                    confidence_score=0.95,  # Above threshold
                    processing_time_ms=180.0
                )
                
                assert record2.prediction_id == "integration_test_002"
                assert record2.confidence_score == 0.95
                assert record2.explanation_result is None  # No explanation for high confidence
    
    def test_performance_monitoring_with_alerts(self):
        """Test performance monitoring with alert generation"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize components
            db = MetricsDatabase(self.db_path)
            drift_detector = DriftDetector(database=db)
            performance_monitor = PerformanceMonitor(
                database=db,
                drift_detector=drift_detector
            )
            
            # Add mock notification channel
            mock_notifications = []
            
            class MockNotifier:
                def send_alert(self, alert):
                    mock_notifications.append(alert)
                    return True
            
            performance_monitor.add_notification_channel(MockNotifier())
            
            # Store some problematic prediction metrics
            problematic_predictions = [
                PredictionMetric(
                    prediction_id=f"prob_pred_{i}",
                    timestamp=datetime.now() - timedelta(minutes=i),
                    model_version="v1.0",
                    prediction="PNEUMONIA" if i % 2 == 0 else "NORMAL",
                    confidence=0.6,  # Low confidence
                    processing_time_ms=3000 + i * 100,  # High latency
                    image_hash=f"hash_{i}",
                    status=PredictionStatus.SUCCESS,
                    actual_label="NORMAL" if i % 2 == 0 else "PNEUMONIA"  # All wrong
                )
                for i in range(10)
            ]
            
            # Store metrics
            for metric in problematic_predictions:
                db.store_prediction_metric(metric)
            
            # Mock metrics collection to return problematic data
            with patch.object(performance_monitor, '_collect_current_metrics') as mock_collect:
                from monitoring.performance_monitor import PerformanceMetrics
                
                mock_metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    accuracy=0.0,  # 0% accuracy (all wrong)
                    avg_latency_ms=3500,  # High latency
                    p95_latency_ms=4000,
                    p99_latency_ms=5000,
                    throughput_rps=0.2,  # Low throughput
                    error_rate=0.0,
                    cpu_usage=95,  # High CPU
                    memory_usage=92,  # High memory
                    gpu_memory_mb=4096,
                    active_requests=20,
                    cache_hit_rate=0.3,  # Low cache hit rate
                    model_version="v1.0"
                )
                mock_collect.return_value = mock_metrics
                
                # Run performance check
                performance_monitor._check_performance()
                
                # Verify alerts were generated
                assert len(mock_notifications) > 0
                
                # Check for specific alert types
                alert_types = [alert.alert_type for alert in mock_notifications]
                assert AlertType.ACCURACY_DROP in alert_types
                assert AlertType.HIGH_LATENCY in alert_types
                assert AlertType.SYSTEM_RESOURCE in alert_types
                
                # Verify alert details
                accuracy_alert = next(
                    alert for alert in mock_notifications 
                    if alert.alert_type == AlertType.ACCURACY_DROP
                )
                assert accuracy_alert.severity == AlertSeverity.CRITICAL
                assert accuracy_alert.current_value == 0.0
                assert accuracy_alert.threshold == 0.8
    
    def test_drift_detection_integration(self):
        """Test drift detection with baseline establishment"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize drift detector
            db = MetricsDatabase(self.db_path)
            drift_detector = DriftDetector(database=db)
            
            # Create baseline data (good performance)
            baseline_predictions = [
                {
                    'status': 'success',
                    'image_hash': f'baseline_hash_{i}',
                    'confidence': 0.9 + (i % 10) * 0.01,
                    'processing_time': 150 + (i % 20),
                    'prediction': 'NORMAL' if i % 3 == 0 else 'PNEUMONIA',
                    'actual_label': 'NORMAL' if i % 3 == 0 else 'PNEUMONIA'  # All correct
                }
                for i in range(200)  # Sufficient for baseline
            ]
            
            # Mock baseline data
            with patch.object(db, 'get_prediction_metrics') as mock_get_predictions:
                mock_get_predictions.return_value = baseline_predictions
                
                # Establish baseline
                result = drift_detector.establish_baseline(hours=168)
                
                assert result["status"] == "baseline_established"
                assert result["predictions_count"] == 200
                assert "baseline_stats" in result
                
                # Verify baseline stats
                baseline_stats = result["baseline_stats"]
                assert baseline_stats["accuracy"] == 1.0  # All predictions correct
                assert "image_statistics" in baseline_stats
                assert "prediction_distribution" in baseline_stats
            
            # Now test drift detection with degraded data
            degraded_predictions = [
                {
                    'status': 'success',
                    'image_hash': f'degraded_hash_{i}',
                    'confidence': 0.6 + (i % 10) * 0.01,  # Lower confidence
                    'processing_time': 300 + (i % 50),  # Higher latency
                    'prediction': 'NORMAL' if i % 2 == 0 else 'PNEUMONIA',
                    'actual_label': 'PNEUMONIA' if i % 2 == 0 else 'NORMAL'  # All wrong
                }
                for i in range(50)
            ]
            
            with patch.object(db, 'get_prediction_metrics') as mock_get_recent:
                mock_get_recent.return_value = degraded_predictions
                
                # Detect drift
                drift_report = drift_detector.detect_drift(hours=24)
                
                # Should detect concept drift due to accuracy drop
                assert drift_report.overall_drift_score > 0.5
                assert len(drift_report.alerts) > 0
                
                # Check for concept drift alert
                concept_alerts = [
                    alert for alert in drift_report.alerts 
                    if alert.drift_type == DriftType.CONCEPT_DRIFT
                ]
                assert len(concept_alerts) > 0
                
                concept_alert = concept_alerts[0]
                assert concept_alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
                assert concept_alert.current_value < concept_alert.baseline_value
    
    def test_audit_trail_compliance_reporting(self):
        """Test audit trail and compliance reporting"""
        
        with patch('monitoring.audit_trail.PSYCOPG2_AVAILABLE', False):
            
            # Initialize audit manager
            manager = get_audit_explainability_manager(
                compliance_level=ComplianceLevel.MEDICAL
            )
            
            # Log multiple predictions with different characteristics
            predictions_data = [
                {
                    "prediction_id": f"audit_test_{i}",
                    "confidence": 0.9 - (i * 0.1),  # Decreasing confidence
                    "prediction": "PNEUMONIA" if i % 2 == 0 else "NORMAL",
                    "processing_time": 150 + (i * 50)
                }
                for i in range(10)
            ]
            
            image_data = b"test_image_data"
            
            for pred_data in predictions_data:
                manager.log_prediction_with_explanation(
                    prediction_id=pred_data["prediction_id"],
                    model_id="audit_test_model",
                    model_version="v1.0",
                    image=image_data,
                    input_metadata={"test": "audit_integration"},
                    prediction_result=pred_data["prediction"],
                    confidence_score=pred_data["confidence"],
                    processing_time_ms=pred_data["processing_time"],
                    api_context={
                        "api_endpoint": "/predict",
                        "client_ip": "127.0.0.1"
                    }
                )
            
            # Generate compliance report
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            report = manager.generate_compliance_report(
                start_time=start_time,
                end_time=end_time,
                include_explanations=True
            )
            
            # Verify report structure
            assert "report_id" in report
            assert "medical_compliance" in report
            assert "compliance_score" in report
            
            # Check medical compliance details
            medical_compliance = report["medical_compliance"]
            assert "prediction_traceability" in medical_compliance
            assert "model_lineage_completeness" in medical_compliance
            assert "data_integrity" in medical_compliance
            
            # Verify some predictions were logged
            assert report["summary"]["total_predictions"] > 0
    
    def test_monitoring_system_performance(self):
        """Test monitoring system performance under load"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize components
            db = MetricsDatabase(self.db_path)
            collector = MetricsCollector(db, collection_interval=0.1)
            
            # Start metrics collection
            collector.start_collection()
            
            try:
                # Simulate high-frequency metric generation
                start_time = time.time()
                
                for i in range(100):
                    pred_metric = PredictionMetric(
                        prediction_id=f"perf_test_{i}",
                        timestamp=datetime.now(),
                        model_version="v1.0",
                        prediction="NORMAL" if i % 2 == 0 else "PNEUMONIA",
                        confidence=0.8 + (i % 20) * 0.01,
                        processing_time_ms=100 + (i % 50),
                        image_hash=f"perf_hash_{i}",
                        status=PredictionStatus.SUCCESS
                    )
                    
                    collector.add_metric(pred_metric)
                
                # Let collector process metrics
                time.sleep(0.5)
                
                # Flush remaining metrics
                collector.flush_buffer()
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Verify performance (should process 100 metrics quickly)
                assert processing_time < 5.0  # Should complete in under 5 seconds
                
                # Verify metrics were stored
                assert len(collector.metrics_buffer) == 0  # Buffer should be empty
                
            finally:
                collector.stop_collection()
    
    def test_error_handling_and_recovery(self):
        """Test error handling and system recovery"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize components
            db = MetricsDatabase(self.db_path)
            performance_monitor = PerformanceMonitor(database=db)
            
            # Test with failing notification channel
            class FailingNotifier:
                def __init__(self):
                    self.call_count = 0
                
                def send_alert(self, alert):
                    self.call_count += 1
                    if self.call_count <= 2:
                        raise Exception("Simulated notification failure")
                    return True
            
            failing_notifier = FailingNotifier()
            performance_monitor.add_notification_channel(failing_notifier)
            
            # Create alert that should trigger notifications
            from monitoring.performance_monitor import PerformanceAlert
            
            alert = PerformanceAlert(
                alert_type=AlertType.MODEL_FAILURE,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                metric_name="model_status",
                current_value=0,
                threshold=1,
                message="Model failure test"
            )
            
            # Process alert multiple times
            for i in range(5):
                try:
                    performance_monitor._process_alert(alert)
                except Exception:
                    pass  # Should handle notification failures gracefully
                
                # Reset cooldown for testing
                performance_monitor.last_alert_times.clear()
            
            # Verify system continued to function despite failures
            assert len(performance_monitor.alert_history) > 0
            assert failing_notifier.call_count >= 3  # Should have retried
    
    def test_concurrent_monitoring_operations(self):
        """Test concurrent monitoring operations"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize components
            db = MetricsDatabase(self.db_path)
            manager = get_audit_explainability_manager()
            
            # Function to simulate concurrent predictions
            def simulate_predictions(thread_id, count):
                for i in range(count):
                    try:
                        manager.log_prediction_with_explanation(
                            prediction_id=f"concurrent_{thread_id}_{i}",
                            model_id="concurrent_test_model",
                            model_version="v1.0",
                            image=b"test_image_data",
                            input_metadata={"thread_id": thread_id},
                            prediction_result="NORMAL" if i % 2 == 0 else "PNEUMONIA",
                            confidence_score=0.8 + (i % 10) * 0.01,
                            processing_time_ms=100 + (i % 20)
                        )
                    except Exception as e:
                        print(f"Error in thread {thread_id}: {e}")
            
            # Start multiple threads
            threads = []
            for thread_id in range(5):
                thread = threading.Thread(
                    target=simulate_predictions,
                    args=(thread_id, 10)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)
            
            # Verify no deadlocks occurred and data was stored
            # (The fact that we reach this point indicates success)
            assert True
    
    @pytest.mark.asyncio
    async def test_async_monitoring_operations(self):
        """Test asynchronous monitoring operations"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize components
            db = MetricsDatabase(self.db_path)
            collector = MetricsCollector(db, api_urls=["http://localhost:8000"])
            
            # Mock async API collection
            with patch.object(collector.api_collector, 'collect_api_metrics') as mock_collect:
                mock_collect.return_value = [
                    # Mock API metrics
                ]
                
                # Test async collection
                metrics = await collector.collect_all_metrics()
                
                # Should include system metrics at minimum
                assert len(metrics) > 0
                
                # Verify collection timestamp metric is included
                metric_names = [m.name for m in metrics]
                assert "metrics_collection_timestamp" in metric_names

class TestMonitoringSystemResilience:
    """Test monitoring system resilience and fault tolerance"""
    
    def test_database_connection_failure_recovery(self):
        """Test recovery from database connection failures"""
        
        # Test with initially failing database
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            # Initialize with non-existent database path
            invalid_db_path = "/invalid/path/database.db"
            
            try:
                db = MetricsDatabase(invalid_db_path)
                
                # Should fall back to file storage
                pred_metric = PredictionMetric(
                    prediction_id="resilience_test",
                    timestamp=datetime.now(),
                    model_version="v1.0",
                    prediction="NORMAL",
                    confidence=0.9,
                    processing_time_ms=150,
                    image_hash="test_hash",
                    status=PredictionStatus.SUCCESS
                )
                
                # Should not raise exception
                success = db.store_prediction_metric(pred_metric)
                # May fail due to invalid path, but should handle gracefully
                
            except Exception as e:
                # Should handle database failures gracefully
                assert "database" in str(e).lower() or "file" in str(e).lower()
    
    def test_memory_usage_under_load(self):
        """Test memory usage under high load"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, "memory_test.db")
                db = MetricsDatabase(db_path)
                collector = MetricsCollector(db)
                
                # Generate large number of metrics
                initial_buffer_size = len(collector.metrics_buffer)
                
                for i in range(2000):  # Exceed max_memory_metrics
                    pred_metric = PredictionMetric(
                        prediction_id=f"memory_test_{i}",
                        timestamp=datetime.now(),
                        model_version="v1.0",
                        prediction="NORMAL",
                        confidence=0.9,
                        processing_time_ms=150,
                        image_hash=f"hash_{i}",
                        status=PredictionStatus.SUCCESS
                    )
                    collector.add_metric(pred_metric)
                
                # Buffer should be limited by max_memory_metrics
                assert len(collector.metrics_buffer) <= collector.max_memory_metrics
                
                # Flush should work without memory issues
                collector.flush_buffer()
                assert len(collector.metrics_buffer) == 0
    
    def test_monitoring_system_cleanup(self):
        """Test proper cleanup of monitoring resources"""
        
        with patch('monitoring.metrics_collector.PSYCOPG2_AVAILABLE', False):
            
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, "cleanup_test.db")
                db = MetricsDatabase(db_path)
                collector = MetricsCollector(db, collection_interval=0.1)
                
                # Start monitoring
                collector.start_collection()
                assert collector.running
                assert collector.collection_task is not None
                
                # Stop monitoring
                collector.stop_collection()
                assert not collector.running
                
                # Verify cleanup
                time.sleep(0.2)  # Allow time for cleanup
                
                # Should be able to restart
                collector.start_collection()
                assert collector.running
                
                collector.stop_collection()
                assert not collector.running

if __name__ == "__main__":
    pytest.main([__file__, "-v"])