"""
Prometheus Metrics Exporter for Chest X-Ray Pneumonia Detection System
Exports metrics in Prometheus format for Grafana dashboards
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from threading import Thread
import os

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info, CollectorRegistry, REGISTRY
    from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def inc(self, amount=1): pass
    
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, labels): pass

try:
    from .metrics_collector import get_metrics_collector
    from .performance_monitor import get_performance_monitor
    from .drift_detector import get_drift_detector
except ImportError:
    from metrics_collector import get_metrics_collector
    from performance_monitor import get_performance_monitor
    from drift_detector import get_drift_detector

logger = logging.getLogger(__name__)

class ChestXrayMetricsCollector:
    """
    Custom Prometheus collector for chest X-ray metrics
    """
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.performance_monitor = get_performance_monitor()
        self.drift_detector = get_drift_detector()
    
    def collect(self):
        """Collect metrics for Prometheus"""
        try:
            # Get recent metrics
            summary = self.metrics_collector.get_prediction_summary(hours=1)
            performance_summary = self.performance_monitor.get_performance_summary(hours=1)
            
            # Model accuracy
            if summary.get('accuracy') is not None:
                accuracy_metric = GaugeMetricFamily(
                    'chest_xray_model_accuracy',
                    'Current model accuracy',
                    labels=['model_version']
                )
                # Get model version from recent predictions
                recent_preds = self.metrics_collector.get_recent_predictions(limit=1)
                model_version = recent_preds[0].get('model_version', 'unknown') if recent_preds else 'unknown'
                accuracy_metric.add_metric([model_version], summary['accuracy'])
                yield accuracy_metric
            
            # Prediction counts
            predictions_total = CounterMetricFamily(
                'chest_xray_predictions_total',
                'Total number of predictions',
                labels=['prediction_class', 'status']
            )
            
            # Add successful predictions by class
            for class_name, count in summary.get('predictions_by_class', {}).items():
                predictions_total.add_metric([class_name, 'success'], count)
            
            # Add error count
            error_count = summary.get('error_count', 0)
            predictions_total.add_metric(['unknown', 'error'], error_count)
            
            yield predictions_total
            
            # Response time metrics
            if performance_summary.get('current_metrics'):
                current_metrics = performance_summary['current_metrics']
                
                response_time_gauge = GaugeMetricFamily(
                    'chest_xray_response_time_seconds',
                    'Response time in seconds',
                    labels=['percentile']
                )
                
                avg_latency = current_metrics.get('avg_latency_ms', 0) / 1000.0
                p95_latency = current_metrics.get('p95_latency_ms', 0) / 1000.0
                p99_latency = current_metrics.get('p99_latency_ms', 0) / 1000.0
                
                response_time_gauge.add_metric(['avg'], avg_latency)
                response_time_gauge.add_metric(['p95'], p95_latency)
                response_time_gauge.add_metric(['p99'], p99_latency)
                
                yield response_time_gauge
                
                # Throughput
                throughput_gauge = GaugeMetricFamily(
                    'chest_xray_throughput_rps',
                    'Requests per second'
                )
                throughput_gauge.add_metric([], current_metrics.get('throughput_rps', 0))
                yield throughput_gauge
                
                # Error rate
                error_rate_gauge = GaugeMetricFamily(
                    'chest_xray_error_rate',
                    'Error rate as a fraction'
                )
                error_rate_gauge.add_metric([], current_metrics.get('error_rate', 0))
                yield error_rate_gauge
                
                # System resources
                cpu_gauge = GaugeMetricFamily(
                    'chest_xray_cpu_usage_percent',
                    'CPU usage percentage'
                )
                cpu_gauge.add_metric([], current_metrics.get('cpu_usage', 0))
                yield cpu_gauge
                
                memory_gauge = GaugeMetricFamily(
                    'chest_xray_memory_usage_percent',
                    'Memory usage percentage'
                )
                memory_gauge.add_metric([], current_metrics.get('memory_usage', 0))
                yield memory_gauge
                
                if current_metrics.get('gpu_memory_mb') is not None:
                    gpu_memory_gauge = GaugeMetricFamily(
                        'chest_xray_gpu_memory_mb',
                        'GPU memory usage in MB'
                    )
                    gpu_memory_gauge.add_metric([], current_metrics['gpu_memory_mb'])
                    yield gpu_memory_gauge
            
            # Active alerts
            active_alerts = self.performance_monitor.get_active_alerts()
            alerts_gauge = GaugeMetricFamily(
                'chest_xray_active_alerts',
                'Number of active alerts',
                labels=['alert_type', 'severity']
            )
            
            # Count alerts by type and severity
            alert_counts = {}
            for alert in active_alerts:
                key = (alert.alert_type.value, alert.severity.value)
                alert_counts[key] = alert_counts.get(key, 0) + 1
            
            for (alert_type, severity), count in alert_counts.items():
                alerts_gauge.add_metric([alert_type, severity], count)
            
            # If no alerts, add a zero metric
            if not alert_counts:
                alerts_gauge.add_metric(['none', 'none'], 0)
            
            yield alerts_gauge
            
            # Drift score (if available)
            try:
                drift_report = self.drift_detector.detect_drift(hours=24)
                drift_gauge = GaugeMetricFamily(
                    'chest_xray_drift_score',
                    'Overall drift detection score',
                    labels=['drift_type']
                )
                
                drift_gauge.add_metric(['overall'], drift_report.overall_drift_score)
                drift_gauge.add_metric(['data'], drift_report.data_drift_score)
                drift_gauge.add_metric(['concept'], drift_report.concept_drift_score)
                drift_gauge.add_metric(['prediction'], drift_report.prediction_drift_score)
                
                yield drift_gauge
                
            except Exception as e:
                logger.debug(f"Could not collect drift metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting Prometheus metrics: {e}")

class PrometheusExporter:
    """
    Prometheus metrics exporter service
    """
    
    def __init__(self, port: int = 8080, update_interval: int = 30):
        self.port = port
        self.update_interval = update_interval
        self.server_thread = None
        self.running = False
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics export disabled")
            return
        
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # Register custom collector
        self.custom_collector = ChestXrayMetricsCollector()
        self.registry.register(self.custom_collector)
        
        # Standard metrics
        self.setup_standard_metrics()
        
        logger.info(f"PrometheusExporter initialized on port {port}")
    
    def setup_standard_metrics(self):
        """Setup standard Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Model performance metrics
        self.accuracy_gauge = Gauge(
            'chest_xray_model_accuracy_current',
            'Current model accuracy',
            ['model_version'],
            registry=self.registry
        )
        
        self.prediction_counter = Counter(
            'chest_xray_predictions_processed_total',
            'Total processed predictions',
            ['prediction_class', 'model_version'],
            registry=self.registry
        )
        
        self.response_time_histogram = Histogram(
            'chest_xray_response_time_seconds_histogram',
            'Response time histogram',
            ['model_version'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'chest_xray_errors_total',
            'Total errors',
            ['error_type', 'model_version'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_gauge = Gauge(
            'chest_xray_system_cpu_percent',
            'System CPU usage',
            registry=self.registry
        )
        
        self.memory_gauge = Gauge(
            'chest_xray_system_memory_percent',
            'System memory usage',
            registry=self.registry
        )
        
        self.gpu_memory_gauge = Gauge(
            'chest_xray_system_gpu_memory_mb',
            'GPU memory usage in MB',
            registry=self.registry
        )
        
        # Model info
        self.model_info = Info(
            'chest_xray_model_info',
            'Model information',
            registry=self.registry
        )
    
    def start(self):
        """Start Prometheus metrics server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start Prometheus exporter - prometheus_client not available")
            return
        
        if self.running:
            logger.warning("Prometheus exporter already running")
            return
        
        try:
            # Start HTTP server for metrics
            start_http_server(self.port, registry=self.registry)
            self.running = True
            
            # Start metrics update thread
            self.server_thread = Thread(target=self._update_metrics_loop, daemon=True)
            self.server_thread.start()
            
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus exporter: {e}")
            self.running = False
    
    def stop(self):
        """Stop Prometheus metrics server"""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        logger.info("Prometheus exporter stopped")
    
    def _update_metrics_loop(self):
        """Update metrics periodically"""
        while self.running:
            try:
                self._update_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
                time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update Prometheus metrics with current values"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Get metrics from collectors
            metrics_collector = get_metrics_collector()
            performance_monitor = get_performance_monitor()
            
            # Update model accuracy
            summary = metrics_collector.get_prediction_summary(hours=1)
            if summary.get('accuracy') is not None:
                recent_preds = metrics_collector.get_recent_predictions(limit=1)
                model_version = recent_preds[0].get('model_version', 'unknown') if recent_preds else 'unknown'
                self.accuracy_gauge.labels(model_version=model_version).set(summary['accuracy'])
            
            # Update system metrics
            performance_summary = performance_monitor.get_performance_summary(hours=1)
            if performance_summary.get('current_metrics'):
                current_metrics = performance_summary['current_metrics']
                
                self.cpu_gauge.set(current_metrics.get('cpu_usage', 0))
                self.memory_gauge.set(current_metrics.get('memory_usage', 0))
                
                if current_metrics.get('gpu_memory_mb') is not None:
                    self.gpu_memory_gauge.set(current_metrics['gpu_memory_mb'])
            
            # Update model info
            recent_preds = metrics_collector.get_recent_predictions(limit=1)
            if recent_preds:
                model_version = recent_preds[0].get('model_version', 'unknown')
                self.model_info.info({
                    'version': model_version,
                    'architecture': 'CNN',  # Could be made dynamic
                    'task': 'pneumonia_detection'
                })
            
        except Exception as e:
            logger.error(f"Error updating standard metrics: {e}")
    
    def record_prediction(self, prediction: str, model_version: str, 
                         processing_time_seconds: float, success: bool = True):
        """Record a prediction for metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Update prediction counter
            self.prediction_counter.labels(
                prediction_class=prediction,
                model_version=model_version
            ).inc()
            
            # Update response time histogram
            self.response_time_histogram.labels(
                model_version=model_version
            ).observe(processing_time_seconds)
            
            # Update error counter if failed
            if not success:
                self.error_counter.labels(
                    error_type='prediction_error',
                    model_version=model_version
                ).inc()
                
        except Exception as e:
            logger.error(f"Error recording prediction metrics: {e}")

# Global exporter instance
_prometheus_exporter = None

def get_prometheus_exporter() -> PrometheusExporter:
    """Get global Prometheus exporter instance"""
    global _prometheus_exporter
    if _prometheus_exporter is None:
        port = int(os.getenv('PROMETHEUS_PORT', 8080))
        _prometheus_exporter = PrometheusExporter(port=port)
    return _prometheus_exporter

def start_prometheus_exporter(port: int = 8080):
    """Start Prometheus metrics exporter"""
    exporter = PrometheusExporter(port=port)
    exporter.start()
    return exporter

if __name__ == "__main__":
    # Test Prometheus exporter
    print("Testing Prometheus Exporter...")
    
    if not PROMETHEUS_AVAILABLE:
        print("❌ Prometheus client not available")
        exit(1)
    
    exporter = PrometheusExporter(port=8080)
    
    # Test metrics recording
    exporter.record_prediction("PNEUMONIA", "v1.0.0", 0.045, success=True)
    exporter.record_prediction("NORMAL", "v1.0.0", 0.032, success=True)
    
    print("✅ Test predictions recorded")
    
    # Start exporter
    exporter.start()
    
    print(f"✅ Prometheus exporter started on port 8080")
    print("   Metrics available at http://localhost:8080/metrics")
    print("   Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exporter.stop()
        print("\n👋 Prometheus exporter stopped")