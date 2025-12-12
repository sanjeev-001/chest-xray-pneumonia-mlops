"""
Monitoring Service Main Module
Tracks model performance and system metrics with drift detection and alerting
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .metrics_collector import get_metrics_collector, PredictionStatus
from .drift_detector import get_drift_detector
from .performance_monitor import get_performance_monitor, setup_email_notifications, setup_slack_notifications
from .prometheus_exporter import get_prometheus_exporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Monitoring Service",
    description="Tracks model performance, system metrics, drift detection, and alerting for chest X-ray pneumonia detection",
    version="0.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class PredictionRequest(BaseModel):
    prediction: str
    confidence: float
    processing_time_ms: float
    model_version: str
    image_id: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    actual_label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ModelRegistration(BaseModel):
    name: str
    version: str
    architecture: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str = "active"
    model_path: Optional[str] = None

class ExperimentRegistration(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str = "running"
    model_name: Optional[str] = None
    model_version: Optional[str] = None

class DriftDetectionRequest(BaseModel):
    hours: int = 24

class BaselineRequest(BaseModel):
    hours: int = 168  # 1 week default

class AlertConfigRequest(BaseModel):
    email_config: Optional[Dict[str, Any]] = None
    slack_webhook: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "monitoring"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    collector = get_metrics_collector()
    # Check if database is available
    db_status = "connected" if collector.database.connection else "disconnected"
    return {
        "status": "ready", 
        "service": "monitoring",
        "database": db_status
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Monitoring Service", "version": "0.1.0"}

@app.post("/predictions")
async def record_prediction(request: PredictionRequest):
    """Record a prediction metric"""
    try:
        collector = get_metrics_collector()
        
        # Convert string status to enum
        status = PredictionStatus.SUCCESS
        if request.status.lower() == "error":
            status = PredictionStatus.ERROR
        elif request.status.lower() == "timeout":
            status = PredictionStatus.TIMEOUT
        
        prediction_id = collector.record_prediction(
            prediction=request.prediction,
            confidence=request.confidence,
            processing_time_ms=request.processing_time_ms,
            model_version=request.model_version,
            image_id=request.image_id,
            status=status,
            error_message=request.error_message,
            actual_label=request.actual_label,
            metadata=request.metadata
        )
        
        return {"prediction_id": prediction_id, "status": "recorded"}
        
    except Exception as e:
        logger.error(f"Failed to record prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
async def get_predictions(
    limit: int = Query(100, ge=1, le=1000),
    hours: int = Query(24, ge=1, le=168)
):
    """Get recent predictions"""
    try:
        collector = get_metrics_collector()
        predictions = collector.get_recent_predictions(limit=limit)
        
        # Filter by time if needed
        if hours < 168:  # Only filter if less than a week
            cutoff_time = datetime.now() - timedelta(hours=hours)
            predictions = [
                p for p in predictions 
                if datetime.fromisoformat(p['timestamp']) >= cutoff_time
            ]
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "limit": limit,
            "hours": hours
        }
        
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/summary")
async def get_prediction_summary(hours: int = Query(24, ge=1, le=168)):
    """Get prediction summary statistics"""
    try:
        collector = get_metrics_collector()
        summary = collector.get_prediction_summary(hours=hours)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get prediction summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/accuracy")
async def get_accuracy_metrics(hours: int = Query(24, ge=1, le=168)):
    """Get accuracy metrics when actual labels are available"""
    try:
        collector = get_metrics_collector()
        accuracy = collector.get_accuracy_metrics(hours=hours)
        return accuracy
        
    except Exception as e:
        logger.error(f"Failed to get accuracy metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/metrics")
async def get_system_metrics(
    limit: int = Query(100, ge=1, le=1000),
    hours: int = Query(24, ge=1, le=168)
):
    """Get system performance metrics"""
    try:
        collector = get_metrics_collector()
        metrics = collector.get_recent_system_metrics(limit=limit)
        
        # Filter by time if needed
        if hours < 168:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = [
                m for m in metrics 
                if datetime.fromisoformat(m['timestamp']) >= cutoff_time
            ]
        
        return {
            "metrics": metrics,
            "count": len(metrics),
            "limit": limit,
            "hours": hours
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/trends")
async def get_performance_trends(hours: int = Query(24, ge=1, le=168)):
    """Get performance trends over time"""
    try:
        collector = get_metrics_collector()
        trends = collector.get_performance_trends(hours=hours)
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models")
async def register_model(request: ModelRegistration):
    """Register a new model"""
    try:
        collector = get_metrics_collector()
        model_id = collector.register_model(
            name=request.name,
            version=request.version,
            architecture=request.architecture,
            metrics=request.metrics,
            status=request.status,
            model_path=request.model_path
        )
        
        return {"model_id": model_id, "status": "registered"}
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models(limit: int = Query(100, ge=1, le=1000)):
    """Get registered models"""
    try:
        collector = get_metrics_collector()
        models = collector.database.get_models(limit=limit)
        
        return {
            "models": models,
            "count": len(models),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments")
async def register_experiment(request: ExperimentRegistration):
    """Register a new experiment"""
    try:
        collector = get_metrics_collector()
        experiment_id = collector.register_experiment(
            name=request.name,
            parameters=request.parameters,
            metrics=request.metrics,
            status=request.status,
            model_name=request.model_name,
            model_version=request.model_version
        )
        
        return {"experiment_id": experiment_id, "status": "registered"}
        
    except Exception as e:
        logger.error(f"Failed to register experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
async def get_experiments(
    model_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get experiments, optionally filtered by model"""
    try:
        collector = get_metrics_collector()
        experiments = collector.database.get_experiments(model_id=model_id, limit=limit)
        
        return {
            "experiments": experiments,
            "count": len(experiments),
            "limit": limit,
            "model_id": model_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/experiments/{experiment_id}/metrics")
async def update_experiment_metrics(
    experiment_id: str,
    metrics: Dict[str, Any],
    status: Optional[str] = None
):
    """Update experiment metrics"""
    try:
        collector = get_metrics_collector()
        success = collector.update_experiment_metrics(
            experiment_id=experiment_id,
            metrics=metrics,
            status=status
        )
        
        if success:
            return {"status": "updated"}
        else:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
    except Exception as e:
        logger.error(f"Failed to update experiment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/metrics/cleanup")
async def cleanup_old_metrics(days: int = Query(30, ge=1, le=365)):
    """Clean up old metrics from database"""
    try:
        collector = get_metrics_collector()
        collector.cleanup_old_metrics(days=days)
        
        return {"status": "cleanup_completed", "days": days}
        
    except Exception as e:
        logger.error(f"Failed to cleanup metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Drift Detection Endpoints

@app.post("/drift/baseline")
async def establish_baseline(request: BaselineRequest):
    """Establish baseline statistics for drift detection"""
    try:
        drift_detector = get_drift_detector()
        result = drift_detector.establish_baseline(hours=request.hours)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to establish baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/detect")
async def detect_drift(request: DriftDetectionRequest):
    """Run drift detection analysis"""
    try:
        drift_detector = get_drift_detector()
        report = drift_detector.detect_drift(hours=request.hours)
        
        return {
            "timestamp": report.timestamp.isoformat(),
            "time_window_hours": report.time_window_hours,
            "alerts_count": len(report.alerts),
            "data_drift_score": report.data_drift_score,
            "concept_drift_score": report.concept_drift_score,
            "prediction_drift_score": report.prediction_drift_score,
            "overall_drift_score": report.overall_drift_score,
            "alerts": [
                {
                    "drift_type": alert.drift_type.value,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "baseline_value": alert.baseline_value,
                    "threshold": alert.threshold,
                    "description": alert.description,
                    "metadata": alert.metadata
                }
                for alert in report.alerts
            ],
            "recommendations": report.recommendations
        }
        
    except Exception as e:
        logger.error(f"Failed to detect drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/history")
async def get_drift_history(days: int = Query(30, ge=1, le=365)):
    """Get drift detection history"""
    try:
        drift_detector = get_drift_detector()
        history = drift_detector.get_drift_history(days=days)
        
        return {
            "history": history,
            "days": days
        }
        
    except Exception as e:
        logger.error(f"Failed to get drift history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance Monitoring Endpoints

@app.get("/performance/summary")
async def get_performance_summary(hours: int = Query(24, ge=1, le=168)):
    """Get comprehensive performance summary"""
    try:
        performance_monitor = get_performance_monitor()
        summary = performance_monitor.get_performance_summary(hours=hours)
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/alerts")
async def get_active_alerts():
    """Get currently active performance alerts"""
    try:
        performance_monitor = get_performance_monitor()
        alerts = performance_monitor.get_active_alerts()
        
        return {
            "active_alerts": [
                {
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "metadata": alert.metadata,
                    "resolved": alert.resolved
                }
                for alert in alerts
            ],
            "count": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/alerts/history")
async def get_alert_history(hours: int = Query(24, ge=1, le=168)):
    """Get alert history for specified time period"""
    try:
        performance_monitor = get_performance_monitor()
        alerts = performance_monitor.get_alert_history(hours=hours)
        
        return {
            "alert_history": [
                {
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                for alert in alerts
            ],
            "count": len(alerts),
            "hours": hours
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/alerts/{alert_key}/resolve")
async def resolve_alert(alert_key: str):
    """Manually resolve a performance alert"""
    try:
        performance_monitor = get_performance_monitor()
        success = performance_monitor.resolve_alert(alert_key)
        
        if success:
            return {"status": "resolved", "alert_key": alert_key}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/monitoring/start")
async def start_performance_monitoring():
    """Start continuous performance monitoring"""
    try:
        performance_monitor = get_performance_monitor()
        performance_monitor.start_monitoring()
        
        return {"status": "monitoring_started"}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/monitoring/stop")
async def stop_performance_monitoring():
    """Stop continuous performance monitoring"""
    try:
        performance_monitor = get_performance_monitor()
        performance_monitor.stop_monitoring()
        
        return {"status": "monitoring_stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/configure")
async def configure_alerts(request: AlertConfigRequest):
    """Configure alert notification channels"""
    try:
        if request.email_config:
            setup_email_notifications(
                smtp_host=request.email_config.get('smtp_host'),
                smtp_port=request.email_config.get('smtp_port', 587),
                username=request.email_config.get('username'),
                password=request.email_config.get('password'),
                from_email=request.email_config.get('from_email'),
                to_emails=request.email_config.get('to_emails', [])
            )
        
        if request.slack_webhook:
            setup_slack_notifications(request.slack_webhook)
        
        return {"status": "alerts_configured"}
        
    except Exception as e:
        logger.error(f"Failed to configure alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prometheus Metrics Endpoint

@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    try:
        prometheus_exporter = get_prometheus_exporter()
        
        return {
            "status": "prometheus_exporter_available",
            "port": prometheus_exporter.port,
            "metrics_url": f"http://localhost:{prometheus_exporter.port}/metrics"
        }
        
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics/prometheus/start")
async def start_prometheus_exporter():
    """Start Prometheus metrics exporter"""
    try:
        prometheus_exporter = get_prometheus_exporter()
        prometheus_exporter.start()
        
        return {
            "status": "prometheus_started",
            "port": prometheus_exporter.port,
            "metrics_url": f"http://localhost:{prometheus_exporter.port}/metrics"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Prometheus exporter: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize monitoring components on startup"""
    logger.info("Starting monitoring service...")
    
    # Start system monitoring
    collector = get_metrics_collector()
    collector.start_system_monitoring()
    
    # Start performance monitoring if enabled
    if os.getenv("AUTO_START_PERFORMANCE_MONITORING", "false").lower() == "true":
        performance_monitor = get_performance_monitor()
        performance_monitor.start_monitoring()
        logger.info("Performance monitoring started automatically")
    
    # Start Prometheus exporter if enabled
    if os.getenv("AUTO_START_PROMETHEUS", "false").lower() == "true":
        prometheus_exporter = get_prometheus_exporter()
        prometheus_exporter.start()
        logger.info("Prometheus exporter started automatically")
    
    logger.info("Monitoring service startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up monitoring components on shutdown"""
    logger.info("Shutting down monitoring service...")
    
    # Stop system monitoring
    collector = get_metrics_collector()
    collector.stop_system_monitoring()
    
    # Stop performance monitoring
    performance_monitor = get_performance_monitor()
    performance_monitor.stop_monitoring()
    
    # Stop Prometheus exporter
    prometheus_exporter = get_prometheus_exporter()
    prometheus_exporter.stop()
    
    logger.info("Monitoring service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)