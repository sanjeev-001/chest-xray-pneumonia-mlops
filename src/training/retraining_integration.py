"""
Retraining Integration Service
Connects monitoring alerts with automated retraining workflows
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import time

try:
    from .retraining_orchestrator import (
        RetrainingOrchestrator, RetrainingTrigger, 
        trigger_performance_retraining, trigger_drift_retraining,
        get_retraining_orchestrator
    )
    from .trainer import ModelTrainer
    from .experiment_tracker import ExperimentTracker
    from .model_registry import ModelRegistry
except ImportError:
    from retraining_orchestrator import (
        RetrainingOrchestrator, RetrainingTrigger,
        trigger_performance_retraining, trigger_drift_retraining,
        get_retraining_orchestrator
    )
    from trainer import ModelTrainer
    from experiment_tracker import ExperimentTracker
    from model_registry import ModelRegistry

try:
    from monitoring.performance_monitor import PerformanceAlert, AlertType, AlertSeverity
    from monitoring.drift_detector import DriftReport, DriftSeverity
except ImportError:
    # Mock classes for testing
    class PerformanceAlert:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AlertType:
        ACCURACY_DROP = "accuracy_drop"
        HIGH_LATENCY = "high_latency"
        DRIFT_DETECTED = "drift_detected"
    
    class AlertSeverity:
        CRITICAL = "critical"
        HIGH = "high"
        WARNING = "warning"
    
    class DriftReport:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class DriftSeverity:
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"

logger = logging.getLogger(__name__)

@dataclass
class RetrainingPolicy:
    """Policy for automated retraining decisions"""
    
    # Performance thresholds
    accuracy_threshold: float = 0.80
    f1_threshold: float = 0.75
    latency_threshold_ms: float = 2000
    error_rate_threshold: float = 0.05
    
    # Drift thresholds
    data_drift_threshold: float = 0.3
    concept_drift_threshold: float = 0.2
    
    # Retraining constraints
    min_time_between_retraining_hours: int = 24
    max_retraining_per_day: int = 2
    cooldown_after_failed_retraining_hours: int = 6
    
    # Alert requirements
    min_alerts_for_retraining: int = 3
    alert_window_hours: int = 2
    
    # Model promotion requirements
    min_improvement_threshold: float = 0.02  # 2% improvement required
    
    def should_trigger_retraining(self, 
                                 alert_history: List[PerformanceAlert],
                                 last_retraining_time: Optional[datetime],
                                 retraining_count_today: int) -> Dict[str, Any]:
        """Determine if retraining should be triggered based on policy"""
        
        decision = {
            "should_trigger": False,
            "reason": "",
            "trigger_type": None,
            "priority": 3,
            "blocking_factors": []
        }
        
        # Check cooldown period
        if last_retraining_time:
            hours_since_last = (datetime.now() - last_retraining_time).total_seconds() / 3600
            if hours_since_last < self.min_time_between_retraining_hours:
                decision["blocking_factors"].append(
                    f"Cooldown period: {hours_since_last:.1f}h < {self.min_time_between_retraining_hours}h"
                )
        
        # Check daily limit
        if retraining_count_today >= self.max_retraining_per_day:
            decision["blocking_factors"].append(
                f"Daily limit reached: {retraining_count_today}/{self.max_retraining_per_day}"
            )
        
        # If blocked, return early
        if decision["blocking_factors"]:
            decision["reason"] = f"Blocked: {'; '.join(decision['blocking_factors'])}"
            return decision
        
        # Analyze recent alerts
        now = datetime.now()
        recent_alerts = [
            alert for alert in alert_history
            if (now - alert.timestamp).total_seconds() / 3600 <= self.alert_window_hours
        ]
        
        if len(recent_alerts) < self.min_alerts_for_retraining:
            decision["reason"] = f"Insufficient alerts: {len(recent_alerts)} < {self.min_alerts_for_retraining}"
            return decision
        
        # Check for critical performance issues
        critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        accuracy_alerts = [a for a in recent_alerts if a.alert_type == AlertType.ACCURACY_DROP]
        drift_alerts = [a for a in recent_alerts if a.alert_type == AlertType.DRIFT_DETECTED]
        
        if critical_alerts and accuracy_alerts:
            decision["should_trigger"] = True
            decision["trigger_type"] = RetrainingTrigger.PERFORMANCE_DEGRADATION
            decision["priority"] = 1  # High priority
            decision["reason"] = f"Critical performance degradation: {len(critical_alerts)} critical alerts, {len(accuracy_alerts)} accuracy alerts"
        
        elif drift_alerts:
            decision["should_trigger"] = True
            decision["trigger_type"] = RetrainingTrigger.DATA_DRIFT
            decision["priority"] = 2  # Medium priority
            decision["reason"] = f"Data drift detected: {len(drift_alerts)} drift alerts"
        
        elif len(recent_alerts) >= self.min_alerts_for_retraining * 2:
            decision["should_trigger"] = True
            decision["trigger_type"] = RetrainingTrigger.PERFORMANCE_DEGRADATION
            decision["priority"] = 2  # Medium priority
            decision["reason"] = f"Multiple performance issues: {len(recent_alerts)} alerts in {self.alert_window_hours}h"
        
        else:
            decision["reason"] = f"Conditions not met for retraining: {len(recent_alerts)} alerts, no critical issues"
        
        return decision

class RetrainingIntegrationService:
    """Service that integrates monitoring alerts with retraining workflows"""
    
    def __init__(self,
                 retraining_orchestrator: RetrainingOrchestrator,
                 policy: RetrainingPolicy = None):
        
        self.orchestrator = retraining_orchestrator
        self.policy = policy or RetrainingPolicy()
        
        # State tracking
        self.alert_history: List[PerformanceAlert] = []
        self.drift_reports: List[DriftReport] = []
        self.retraining_history: List[Dict[str, Any]] = []
        
        # Current model tracking
        self.current_model_id = "chest_xray_pneumonia"
        self.current_model_version = "v1.0"
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info("RetrainingIntegrationService initialized")
    
    def handle_performance_alert(self, alert: PerformanceAlert) -> Optional[str]:
        """Handle performance alert and potentially trigger retraining"""
        
        with self.lock:
            # Add to alert history
            self.alert_history.append(alert)
            
            # Keep only recent alerts (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.alert_history = [
                a for a in self.alert_history 
                if a.timestamp >= cutoff_time
            ]
        
        logger.info(f"Received performance alert: {alert.alert_type} (severity: {alert.severity})")
        
        # Check if retraining should be triggered
        decision = self._evaluate_retraining_decision()
        
        if decision["should_trigger"]:
            return self._trigger_retraining_from_alerts(decision)
        else:
            logger.info(f"Retraining not triggered: {decision['reason']}")
            return None
    
    def handle_drift_report(self, drift_report: DriftReport) -> Optional[str]:
        """Handle drift detection report and potentially trigger retraining"""
        
        with self.lock:
            self.drift_reports.append(drift_report)
            
            # Keep only recent reports (last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.drift_reports = [
                r for r in self.drift_reports 
                if r.timestamp >= cutoff_time
            ]
        
        logger.info(f"Received drift report: overall score {drift_report.overall_drift_score:.3f}")
        
        # Check if drift is severe enough to trigger retraining
        if (drift_report.overall_drift_score >= self.policy.data_drift_threshold or
            any(alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] 
                for alert in drift_report.alerts)):
            
            return self._trigger_drift_retraining(drift_report)
        else:
            logger.info(f"Drift not severe enough for retraining: {drift_report.overall_drift_score:.3f}")
            return None
    
    def _evaluate_retraining_decision(self) -> Dict[str, Any]:
        """Evaluate whether retraining should be triggered"""
        
        # Get recent retraining history
        last_retraining_time = self._get_last_retraining_time()
        retraining_count_today = self._get_retraining_count_today()
        
        return self.policy.should_trigger_retraining(
            alert_history=self.alert_history,
            last_retraining_time=last_retraining_time,
            retraining_count_today=retraining_count_today
        )
    
    def _trigger_retraining_from_alerts(self, decision: Dict[str, Any]) -> str:
        """Trigger retraining based on performance alerts"""
        
        # Collect relevant metrics from recent alerts
        performance_metrics = {}
        threshold_violations = []
        
        recent_alerts = [
            alert for alert in self.alert_history
            if (datetime.now() - alert.timestamp).total_seconds() / 3600 <= self.policy.alert_window_hours
        ]
        
        for alert in recent_alerts:
            if alert.alert_type == AlertType.ACCURACY_DROP:
                performance_metrics["accuracy"] = alert.current_value
                threshold_violations.append(f"Accuracy: {alert.current_value:.3f} < {alert.threshold:.3f}")
            
            elif alert.alert_type == AlertType.HIGH_LATENCY:
                performance_metrics["latency_ms"] = alert.current_value
                threshold_violations.append(f"Latency: {alert.current_value:.1f}ms > {alert.threshold:.1f}ms")
        
        # Trigger retraining
        request_id = trigger_performance_retraining(
            current_model_id=self.current_model_id,
            current_model_version=self.current_model_version,
            performance_metrics=performance_metrics,
            threshold_violations=threshold_violations
        )
        
        # Record retraining trigger
        self._record_retraining_trigger(request_id, decision)
        
        logger.info(f"Triggered performance retraining: {request_id}")
        return request_id
    
    def _trigger_drift_retraining(self, drift_report: DriftReport) -> str:
        """Trigger retraining based on drift detection"""
        
        request_id = trigger_drift_retraining(
            current_model_id=self.current_model_id,
            current_model_version=self.current_model_version,
            drift_report={
                "overall_drift_score": drift_report.overall_drift_score,
                "data_drift_score": drift_report.data_drift_score,
                "concept_drift_score": drift_report.concept_drift_score,
                "alerts": [
                    {
                        "drift_type": alert.drift_type,
                        "severity": alert.severity,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "baseline_value": alert.baseline_value
                    }
                    for alert in drift_report.alerts
                ],
                "recommendations": drift_report.recommendations
            }
        )
        
        # Record retraining trigger
        self._record_retraining_trigger(request_id, {
            "trigger_type": RetrainingTrigger.DATA_DRIFT,
            "reason": f"Drift score: {drift_report.overall_drift_score:.3f}",
            "priority": 2
        })
        
        logger.info(f"Triggered drift retraining: {request_id}")
        return request_id
    
    def _record_retraining_trigger(self, request_id: str, decision: Dict[str, Any]):
        """Record retraining trigger for history tracking"""
        
        with self.lock:
            self.retraining_history.append({
                "request_id": request_id,
                "triggered_at": datetime.now(),
                "trigger_type": decision.get("trigger_type"),
                "reason": decision.get("reason"),
                "priority": decision.get("priority"),
                "model_id": self.current_model_id,
                "model_version": self.current_model_version
            })
    
    def _get_last_retraining_time(self) -> Optional[datetime]:
        """Get timestamp of last retraining"""
        if not self.retraining_history:
            return None
        
        return max(entry["triggered_at"] for entry in self.retraining_history)
    
    def _get_retraining_count_today(self) -> int:
        """Get count of retraining requests today"""
        today = datetime.now().date()
        
        return sum(
            1 for entry in self.retraining_history
            if entry["triggered_at"].date() == today
        )
    
    def update_current_model(self, model_id: str, model_version: str):
        """Update current production model information"""
        with self.lock:
            self.current_model_id = model_id
            self.current_model_version = model_version
        
        logger.info(f"Updated current model: {model_id}:{model_version}")
    
    def get_retraining_statistics(self) -> Dict[str, Any]:
        """Get retraining statistics and history"""
        
        with self.lock:
            recent_alerts = [
                a for a in self.alert_history
                if (datetime.now() - a.timestamp).total_seconds() / 3600 <= 24
            ]
            
            recent_retraining = [
                r for r in self.retraining_history
                if (datetime.now() - r["triggered_at"]).total_seconds() / 3600 <= 24
            ]
        
        return {
            "current_model": {
                "model_id": self.current_model_id,
                "model_version": self.current_model_version
            },
            "policy": {
                "accuracy_threshold": self.policy.accuracy_threshold,
                "min_improvement_threshold": self.policy.min_improvement_threshold,
                "max_retraining_per_day": self.policy.max_retraining_per_day,
                "min_time_between_retraining_hours": self.policy.min_time_between_retraining_hours
            },
            "recent_activity": {
                "alerts_last_24h": len(recent_alerts),
                "retraining_requests_last_24h": len(recent_retraining),
                "retraining_count_today": self._get_retraining_count_today(),
                "last_retraining": self._get_last_retraining_time().isoformat() if self._get_last_retraining_time() else None
            },
            "alert_summary": {
                "total_alerts": len(self.alert_history),
                "critical_alerts": len([a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL]),
                "accuracy_alerts": len([a for a in self.alert_history if a.alert_type == AlertType.ACCURACY_DROP])
            },
            "drift_summary": {
                "total_reports": len(self.drift_reports),
                "high_drift_reports": len([
                    r for r in self.drift_reports 
                    if r.overall_drift_score >= self.policy.data_drift_threshold
                ])
            }
        }
    
    def simulate_retraining_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate retraining decision for testing purposes"""
        
        # Create mock alerts based on scenario
        mock_alerts = []
        
        if scenario.get("accuracy_drop"):
            mock_alerts.append(PerformanceAlert(
                alert_type=AlertType.ACCURACY_DROP,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                current_value=scenario["accuracy_drop"],
                threshold=self.policy.accuracy_threshold,
                metric_name="accuracy"
            ))
        
        if scenario.get("high_latency"):
            mock_alerts.append(PerformanceAlert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                timestamp=datetime.now(),
                current_value=scenario["high_latency"],
                threshold=self.policy.latency_threshold_ms,
                metric_name="latency"
            ))
        
        # Evaluate decision with mock alerts
        decision = self.policy.should_trigger_retraining(
            alert_history=mock_alerts,
            last_retraining_time=scenario.get("last_retraining_time"),
            retraining_count_today=scenario.get("retraining_count_today", 0)
        )
        
        return {
            "scenario": scenario,
            "decision": decision,
            "mock_alerts_count": len(mock_alerts)
        }

# Global integration service instance
_integration_service = None

def get_retraining_integration_service(
    retraining_orchestrator: RetrainingOrchestrator = None,
    policy: RetrainingPolicy = None
) -> RetrainingIntegrationService:
    """Get global retraining integration service instance"""
    global _integration_service
    
    if _integration_service is None:
        if retraining_orchestrator is None:
            retraining_orchestrator = get_retraining_orchestrator()
        
        _integration_service = RetrainingIntegrationService(
            retraining_orchestrator=retraining_orchestrator,
            policy=policy
        )
    
    return _integration_service