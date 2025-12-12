"""
Drift Detection for Chest X-Ray Pneumonia Detection System
Monitors data drift and concept drift in production
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    from .metrics_collector import MetricsDatabase
except ImportError:
    from metrics_collector import MetricsDatabase

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of drift detection"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftAlert:
    """Drift detection alert"""
    drift_type: DriftType
    severity: DriftSeverity
    timestamp: datetime
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    p_value: Optional[float] = None
    description: str = ""
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: datetime
    time_window_hours: int
    alerts: List[DriftAlert]
    data_drift_score: float
    concept_drift_score: float
    prediction_drift_score: float
    overall_drift_score: float
    recommendations: List[str]

class ImageStatistics:
    """Calculate and compare image statistics for drift detection"""
    
    @staticmethod
    def calculate_image_stats(image_hashes: List[str]) -> Dict[str, float]:
        """Calculate basic statistics from image hashes (proxy for image features)"""
        if not image_hashes:
            return {}
        
        # Convert hashes to numeric values for statistical analysis
        hash_values = []
        for hash_str in image_hashes:
            try:
                # Convert hex hash to integer, then normalize
                hash_int = int(hash_str[:8], 16) if len(hash_str) >= 8 else 0
                hash_values.append(hash_int)
            except ValueError:
                continue
        
        if not hash_values:
            return {}
        
        hash_array = np.array(hash_values)
        
        return {
            "mean": float(np.mean(hash_array)),
            "std": float(np.std(hash_array)),
            "min": float(np.min(hash_array)),
            "max": float(np.max(hash_array)),
            "median": float(np.median(hash_array)),
            "q25": float(np.percentile(hash_array, 25)),
            "q75": float(np.percentile(hash_array, 75)),
            "skewness": float(stats.skew(hash_array)) if SCIPY_AVAILABLE else 0.0,
            "kurtosis": float(stats.kurtosis(hash_array)) if SCIPY_AVAILABLE else 0.0
        }
    
    @staticmethod
    def compare_distributions(baseline_stats: Dict[str, float], 
                            current_stats: Dict[str, float],
                            threshold: float = 0.05) -> Tuple[bool, float, Dict[str, float]]:
        """Compare two statistical distributions"""
        if not baseline_stats or not current_stats:
            return False, 1.0, {}
        
        # Calculate relative differences for each statistic
        differences = {}
        significant_diffs = 0
        total_metrics = 0
        
        for metric in ['mean', 'std', 'median', 'skewness', 'kurtosis']:
            if metric in baseline_stats and metric in current_stats:
                baseline_val = baseline_stats[metric]
                current_val = current_stats[metric]
                
                if baseline_val != 0:
                    rel_diff = abs(current_val - baseline_val) / abs(baseline_val)
                    differences[metric] = rel_diff
                    
                    if rel_diff > threshold:
                        significant_diffs += 1
                    
                    total_metrics += 1
        
        # Calculate overall drift score
        drift_score = significant_diffs / total_metrics if total_metrics > 0 else 0.0
        is_drift = drift_score > 0.3  # 30% of metrics showing significant change
        
        return is_drift, drift_score, differences

class DriftDetector:
    """
    Main drift detection service
    """
    
    def __init__(self, database: MetricsDatabase = None):
        self.database = database or MetricsDatabase()
        
        # Drift detection thresholds
        self.data_drift_threshold = 0.05  # 5% change threshold
        self.concept_drift_threshold = 0.1  # 10% accuracy drop
        self.prediction_drift_threshold = 0.15  # 15% prediction distribution change
        
        # Baseline statistics (loaded from file or calculated)
        self.baseline_stats = {}
        self.baseline_file = Path("monitoring/baseline_stats.json")
        
        # Load existing baseline if available
        self._load_baseline_stats()
        
        logger.info("DriftDetector initialized")
    
    def _load_baseline_stats(self):
        """Load baseline statistics from file"""
        try:
            if self.baseline_file.exists():
                with open(self.baseline_file, 'r') as f:
                    self.baseline_stats = json.load(f)
                logger.info(f"Loaded baseline statistics from {self.baseline_file}")
            else:
                logger.info("No baseline statistics file found, will create on first calculation")
        except Exception as e:
            logger.warning(f"Failed to load baseline statistics: {e}")
            self.baseline_stats = {}
    
    def _save_baseline_stats(self):
        """Save baseline statistics to file"""
        try:
            self.baseline_file.parent.mkdir(exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2, default=str)
            logger.info(f"Saved baseline statistics to {self.baseline_file}")
        except Exception as e:
            logger.error(f"Failed to save baseline statistics: {e}")
    
    def establish_baseline(self, hours: int = 168) -> Dict[str, Any]:
        """Establish baseline statistics from historical data (default: 1 week)"""
        logger.info(f"Establishing baseline from last {hours} hours of data")
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Get historical predictions
        predictions = self.database.get_prediction_metrics(start_time=start_time, limit=10000)
        
        if len(predictions) < 100:
            logger.warning(f"Insufficient data for baseline ({len(predictions)} predictions). Need at least 100.")
            return {"error": "Insufficient data for baseline establishment"}
        
        # Calculate baseline statistics
        baseline = self._calculate_baseline_stats(predictions)
        
        # Save baseline
        self.baseline_stats = baseline
        self._save_baseline_stats()
        
        logger.info(f"Baseline established with {len(predictions)} predictions")
        return {
            "status": "baseline_established",
            "predictions_count": len(predictions),
            "time_period_hours": hours,
            "baseline_stats": baseline
        }
    
    def _calculate_baseline_stats(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate baseline statistics from predictions"""
        # Filter successful predictions
        successful_preds = [p for p in predictions if p.get('status') == 'success']
        
        if not successful_preds:
            return {}
        
        # Image statistics (from hashes)
        image_hashes = [p.get('image_hash', '') for p in successful_preds if p.get('image_hash')]
        image_stats = ImageStatistics.calculate_image_stats(image_hashes)
        
        # Prediction statistics
        confidences = [p['confidence'] for p in successful_preds]
        processing_times = [p.get('processing_time', 0) for p in successful_preds]
        
        # Prediction distribution
        predictions_by_class = {"NORMAL": 0, "PNEUMONIA": 0}
        for p in successful_preds:
            pred_class = p.get('prediction', '')
            if pred_class in predictions_by_class:
                predictions_by_class[pred_class] += 1
        
        total_preds = sum(predictions_by_class.values())
        prediction_distribution = {
            k: v / total_preds if total_preds > 0 else 0 
            for k, v in predictions_by_class.items()
        }
        
        # Accuracy (if actual labels available)
        labeled_preds = [p for p in successful_preds if p.get('actual_label')]
        accuracy = None
        if labeled_preds:
            correct = sum(1 for p in labeled_preds if p['prediction'] == p['actual_label'])
            accuracy = correct / len(labeled_preds)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(successful_preds),
            "image_statistics": image_stats,
            "confidence_stats": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "median": float(np.median(confidences))
            },
            "processing_time_stats": {
                "mean": float(np.mean(processing_times)),
                "std": float(np.std(processing_times)),
                "median": float(np.median(processing_times))
            },
            "prediction_distribution": prediction_distribution,
            "accuracy": accuracy
        }
    
    def detect_drift(self, hours: int = 24) -> DriftReport:
        """Detect drift in the specified time window"""
        logger.info(f"Running drift detection for last {hours} hours")
        
        if not self.baseline_stats:
            logger.warning("No baseline statistics available. Run establish_baseline() first.")
            return DriftReport(
                timestamp=datetime.now(),
                time_window_hours=hours,
                alerts=[],
                data_drift_score=0.0,
                concept_drift_score=0.0,
                prediction_drift_score=0.0,
                overall_drift_score=0.0,
                recommendations=["Establish baseline statistics before running drift detection"]
            )
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Get recent predictions
        recent_predictions = self.database.get_prediction_metrics(start_time=start_time, limit=10000)
        
        if len(recent_predictions) < 10:
            logger.warning(f"Insufficient recent data for drift detection ({len(recent_predictions)} predictions)")
            return DriftReport(
                timestamp=datetime.now(),
                time_window_hours=hours,
                alerts=[],
                data_drift_score=0.0,
                concept_drift_score=0.0,
                prediction_drift_score=0.0,
                overall_drift_score=0.0,
                recommendations=["Insufficient recent data for drift detection"]
            )
        
        # Calculate current statistics
        current_stats = self._calculate_baseline_stats(recent_predictions)
        
        # Detect different types of drift
        alerts = []
        
        # 1. Data Drift Detection
        data_drift_score, data_alerts = self._detect_data_drift(current_stats)
        alerts.extend(data_alerts)
        
        # 2. Concept Drift Detection
        concept_drift_score, concept_alerts = self._detect_concept_drift(current_stats)
        alerts.extend(concept_alerts)
        
        # 3. Prediction Drift Detection
        prediction_drift_score, prediction_alerts = self._detect_prediction_drift(current_stats)
        alerts.extend(prediction_alerts)
        
        # Calculate overall drift score
        overall_drift_score = max(data_drift_score, concept_drift_score, prediction_drift_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(alerts, overall_drift_score)
        
        report = DriftReport(
            timestamp=datetime.now(),
            time_window_hours=hours,
            alerts=alerts,
            data_drift_score=data_drift_score,
            concept_drift_score=concept_drift_score,
            prediction_drift_score=prediction_drift_score,
            overall_drift_score=overall_drift_score,
            recommendations=recommendations
        )
        
        logger.info(f"Drift detection completed. Overall score: {overall_drift_score:.3f}, Alerts: {len(alerts)}")
        
        return report
    
    def _detect_data_drift(self, current_stats: Dict[str, Any]) -> Tuple[float, List[DriftAlert]]:
        """Detect data drift by comparing image statistics"""
        alerts = []
        
        baseline_image_stats = self.baseline_stats.get('image_statistics', {})
        current_image_stats = current_stats.get('image_statistics', {})
        
        if not baseline_image_stats or not current_image_stats:
            return 0.0, alerts
        
        # Compare image statistics
        is_drift, drift_score, differences = ImageStatistics.compare_distributions(
            baseline_image_stats, current_image_stats, self.data_drift_threshold
        )
        
        if is_drift:
            # Determine severity
            if drift_score > 0.7:
                severity = DriftSeverity.CRITICAL
            elif drift_score > 0.5:
                severity = DriftSeverity.HIGH
            elif drift_score > 0.3:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            alert = DriftAlert(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                timestamp=datetime.now(),
                metric_name="image_statistics",
                current_value=drift_score,
                baseline_value=0.0,
                threshold=self.data_drift_threshold,
                description=f"Data drift detected in image statistics (score: {drift_score:.3f})",
                metadata={"differences": differences}
            )
            alerts.append(alert)
        
        return drift_score, alerts
    
    def _detect_concept_drift(self, current_stats: Dict[str, Any]) -> Tuple[float, List[DriftAlert]]:
        """Detect concept drift by comparing model accuracy"""
        alerts = []
        
        baseline_accuracy = self.baseline_stats.get('accuracy')
        current_accuracy = current_stats.get('accuracy')
        
        if baseline_accuracy is None or current_accuracy is None:
            return 0.0, alerts
        
        # Calculate accuracy drop
        accuracy_drop = baseline_accuracy - current_accuracy
        drift_score = max(0, accuracy_drop / baseline_accuracy)  # Normalized accuracy drop
        
        # Check if accuracy dropped significantly
        if accuracy_drop > self.concept_drift_threshold:
            # Determine severity based on accuracy drop
            if current_accuracy < 0.6:  # Below 60% accuracy
                severity = DriftSeverity.CRITICAL
            elif current_accuracy < 0.7:  # Below 70% accuracy
                severity = DriftSeverity.HIGH
            elif current_accuracy < 0.8:  # Below 80% accuracy (requirement threshold)
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            alert = DriftAlert(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                timestamp=datetime.now(),
                metric_name="accuracy",
                current_value=current_accuracy,
                baseline_value=baseline_accuracy,
                threshold=0.8,  # 80% threshold from requirements
                description=f"Model accuracy dropped from {baseline_accuracy:.3f} to {current_accuracy:.3f}",
                metadata={"accuracy_drop": accuracy_drop}
            )
            alerts.append(alert)
        
        return drift_score, alerts
    
    def _detect_prediction_drift(self, current_stats: Dict[str, Any]) -> Tuple[float, List[DriftAlert]]:
        """Detect prediction drift by comparing prediction distributions"""
        alerts = []
        
        baseline_dist = self.baseline_stats.get('prediction_distribution', {})
        current_dist = current_stats.get('prediction_distribution', {})
        
        if not baseline_dist or not current_dist:
            return 0.0, alerts
        
        # Calculate distribution differences
        drift_score = 0.0
        max_diff = 0.0
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            baseline_prob = baseline_dist.get(class_name, 0)
            current_prob = current_dist.get(class_name, 0)
            
            diff = abs(current_prob - baseline_prob)
            max_diff = max(max_diff, diff)
            drift_score += diff
        
        drift_score = drift_score / 2  # Average difference
        
        # Check if prediction distribution changed significantly
        if drift_score > self.prediction_drift_threshold:
            # Determine severity
            if drift_score > 0.4:
                severity = DriftSeverity.CRITICAL
            elif drift_score > 0.3:
                severity = DriftSeverity.HIGH
            elif drift_score > 0.2:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            alert = DriftAlert(
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=severity,
                timestamp=datetime.now(),
                metric_name="prediction_distribution",
                current_value=drift_score,
                baseline_value=0.0,
                threshold=self.prediction_drift_threshold,
                description=f"Prediction distribution changed significantly (score: {drift_score:.3f})",
                metadata={
                    "baseline_distribution": baseline_dist,
                    "current_distribution": current_dist
                }
            )
            alerts.append(alert)
        
        return drift_score, alerts
    
    def _generate_recommendations(self, alerts: List[DriftAlert], overall_score: float) -> List[str]:
        """Generate recommendations based on drift detection results"""
        recommendations = []
        
        if not alerts:
            recommendations.append("No significant drift detected - model performance is stable")
            return recommendations
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append("URGENT: Critical drift detected - immediate model retraining recommended")
        
        # Data drift recommendations
        data_drift_alerts = [a for a in alerts if a.drift_type == DriftType.DATA_DRIFT]
        if data_drift_alerts:
            recommendations.append("Data drift detected - investigate input data quality and preprocessing")
            recommendations.append("Consider updating data preprocessing pipeline")
        
        # Concept drift recommendations
        concept_drift_alerts = [a for a in alerts if a.drift_type == DriftType.CONCEPT_DRIFT]
        if concept_drift_alerts:
            recommendations.append("Concept drift detected - model retraining with recent data recommended")
            recommendations.append("Review model architecture and hyperparameters")
        
        # Prediction drift recommendations
        prediction_drift_alerts = [a for a in alerts if a.drift_type == DriftType.PREDICTION_DRIFT]
        if prediction_drift_alerts:
            recommendations.append("Prediction distribution changed - validate against ground truth labels")
            recommendations.append("Consider model calibration or threshold adjustment")
        
        # Overall recommendations based on score
        if overall_score > 0.7:
            recommendations.append("High drift score - schedule immediate model evaluation and retraining")
        elif overall_score > 0.5:
            recommendations.append("Moderate drift detected - plan model retraining within 1-2 weeks")
        elif overall_score > 0.3:
            recommendations.append("Low drift detected - monitor closely and consider retraining if trend continues")
        
        return recommendations
    
    def get_drift_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical drift detection results"""
        # This would typically be stored in database
        # For now, return placeholder
        return []
    
    def update_baseline(self, hours: int = 168) -> Dict[str, Any]:
        """Update baseline statistics with recent data"""
        return self.establish_baseline(hours)

# Global drift detector instance
_drift_detector = None

def get_drift_detector() -> DriftDetector:
    """Get global drift detector instance"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector

if __name__ == "__main__":
    # Test drift detection
    print("Testing Drift Detection System...")
    
    detector = DriftDetector()
    
    # Test baseline establishment (will fail without data, but tests the flow)
    try:
        result = detector.establish_baseline(hours=24)
        print(f"✅ Baseline establishment: {result}")
    except Exception as e:
        print(f"⚠️  Baseline establishment (expected to fail without data): {e}")
    
    # Test drift detection (will return empty results without baseline)
    try:
        report = detector.detect_drift(hours=24)
        print(f"✅ Drift detection completed: {len(report.alerts)} alerts")
    except Exception as e:
        print(f"❌ Drift detection failed: {e}")
    
    print("✅ Drift detection system initialized!")