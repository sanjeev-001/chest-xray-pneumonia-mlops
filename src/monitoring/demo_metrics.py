#!/usr/bin/env python3
"""
Demo script for metrics collection system
Shows how to use the metrics collector for chest X-ray pneumonia detection
"""

import time
import random
from datetime import datetime
from monitoring.metrics_collector import (
    get_metrics_collector, 
    record_prediction_metric,
    PredictionStatus
)

def demo_metrics_collection():
    """Demonstrate metrics collection functionality"""
    print("Chest X-Ray Pneumonia Detection - Metrics Collection Demo")
    print("=" * 60)
    
    # Get metrics collector
    collector = get_metrics_collector()
    
    # Register a model
    print("\nRegistering model...")
    model_id = collector.register_model(
        name="chest-xray-classifier",
        version="v1.0.0",
        architecture="ResNet50",
        metrics={
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91,
            "auc_roc": 0.96
        }
    )
    print(f"Model registered with ID: {model_id}")
    
    # Register an experiment
    print("\nRegistering experiment...")
    experiment_id = collector.register_experiment(
        name="baseline-training-run-1",
        parameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "optimizer": "Adam",
            "data_augmentation": True
        },
        model_name="chest-xray-classifier",
        model_version="v1.0.0"
    )
    print(f"Experiment registered with ID: {experiment_id}")
    
    # Simulate some predictions
    print("\nSimulating predictions...")
    predictions_data = [
        ("PNEUMONIA", 0.95, 45.2, "Normal chest X-ray processing"),
        ("NORMAL", 0.88, 32.1, "Clear lung fields detected"),
        ("PNEUMONIA", 0.92, 38.7, "Consolidation pattern identified"),
        ("NORMAL", 0.91, 29.3, "No abnormalities found"),
        ("PNEUMONIA", 0.0, 120.0, "Model inference failed", PredictionStatus.ERROR),
        ("PNEUMONIA", 0.87, 41.5, "Bilateral infiltrates detected"),
        ("NORMAL", 0.93, 27.8, "Healthy lung tissue"),
    ]
    
    for i, prediction_data in enumerate(predictions_data, 1):
        if len(prediction_data) == 5:
            prediction, confidence, time_ms, description, status = prediction_data
            error_msg = "Model loading timeout" if status == PredictionStatus.ERROR else None
        else:
            prediction, confidence, time_ms, description = prediction_data
            status = PredictionStatus.SUCCESS
            error_msg = None
        
        # Record prediction
        prediction_id = collector.record_prediction(
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=time_ms,
            model_version="v1.0.0",
            image_id=f"chest_xray_{i:03d}.jpg",
            status=status,
            error_message=error_msg,
            metadata={
                "description": description,
                "patient_age": random.randint(25, 75),
                "hospital": "General Hospital",
                "radiologist": f"Dr. Smith_{i}"
            }
        )
        
        print(f"  Prediction {i}: {prediction} (confidence: {confidence:.2f}) - {prediction_id[:8]}...")
        time.sleep(0.1)  # Small delay to simulate real processing
    
    # Get prediction summary
    print("\nPrediction Summary (last hour):")
    summary = collector.get_prediction_summary(hours=1)
    print(f"  - Total predictions: {summary['total_predictions']}")
    print(f"  - Successful predictions: {summary['successful_predictions']}")
    print(f"  - Error count: {summary['error_count']}")
    print(f"  - Error rate: {summary['error_rate']:.2%}")
    print(f"  - Average confidence: {summary['avg_confidence']:.3f}")
    print(f"  - Average processing time: {summary['avg_processing_time_ms']:.1f}ms")
    print(f"  - Predictions by class:")
    for class_name, count in summary['predictions_by_class'].items():
        print(f"    * {class_name}: {count}")
    
    # Get recent predictions
    print("\nRecent Predictions:")
    recent = collector.get_recent_predictions(limit=5)
    for pred in recent:
        status_icon = "OK" if pred['status'] == PredictionStatus.SUCCESS else "ERR"
        print(f"  [{status_icon}] {pred['prediction']} (conf: {pred['confidence']:.2f}, "
              f"time: {pred['processing_time_ms']:.1f}ms)")
    
    # Simulate system metrics collection
    print("\nSystem Metrics:")
    collector._collect_system_metrics()
    system_metrics = collector.get_recent_system_metrics(limit=1)
    if system_metrics:
        metrics = system_metrics[0]
        print(f"  - CPU Usage: {metrics['cpu_percent']:.1f}%")
        print(f"  - Memory Usage: {metrics['memory_percent']:.1f}%")
        print(f"  - Disk Usage: {metrics['disk_usage_percent']:.1f}%")
        print(f"  - Active Connections: {metrics['active_connections']}")
    
    # Update experiment with final metrics
    if experiment_id:
        print(f"\nUpdating experiment {experiment_id[:8]}... with final metrics")
        final_metrics = {
            "final_accuracy": 0.924,
            "final_loss": 0.156,
            "training_time_minutes": 45,
            "total_epochs": 50,
            "best_epoch": 42
        }
        collector.update_experiment_metrics(experiment_id, final_metrics, status="completed")
        print("Experiment updated successfully")
    
    print("\nMetrics collection demo completed!")
    print("The system is ready to track predictions, system performance, and experiments.")
    print("Use the monitoring API endpoints to access this data in production.")

if __name__ == "__main__":
    demo_metrics_collection()