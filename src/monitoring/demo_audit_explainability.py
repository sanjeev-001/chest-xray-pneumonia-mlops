"""
Demo script for Audit Trail and Explainability System
Shows how to use the integrated audit and explainability features
"""

import logging
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from audit_trail import (
    AuditTrailDatabase, AuditTrailManager, ComplianceLevel,
    get_audit_manager, AuditEventType
)
from explainability import ExplainabilityService, get_explainability_service
from audit_explainability_manager import (
    AuditExplainabilityManager, get_audit_explainability_manager
)

def create_sample_image() -> bytes:
    """Create a sample image for testing"""
    # Create a simple 224x224 RGB image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Convert to bytes (simulate JPEG encoding)
    import cv2
    _, encoded = cv2.imencode('.jpg', image)
    return encoded.tobytes()

def demo_basic_audit_logging():
    """Demonstrate basic audit logging functionality"""
    print("\n" + "="*60)
    print("DEMO: Basic Audit Logging")
    print("="*60)
    
    # Initialize audit manager with file-based storage (fallback)
    connection_params = {
        "host": "localhost",
        "database": "demo_audit",
        "user": "demo_user",
        "password": "demo_pass"
    }
    
    try:
        audit_manager = get_audit_manager(
            connection_params=connection_params,
            compliance_level=ComplianceLevel.MEDICAL
        )
        
        # Set session context
        audit_manager.set_session_context(
            user_id="demo_user",
            session_id="demo_session_123"
        )
        
        # Log a prediction
        success = audit_manager.log_prediction(
            prediction_id="demo_pred_001",
            model_id="chest_xray_model",
            model_version="v1.2.0",
            input_image_hash="abc123def456",
            input_metadata={
                "image_size": [224, 224],
                "preprocessing": ["resize", "normalize"],
                "source": "demo_hospital"
            },
            prediction_result="PNEUMONIA",
            confidence_score=0.87,
            processing_time_ms=145.2,
            api_endpoint="/api/v1/predict",
            client_ip="192.168.1.100",
            user_agent="DemoClient/1.0"
        )
        
        print(f"✓ Prediction logged successfully: {success}")
        
        # Log model training event
        training_success = audit_manager.log_model_training(
            model_id="chest_xray_model",
            model_version="v1.2.0",
            training_config={
                "architecture": "ResNet50",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "optimizer": "Adam"
            },
            training_data_version="chest_xray_v2.1",
            training_data_hash="training_hash_456",
            experiment_id="exp_20231201_001",
            code_version="commit_abc123",
            dependencies={
                "torch": "1.12.0",
                "torchvision": "0.13.0",
                "numpy": "1.21.0"
            },
            performance_metrics={
                "accuracy": 0.94,
                "precision": 0.92,
                "recall": 0.91,
                "f1_score": 0.915,
                "auc_roc": 0.96
            },
            validation_results={
                "val_accuracy": 0.91,
                "val_loss": 0.23,
                "confusion_matrix": [[850, 45], [32, 873]]
            }
        )
        
        print(f"✓ Model training logged successfully: {training_success}")
        
        # Log model deployment
        deployment_success = audit_manager.log_model_deployment(
            model_id="chest_xray_model",
            model_version="v1.2.0",
            deployment_environment="production",
            deployment_config={
                "replicas": 3,
                "cpu_limit": "2000m",
                "memory_limit": "4Gi",
                "gpu_enabled": True
            }
        )
        
        print(f"✓ Model deployment logged successfully: {deployment_success}")
        
        # Log system access
        access_success = audit_manager.log_system_access(
            user_id="demo_user",
            action="model_prediction",
            resource="chest_xray_model:v1.2.0",
            success=True,
            details={
                "endpoint": "/api/v1/predict",
                "method": "POST",
                "response_code": 200
            }
        )
        
        print(f"✓ System access logged successfully: {access_success}")
        
    except Exception as e:
        print(f"✗ Audit logging failed: {e}")
        print("Note: This is expected if PostgreSQL is not available - system will use file fallback")

def demo_explainability():
    """Demonstrate explainability functionality"""
    print("\n" + "="*60)
    print("DEMO: Model Explainability")
    print("="*60)
    
    try:
        # Get explainability service
        explainability_service = get_explainability_service()
        
        # Create sample image
        sample_image = create_sample_image()
        print(f"✓ Created sample image: {len(sample_image)} bytes")
        
        # Note: In a real scenario, you would load an actual trained model
        print("Note: Explainability demo requires a trained PyTorch model")
        print("      In production, this would generate GRAD-CAM visualizations")
        
        # Simulate explanation result
        from explainability import ExplanationResult
        
        mock_explanation = ExplanationResult(
            prediction_id="demo_pred_001",
            model_id="chest_xray_model",
            model_version="v1.2.0",
            explanation_type="comprehensive",
            confidence_score=0.87,
            predicted_class="PNEUMONIA",
            explanation_data={
                "gradcam": {
                    "target_layer": "layer4.2.conv3",
                    "max_activation": 0.94,
                    "important_regions": [
                        {
                            "region_id": 0,
                            "bounding_box": {"x": 45, "y": 67, "width": 89, "height": 76},
                            "mean_activation": 0.82,
                            "centroid": {"x": 89, "y": 105}
                        }
                    ]
                },
                "attention": {
                    "max_attention": 0.91,
                    "mean_attention": 0.34,
                    "attention_regions": [
                        {
                            "region_id": 0,
                            "bounding_box": {"x": 50, "y": 70, "width": 85, "height": 72},
                            "mean_activation": 0.78
                        }
                    ]
                },
                "feature_importance": {
                    "layer_analysis": {
                        "conv1": {"mean_activation": 0.45, "sparsity": 0.23},
                        "conv2": {"mean_activation": 0.67, "sparsity": 0.18}
                    },
                    "model_complexity": {
                        "total_parameters": 25557032,
                        "trainable_parameters": 25557032
                    }
                }
            },
            visualization_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        
        print("✓ Mock explanation generated:")
        print(f"  - Prediction: {mock_explanation.predicted_class}")
        print(f"  - Confidence: {mock_explanation.confidence_score:.3f}")
        print(f"  - GRAD-CAM regions: {len(mock_explanation.explanation_data['gradcam']['important_regions'])}")
        print(f"  - Attention regions: {len(mock_explanation.explanation_data['attention']['attention_regions'])}")
        print(f"  - Visualization available: {mock_explanation.visualization_data is not None}")
        
    except Exception as e:
        print(f"✗ Explainability demo failed: {e}")

def demo_integrated_system():
    """Demonstrate integrated audit and explainability system"""
    print("\n" + "="*60)
    print("DEMO: Integrated Audit & Explainability")
    print("="*60)
    
    try:
        # Get integrated manager
        manager = get_audit_explainability_manager(
            compliance_level=ComplianceLevel.MEDICAL,
            auto_explain_threshold=0.8
        )
        
        # Create sample image
        sample_image = create_sample_image()
        
        print("Testing prediction with LOW confidence (triggers explanation)...")
        
        # Log prediction with low confidence (should trigger explanation)
        record_low_conf = manager.log_prediction_with_explanation(
            prediction_id="demo_pred_low_conf",
            model_id="chest_xray_model",
            model_version="v1.2.0",
            image=sample_image,
            input_metadata={
                "image_size": [224, 224],
                "preprocessing": ["resize", "normalize"],
                "source": "demo_hospital"
            },
            prediction_result="PNEUMONIA",
            confidence_score=0.65,  # Below threshold
            processing_time_ms=180.5,
            api_context={
                "api_endpoint": "/api/v1/predict",
                "client_ip": "192.168.1.100",
                "user_agent": "DemoClient/1.0"
            }
        )
        
        print(f"✓ Low confidence prediction logged:")
        print(f"  - Prediction ID: {record_low_conf.prediction_id}")
        print(f"  - Confidence: {record_low_conf.confidence_score}")
        print(f"  - Explanation generated: {record_low_conf.explanation_result is not None}")
        print(f"  - Audit trail ID: {record_low_conf.audit_trail_id}")
        print(f"  - Integrity hash: {record_low_conf.integrity_hash[:16]}...")
        
        print("\nTesting prediction with HIGH confidence (no explanation)...")
        
        # Log prediction with high confidence (should not trigger explanation)
        record_high_conf = manager.log_prediction_with_explanation(
            prediction_id="demo_pred_high_conf",
            model_id="chest_xray_model",
            model_version="v1.2.0",
            image=sample_image,
            input_metadata={
                "image_size": [224, 224],
                "preprocessing": ["resize", "normalize"],
                "source": "demo_hospital"
            },
            prediction_result="NORMAL",
            confidence_score=0.92,  # Above threshold
            processing_time_ms=125.3,
            api_context={
                "api_endpoint": "/api/v1/predict",
                "client_ip": "192.168.1.101",
                "user_agent": "DemoClient/1.0"
            }
        )
        
        print(f"✓ High confidence prediction logged:")
        print(f"  - Prediction ID: {record_high_conf.prediction_id}")
        print(f"  - Confidence: {record_high_conf.confidence_score}")
        print(f"  - Explanation generated: {record_high_conf.explanation_result is not None}")
        print(f"  - Audit trail ID: {record_high_conf.audit_trail_id}")
        
        print("\nTesting FORCED explanation...")
        
        # Force explanation regardless of confidence
        record_forced = manager.log_prediction_with_explanation(
            prediction_id="demo_pred_forced",
            model_id="chest_xray_model",
            model_version="v1.2.0",
            image=sample_image,
            input_metadata={
                "image_size": [224, 224],
                "preprocessing": ["resize", "normalize"],
                "source": "demo_hospital"
            },
            prediction_result="NORMAL",
            confidence_score=0.95,  # High confidence
            processing_time_ms=165.8,
            force_explanation=True,  # Force explanation
            explanation_types=["gradcam", "attention"]
        )
        
        print(f"✓ Forced explanation prediction logged:")
        print(f"  - Prediction ID: {record_forced.prediction_id}")
        print(f"  - Confidence: {record_forced.confidence_score}")
        print(f"  - Explanation generated: {record_forced.explanation_result is not None}")
        
    except Exception as e:
        print(f"✗ Integrated system demo failed: {e}")

def demo_compliance_reporting():
    """Demonstrate compliance reporting"""
    print("\n" + "="*60)
    print("DEMO: Compliance Reporting")
    print("="*60)
    
    try:
        # Get integrated manager
        manager = get_audit_explainability_manager()
        
        # Generate compliance report for last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        print(f"Generating compliance report for period:")
        print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        report = manager.generate_compliance_report(
            start_time=start_time,
            end_time=end_time,
            include_explanations=True
        )
        
        print(f"\n✓ Compliance report generated:")
        print(f"  - Report ID: {report.get('report_id', 'N/A')}")
        print(f"  - Report type: {report.get('report_type', 'N/A')}")
        print(f"  - Compliance score: {report.get('compliance_score', 0):.3f}")
        
        if 'medical_compliance' in report:
            medical = report['medical_compliance']
            print(f"\n  Medical Compliance Details:")
            
            if 'prediction_traceability' in medical:
                trace = medical['prediction_traceability']
                print(f"    - Prediction traceability: {trace.get('compliant', False)}")
                print(f"      Score: {trace.get('traceability_score', 0):.3f}")
            
            if 'model_lineage_completeness' in medical:
                lineage = medical['model_lineage_completeness']
                print(f"    - Model lineage completeness: {lineage.get('compliant', False)}")
                print(f"      Coverage: {lineage.get('lineage_coverage', 0):.3f}")
            
            if 'explanation_coverage' in medical:
                explain = medical['explanation_coverage']
                print(f"    - Explanation coverage: {explain.get('compliant', False)}")
                print(f"      Coverage: {explain.get('explanation_coverage', 0):.3f}")
            
            if 'data_integrity' in medical:
                integrity = medical['data_integrity']
                print(f"    - Data integrity: {integrity.get('compliant', False)}")
                print(f"      Score: {integrity.get('integrity_score', 0):.3f}")
        
        # Test prediction lineage verification
        print(f"\n✓ Testing prediction lineage verification...")
        
        lineage_result = manager.verify_prediction_lineage("demo_pred_001")
        print(f"  - Lineage verified: {lineage_result.get('verified', False)}")
        print(f"  - Completeness: {lineage_result.get('lineage_completeness', 0):.3f}")
        
        if not lineage_result.get('verified', False):
            missing = lineage_result.get('missing_lineage_fields', [])
            if missing:
                print(f"  - Missing fields: {', '.join(missing)}")
        
        # Test audit data export
        print(f"\n✓ Testing audit data export...")
        
        export_data = manager.export_audit_data(
            start_time=start_time,
            end_time=end_time,
            export_format="json",
            include_explanations=True
        )
        
        # Parse and show summary
        parsed_export = json.loads(export_data)
        export_meta = parsed_export.get('export_metadata', {})
        
        print(f"  - Export ID: {export_meta.get('export_id', 'N/A')}")
        print(f"  - Export size: {len(export_data)} characters")
        print(f"  - Integrity hash: {export_meta.get('data_integrity_hash', 'N/A')[:16]}...")
        
    except Exception as e:
        print(f"✗ Compliance reporting demo failed: {e}")

def demo_audit_trail_queries():
    """Demonstrate audit trail querying"""
    print("\n" + "="*60)
    print("DEMO: Audit Trail Queries")
    print("="*60)
    
    try:
        # Get audit manager
        audit_manager = get_audit_manager()
        
        # Query recent audit events
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        print("Querying recent audit events...")
        
        events = audit_manager.database.query_audit_events(
            start_time=start_time,
            end_time=end_time,
            limit=10
        )
        
        print(f"✓ Found {len(events)} recent events:")
        
        for i, event in enumerate(events[:5], 1):  # Show first 5
            print(f"  {i}. {event.get('event_type', 'unknown')} - {event.get('component', 'unknown')}")
            print(f"     Action: {event.get('action', 'unknown')}")
            print(f"     Time: {event.get('timestamp', 'unknown')}")
            if event.get('user_id'):
                print(f"     User: {event.get('user_id')}")
        
        if len(events) > 5:
            print(f"     ... and {len(events) - 5} more events")
        
        # Query prediction audit records
        print(f"\nQuerying prediction audit records...")
        
        predictions = audit_manager.database.query_prediction_audit(
            start_time=start_time,
            end_time=end_time,
            limit=5
        )
        
        print(f"✓ Found {len(predictions)} recent predictions:")
        
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred.get('prediction_id', 'unknown')}")
            print(f"     Model: {pred.get('model_id', 'unknown')}:{pred.get('model_version', 'unknown')}")
            print(f"     Result: {pred.get('prediction_result', 'unknown')}")
            print(f"     Confidence: {pred.get('confidence_score', 0):.3f}")
            print(f"     Time: {pred.get('timestamp', 'unknown')}")
        
    except Exception as e:
        print(f"✗ Audit trail queries demo failed: {e}")

def main():
    """Run all demos"""
    print("CHEST X-RAY PNEUMONIA DETECTION")
    print("Audit Trail & Explainability System Demo")
    print("="*60)
    
    # Run all demos
    demo_basic_audit_logging()
    demo_explainability()
    demo_integrated_system()
    demo_compliance_reporting()
    demo_audit_trail_queries()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nNotes:")
    print("- Some features require PostgreSQL database connection")
    print("- Explainability requires trained PyTorch models")
    print("- File-based fallback is used when database is unavailable")
    print("- In production, integrate with your model registry and API")

if __name__ == "__main__":
    main()