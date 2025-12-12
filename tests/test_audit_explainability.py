"""
Tests for Audit Trail and Explainability System
"""

import pytest
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.audit_trail import (
    AuditEvent, AuditEventType, ComplianceLevel, PredictionAuditRecord,
    ModelLineage, AuditTrailDatabase, AuditTrailManager
)
from monitoring.explainability import (
    ExplanationResult, ModelExplainer, ExplainabilityService
)
from monitoring.audit_explainability_manager import (
    ComprehensiveAuditRecord, ComplianceReportGenerator, AuditExplainabilityManager
)

class TestAuditTrail:
    """Test audit trail functionality"""
    
    def test_audit_event_creation(self):
        """Test audit event creation and validation"""
        event = AuditEvent(
            event_id="test-123",
            event_type=AuditEventType.PREDICTION,
            timestamp=datetime.now(),
            user_id="test_user",
            session_id="session_123",
            component="test_component",
            action="test_action",
            resource_id="resource_123",
            resource_type="test_resource",
            details={"key": "value"},
            metadata={"meta": "data"},
            compliance_level=ComplianceLevel.MEDICAL
        )
        
        assert event.event_id == "test-123"
        assert event.event_type == AuditEventType.PREDICTION
        assert event.user_id == "test_user"
        assert event.data_hash is not None
        
        # Test dictionary conversion
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "prediction"
        assert event_dict["compliance_level"] == "medical"
    
    def test_prediction_audit_record(self):
        """Test prediction audit record creation"""
        record = PredictionAuditRecord(
            prediction_id="pred_123",
            timestamp=datetime.now(),
            model_id="model_123",
            model_version="v1.0",
            input_image_hash="abc123",
            input_metadata={"width": 224, "height": 224},
            prediction_result="PNEUMONIA",
            confidence_score=0.85,
            processing_time_ms=150.5,
            api_endpoint="/predict",
            client_ip="192.168.1.1",
            model_lineage={"training_data": "v1.0"}
        )
        
        assert record.prediction_id == "pred_123"
        assert record.confidence_score == 0.85
        assert record.model_lineage["training_data"] == "v1.0"
        
        # Test dictionary conversion
        record_dict = record.to_dict()
        assert "timestamp" in record_dict
        assert record_dict["prediction_result"] == "PNEUMONIA"
    
    def test_model_lineage(self):
        """Test model lineage tracking"""
        lineage = ModelLineage(
            model_id="model_123",
            model_version="v1.0",
            training_data_version="data_v1.0",
            training_data_hash="hash123",
            training_config={"lr": 0.001, "batch_size": 32},
            training_timestamp=datetime.now(),
            experiment_id="exp_123",
            code_version="commit_abc",
            dependencies={"torch": "1.9.0", "numpy": "1.21.0"},
            performance_metrics={"accuracy": 0.95, "f1": 0.93},
            validation_results={"val_acc": 0.92}
        )
        
        assert lineage.model_id == "model_123"
        assert lineage.performance_metrics["accuracy"] == 0.95
        
        # Test dictionary conversion
        lineage_dict = lineage.to_dict()
        assert "training_timestamp" in lineage_dict
        assert lineage_dict["dependencies"]["torch"] == "1.9.0"
    
    @patch('monitoring.audit_trail.psycopg2')
    def test_audit_trail_database_file_fallback(self, mock_psycopg2):
        """Test file-based storage fallback when PostgreSQL is not available"""
        mock_psycopg2.connect.side_effect = Exception("Connection failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create database with file fallback
            db = AuditTrailDatabase({
                "host": "localhost",
                "database": "test"
            })
            
            # Test storing audit event
            event = AuditEvent(
                event_id="test-123",
                event_type=AuditEventType.PREDICTION,
                timestamp=datetime.now(),
                component="test",
                action="test",
                details={"test": "data"},
                compliance_level=ComplianceLevel.BASIC
            )
            
            # This should use file fallback
            success = db._store_audit_event_file(event)
            assert success
    
    def test_audit_trail_manager(self):
        """Test audit trail manager functionality"""
        # Mock database
        mock_db = Mock()
        mock_db.store_prediction_audit.return_value = True
        mock_db.store_audit_event.return_value = True
        mock_db.get_model_lineage.return_value = {"training_data": "v1.0"}
        
        manager = AuditTrailManager(mock_db, ComplianceLevel.MEDICAL)
        
        # Test prediction logging
        success = manager.log_prediction(
            prediction_id="pred_123",
            model_id="model_123",
            model_version="v1.0",
            input_image_hash="hash123",
            input_metadata={"test": "data"},
            prediction_result="NORMAL",
            confidence_score=0.9,
            processing_time_ms=100.0
        )
        
        assert success
        mock_db.store_prediction_audit.assert_called_once()
        mock_db.store_audit_event.assert_called_once()
    
    def test_compliance_report_generation(self):
        """Test compliance report generation"""
        # Mock database with sample data
        mock_db = Mock()
        mock_db.query_audit_events.return_value = [
            {
                "event_id": "1",
                "event_type": "prediction",
                "component": "api",
                "user_id": "user1",
                "details": {"test": "data"},
                "data_hash": hashlib.sha256(json.dumps({"test": "data"}, sort_keys=True).encode()).hexdigest()
            }
        ]
        mock_db.query_prediction_audit.return_value = [
            {
                "prediction_id": "pred1",
                "model_id": "model1",
                "model_version": "v1.0",
                "model_lineage": {"training_data": "v1.0"},
                "explanation_available": True
            }
        ]
        
        manager = AuditTrailManager(mock_db)
        report_gen = ComplianceReportGenerator(manager)
        
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        
        report = report_gen.generate_medical_compliance_report(
            start_time=start_time,
            end_time=end_time
        )
        
        assert "medical_compliance" in report
        assert "compliance_score" in report
        assert "prediction_traceability" in report["medical_compliance"]
        assert "model_lineage_completeness" in report["medical_compliance"]

class TestExplainability:
    """Test explainability functionality"""
    
    def test_explanation_result_creation(self):
        """Test explanation result creation"""
        result = ExplanationResult(
            prediction_id="pred_123",
            model_id="model_123",
            model_version="v1.0",
            explanation_type="gradcam",
            confidence_score=0.85,
            predicted_class="PNEUMONIA",
            explanation_data={"gradcam": {"max_activation": 0.9}},
            visualization_data="base64_image_data"
        )
        
        assert result.prediction_id == "pred_123"
        assert result.explanation_type == "gradcam"
        assert result.generated_at is not None
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        assert result_dict["predicted_class"] == "PNEUMONIA"
        assert "generated_at" in result_dict
    
    @patch('monitoring.explainability.torch')
    def test_gradcam_initialization(self, mock_torch):
        """Test GRAD-CAM initialization"""
        # Mock PyTorch components
        mock_model = Mock()
        mock_model.named_modules.return_value = [
            ("features.conv1", Mock()),
            ("features.conv2", Mock())
        ]
        
        from monitoring.explainability import GradCAM
        
        # This should work with mocked torch
        gradcam = GradCAM(mock_model, "features.conv2")
        assert gradcam.target_layer == "features.conv2"
        assert gradcam.model == mock_model
    
    @patch('monitoring.explainability.torch')
    @patch('monitoring.explainability.transforms')
    def test_model_explainer_initialization(self, mock_transforms, mock_torch):
        """Test model explainer initialization"""
        mock_model = Mock()
        mock_transforms.Compose.return_value = Mock()
        
        from monitoring.explainability import ModelExplainer
        
        explainer = ModelExplainer(mock_model, device="cpu")
        assert explainer.model == mock_model
        assert explainer.device == "cpu"
        assert explainer.class_names == ["NORMAL", "PNEUMONIA"]
    
    def test_explainability_service(self):
        """Test explainability service"""
        from monitoring.explainability import ExplainabilityService
        
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ExplainabilityService(temp_dir)
            
            # Test model info for non-existent model
            info = service.get_model_info("model_123", "v1.0")
            assert info == {}
            
            # Test cleanup
            service.cleanup_model("model_123", "v1.0")  # Should not raise error

class TestIntegratedSystem:
    """Test integrated audit and explainability system"""
    
    def test_comprehensive_audit_record(self):
        """Test comprehensive audit record creation"""
        explanation_result = ExplanationResult(
            prediction_id="pred_123",
            model_id="model_123",
            model_version="v1.0",
            explanation_type="gradcam",
            confidence_score=0.85,
            predicted_class="PNEUMONIA",
            explanation_data={"gradcam": {"test": "data"}}
        )
        
        record = ComprehensiveAuditRecord(
            prediction_id="pred_123",
            timestamp=datetime.now(),
            model_id="model_123",
            model_version="v1.0",
            input_image_hash="hash123",
            input_metadata={"test": "data"},
            prediction_result="PNEUMONIA",
            confidence_score=0.85,
            processing_time_ms=150.0,
            explanation_result=explanation_result,
            audit_trail_id="audit_123",
            compliance_level=ComplianceLevel.MEDICAL,
            api_context={"endpoint": "/predict"}
        )
        
        assert record.prediction_id == "pred_123"
        assert record.explanation_result is not None
        assert record.integrity_hash is not None
        
        # Test dictionary conversion
        record_dict = record.to_dict()
        assert "explanation_result" in record_dict
        assert record_dict["explanation_result"]["explanation_type"] == "gradcam"
    
    def test_audit_explainability_manager_initialization(self):
        """Test integrated manager initialization"""
        mock_audit_manager = Mock()
        mock_explainability_service = Mock()
        
        manager = AuditExplainabilityManager(
            audit_manager=mock_audit_manager,
            explainability_service=mock_explainability_service,
            compliance_level=ComplianceLevel.MEDICAL,
            auto_explain_threshold=0.7
        )
        
        assert manager.audit_manager == mock_audit_manager
        assert manager.explainability_service == mock_explainability_service
        assert manager.compliance_level == ComplianceLevel.MEDICAL
        assert manager.auto_explain_threshold == 0.7
    
    def test_prediction_logging_with_explanation(self):
        """Test prediction logging with automatic explanation"""
        # Mock components
        mock_audit_manager = Mock()
        mock_audit_manager.log_prediction.return_value = True
        
        mock_explainability_service = Mock()
        mock_explanation = ExplanationResult(
            prediction_id="pred_123",
            model_id="model_123",
            model_version="v1.0",
            explanation_type="gradcam",
            confidence_score=0.6,  # Low confidence
            predicted_class="PNEUMONIA",
            explanation_data={"gradcam": {"test": "data"}}
        )
        mock_explainability_service.explain_prediction.return_value = mock_explanation
        
        manager = AuditExplainabilityManager(
            audit_manager=mock_audit_manager,
            explainability_service=mock_explainability_service,
            auto_explain_threshold=0.7
        )
        
        # Test with low confidence (should trigger explanation)
        image_data = b"fake_image_data"
        record = manager.log_prediction_with_explanation(
            prediction_id="pred_123",
            model_id="model_123",
            model_version="v1.0",
            image=image_data,
            input_metadata={"test": "data"},
            prediction_result="PNEUMONIA",
            confidence_score=0.6,  # Below threshold
            processing_time_ms=150.0
        )
        
        assert record.prediction_id == "pred_123"
        assert record.explanation_result is not None
        mock_explainability_service.explain_prediction.assert_called_once()
        mock_audit_manager.log_prediction.assert_called_once()
    
    def test_prediction_logging_without_explanation(self):
        """Test prediction logging without explanation for high confidence"""
        mock_audit_manager = Mock()
        mock_audit_manager.log_prediction.return_value = True
        
        mock_explainability_service = Mock()
        
        manager = AuditExplainabilityManager(
            audit_manager=mock_audit_manager,
            explainability_service=mock_explainability_service,
            auto_explain_threshold=0.7
        )
        
        # Test with high confidence (should not trigger explanation)
        image_data = b"fake_image_data"
        record = manager.log_prediction_with_explanation(
            prediction_id="pred_123",
            model_id="model_123",
            model_version="v1.0",
            image=image_data,
            input_metadata={"test": "data"},
            prediction_result="NORMAL",
            confidence_score=0.9,  # Above threshold
            processing_time_ms=100.0
        )
        
        assert record.prediction_id == "pred_123"
        assert record.explanation_result is None
        mock_explainability_service.explain_prediction.assert_not_called()
        mock_audit_manager.log_prediction.assert_called_once()
    
    def test_compliance_report_generation(self):
        """Test compliance report generation"""
        mock_audit_manager = Mock()
        mock_report_generator = Mock()
        mock_report = {
            "report_id": "report_123",
            "medical_compliance": {
                "prediction_traceability": {"compliant": True},
                "model_lineage_completeness": {"compliant": True}
            }
        }
        mock_report_generator.generate_medical_compliance_report.return_value = mock_report
        
        manager = AuditExplainabilityManager(audit_manager=mock_audit_manager)
        manager.report_generator = mock_report_generator
        
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        
        report = manager.generate_compliance_report(
            start_time=start_time,
            end_time=end_time
        )
        
        assert report["report_id"] == "report_123"
        assert "medical_compliance" in report
        mock_report_generator.generate_medical_compliance_report.assert_called_once()
    
    def test_prediction_lineage_verification(self):
        """Test prediction lineage verification"""
        mock_audit_manager = Mock()
        mock_db = Mock()
        
        # Mock prediction record
        mock_db.query_prediction_audit.return_value = [{
            "prediction_id": "pred_123",
            "model_id": "model_123",
            "model_version": "v1.0"
        }]
        
        # Mock model lineage
        mock_db.get_model_lineage.return_value = {
            "training_data_version": "v1.0",
            "training_data_hash": "hash123",
            "training_config": {"lr": 0.001},
            "experiment_id": "exp_123",
            "code_version": "commit_abc",
            "dependencies": {"torch": "1.9.0"},
            "performance_metrics": {"accuracy": 0.95}
        }
        
        mock_audit_manager.database = mock_db
        
        manager = AuditExplainabilityManager(audit_manager=mock_audit_manager)
        
        result = manager.verify_prediction_lineage("pred_123")
        
        assert result["verified"] is True
        assert "prediction_record" in result
        assert "model_lineage" in result
        assert result["lineage_completeness"] == 1.0
    
    def test_audit_data_export(self):
        """Test audit data export functionality"""
        mock_audit_manager = Mock()
        mock_report_generator = Mock()
        mock_report = {
            "report_id": "report_123",
            "summary": {"total_events": 100}
        }
        mock_report_generator.generate_medical_compliance_report.return_value = mock_report
        
        manager = AuditExplainabilityManager(audit_manager=mock_audit_manager)
        manager.report_generator = mock_report_generator
        
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        
        export_data = manager.export_audit_data(
            start_time=start_time,
            end_time=end_time,
            export_format="json"
        )
        
        # Parse exported JSON
        parsed_data = json.loads(export_data)
        
        assert "export_metadata" in parsed_data
        assert "compliance_report" in parsed_data
        assert "export_id" in parsed_data["export_metadata"]
        assert "data_integrity_hash" in parsed_data["export_metadata"]

class TestComplianceReporting:
    """Test compliance reporting functionality"""
    
    def test_prediction_traceability_check(self):
        """Test prediction traceability compliance check"""
        mock_audit_manager = Mock()
        report_gen = ComplianceReportGenerator(mock_audit_manager)
        
        # Test with complete predictions
        complete_predictions = [
            {
                "prediction_id": "pred1",
                "timestamp": "2023-01-01T00:00:00",
                "model_id": "model1",
                "model_version": "v1.0",
                "input_image_hash": "hash1",
                "prediction_result": "NORMAL",
                "confidence_score": 0.9
            }
        ]
        
        result = report_gen._check_prediction_traceability(complete_predictions)
        
        assert result["total_predictions"] == 1
        assert result["traceable_predictions"] == 1
        assert result["traceability_score"] == 1.0
        assert result["compliant"] is True
        
        # Test with incomplete predictions
        incomplete_predictions = [
            {
                "prediction_id": "pred1",
                "model_id": "model1"
                # Missing required fields
            }
        ]
        
        result = report_gen._check_prediction_traceability(incomplete_predictions)
        
        assert result["total_predictions"] == 1
        assert result["traceable_predictions"] == 0
        assert result["traceability_score"] == 0.0
        assert result["compliant"] is False
        assert len(result["missing_fields"]) > 0
    
    def test_model_lineage_completeness_check(self):
        """Test model lineage completeness check"""
        mock_audit_manager = Mock()
        report_gen = ComplianceReportGenerator(mock_audit_manager)
        
        # Test with complete lineage
        predictions_with_lineage = [
            {
                "model_id": "model1",
                "model_version": "v1.0",
                "model_lineage": {"training_data": "v1.0"}
            }
        ]
        
        result = report_gen._check_model_lineage_completeness(predictions_with_lineage)
        
        assert result["total_models"] == 1
        assert result["models_with_lineage"] == 1
        assert result["lineage_coverage"] == 1.0
        assert result["compliant"] is True
        
        # Test without lineage
        predictions_without_lineage = [
            {
                "model_id": "model1",
                "model_version": "v1.0"
                # No model_lineage field
            }
        ]
        
        result = report_gen._check_model_lineage_completeness(predictions_without_lineage)
        
        assert result["total_models"] == 1
        assert result["models_with_lineage"] == 0
        assert result["lineage_coverage"] == 0.0
        assert result["compliant"] is False
    
    def test_explanation_coverage_check(self):
        """Test explanation coverage check"""
        mock_audit_manager = Mock()
        report_gen = ComplianceReportGenerator(mock_audit_manager)
        
        # Test with explanations
        predictions_with_explanations = [
            {
                "prediction_id": "pred1",
                "explanation_available": True,
                "explanation_data": {"gradcam": {"test": "data"}}
            },
            {
                "prediction_id": "pred2",
                "explanation_available": False
            }
        ]
        
        result = report_gen._check_explanation_coverage(predictions_with_explanations)
        
        assert result["total_predictions"] == 2
        assert result["predictions_with_explanations"] == 1
        assert result["explanation_coverage"] == 0.5
        assert "gradcam" in result["explanation_types"]
    
    def test_data_integrity_check(self):
        """Test data integrity verification"""
        mock_audit_manager = Mock()
        report_gen = ComplianceReportGenerator(mock_audit_manager)
        
        # Create event with correct hash
        details = {"test": "data"}
        correct_hash = hashlib.sha256(json.dumps(details, sort_keys=True, default=str).encode()).hexdigest()
        
        events = [
            {
                "event_id": "event1",
                "details": details,
                "data_hash": correct_hash
            },
            {
                "event_id": "event2",
                "details": {"other": "data"},
                "data_hash": "wrong_hash"
            }
        ]
        
        result = report_gen._check_data_integrity(events)
        
        assert result["total_events"] == 2
        assert result["verified_events"] == 1
        assert result["integrity_score"] == 0.5
        assert len(result["integrity_failures"]) == 1
        assert result["compliant"] is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])