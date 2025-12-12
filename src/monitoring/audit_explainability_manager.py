"""
Integrated Audit Trail and Explainability Manager
Combines audit logging with model explanations for comprehensive compliance
"""

import logging
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import asyncio

try:
    from .audit_trail import (
        AuditTrailManager, AuditTrailDatabase, 
        get_audit_manager, log_prediction_audit,
        ComplianceLevel, AuditEventType
    )
    from .explainability import (
        ExplainabilityService, get_explainability_service,
        ExplanationResult
    )
except ImportError:
    from audit_trail import (
        AuditTrailManager, AuditTrailDatabase, 
        get_audit_manager, log_prediction_audit,
        ComplianceLevel, AuditEventType
    )
    from explainability import (
        ExplainabilityService, get_explainability_service,
        ExplanationResult
    )

logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveAuditRecord:
    """Complete audit record with explanation"""
    prediction_id: str
    timestamp: datetime
    model_id: str
    model_version: str
    input_image_hash: str
    input_metadata: Dict[str, Any]
    prediction_result: str
    confidence_score: float
    processing_time_ms: float
    explanation_result: Optional[ExplanationResult]
    audit_trail_id: str
    compliance_level: ComplianceLevel
    api_context: Dict[str, Any]
    lineage_verified: bool = False
    integrity_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.audit_trail_id:
            self.audit_trail_id = str(uuid.uuid4())
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_integrity_hash()
    
    def _calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for the complete record"""
        data = {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "input_image_hash": self.input_image_hash,
            "prediction_result": self.prediction_result,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "input_image_hash": self.input_image_hash,
            "input_metadata": self.input_metadata,
            "prediction_result": self.prediction_result,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "explanation_result": self.explanation_result.to_dict() if self.explanation_result else None,
            "audit_trail_id": self.audit_trail_id,
            "compliance_level": self.compliance_level.value,
            "api_context": self.api_context,
            "lineage_verified": self.lineage_verified,
            "integrity_hash": self.integrity_hash
        }

class ComplianceReportGenerator:
    """Generates comprehensive compliance reports"""
    
    def __init__(self, audit_manager: AuditTrailManager):
        self.audit_manager = audit_manager
        logger.info("ComplianceReportGenerator initialized")
    
    def generate_medical_compliance_report(self,
                                         start_time: datetime,
                                         end_time: datetime,
                                         include_explanations: bool = True) -> Dict[str, Any]:
        """Generate medical AI compliance report"""
        
        # Get base audit report
        base_report = self.audit_manager.generate_compliance_report(
            start_time=start_time,
            end_time=end_time,
            report_type="full"
        )
        
        # Enhance with medical-specific requirements
        medical_report = {
            **base_report,
            "report_type": "medical_ai_compliance",
            "medical_compliance": {
                "prediction_traceability": self._check_prediction_traceability(
                    base_report["predictions"]
                ),
                "model_lineage_completeness": self._check_model_lineage_completeness(
                    base_report["predictions"]
                ),
                "explanation_coverage": self._check_explanation_coverage(
                    base_report["predictions"]
                ) if include_explanations else None,
                "data_integrity": self._check_data_integrity(
                    base_report["events"]
                ),
                "access_controls": self._check_access_controls(
                    base_report["events"]
                ),
                "regulatory_requirements": self._check_regulatory_requirements(
                    base_report
                )
            }
        }
        
        # Calculate overall compliance score
        medical_report["compliance_score"] = self._calculate_compliance_score(
            medical_report["medical_compliance"]
        )
        
        return medical_report
    
    def _check_prediction_traceability(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check prediction traceability requirements"""
        
        total_predictions = len(predictions)
        traceable_predictions = 0
        missing_fields = []
        
        required_fields = [
            "prediction_id", "timestamp", "model_id", "model_version",
            "input_image_hash", "prediction_result", "confidence_score"
        ]
        
        for prediction in predictions:
            has_all_fields = all(
                field in prediction and prediction[field] is not None
                for field in required_fields
            )
            
            if has_all_fields:
                traceable_predictions += 1
            else:
                for field in required_fields:
                    if field not in prediction or prediction[field] is None:
                        missing_fields.append(field)
        
        traceability_score = traceable_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            "total_predictions": total_predictions,
            "traceable_predictions": traceable_predictions,
            "traceability_score": traceability_score,
            "missing_fields": list(set(missing_fields)),
            "compliant": traceability_score >= 0.95  # 95% threshold
        }
    
    def _check_model_lineage_completeness(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check model lineage completeness"""
        
        models_with_lineage = set()
        models_without_lineage = set()
        
        for prediction in predictions:
            model_key = f"{prediction.get('model_id')}:{prediction.get('model_version')}"
            
            if prediction.get('model_lineage'):
                models_with_lineage.add(model_key)
            else:
                models_without_lineage.add(model_key)
        
        total_models = len(models_with_lineage | models_without_lineage)
        lineage_coverage = len(models_with_lineage) / total_models if total_models > 0 else 0
        
        return {
            "total_models": total_models,
            "models_with_lineage": len(models_with_lineage),
            "models_without_lineage": len(models_without_lineage),
            "lineage_coverage": lineage_coverage,
            "compliant": lineage_coverage >= 1.0  # 100% required for medical AI
        }
    
    def _check_explanation_coverage(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check explanation coverage for predictions"""
        
        total_predictions = len(predictions)
        predictions_with_explanations = 0
        explanation_types = set()
        
        for prediction in predictions:
            if prediction.get('explanation_available') or prediction.get('explanation_data'):
                predictions_with_explanations += 1
                
                # Track explanation types
                if prediction.get('explanation_data'):
                    exp_data = prediction['explanation_data']
                    if isinstance(exp_data, dict):
                        explanation_types.update(exp_data.keys())
        
        explanation_coverage = predictions_with_explanations / total_predictions if total_predictions > 0 else 0
        
        return {
            "total_predictions": total_predictions,
            "predictions_with_explanations": predictions_with_explanations,
            "explanation_coverage": explanation_coverage,
            "explanation_types": list(explanation_types),
            "compliant": explanation_coverage >= 0.8  # 80% threshold for medical AI
        }
    
    def _check_data_integrity(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data integrity of audit events"""
        
        total_events = len(events)
        verified_events = 0
        integrity_failures = []
        
        for event in events:
            data_hash = event.get('data_hash')
            details = event.get('details', {})
            
            if data_hash:
                # Recalculate hash
                content = json.dumps(details, sort_keys=True, default=str)
                calculated_hash = hashlib.sha256(content.encode()).hexdigest()
                
                if calculated_hash == data_hash:
                    verified_events += 1
                else:
                    integrity_failures.append({
                        "event_id": event.get('event_id'),
                        "expected_hash": data_hash,
                        "calculated_hash": calculated_hash
                    })
        
        integrity_score = verified_events / total_events if total_events > 0 else 0
        
        return {
            "total_events": total_events,
            "verified_events": verified_events,
            "integrity_score": integrity_score,
            "integrity_failures": integrity_failures,
            "compliant": integrity_score >= 0.99  # 99% threshold
        }
    
    def _check_access_controls(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check access control compliance"""
        
        access_events = [e for e in events if e.get('event_type') == 'system_access']
        total_access_events = len(access_events)
        
        authenticated_access = 0
        failed_access = 0
        unique_users = set()
        
        for event in access_events:
            if event.get('user_id'):
                authenticated_access += 1
                unique_users.add(event['user_id'])
            
            if event.get('metadata', {}).get('success') is False:
                failed_access += 1
        
        auth_rate = authenticated_access / total_access_events if total_access_events > 0 else 1
        failure_rate = failed_access / total_access_events if total_access_events > 0 else 0
        
        return {
            "total_access_events": total_access_events,
            "authenticated_access": authenticated_access,
            "failed_access": failed_access,
            "unique_users": len(unique_users),
            "authentication_rate": auth_rate,
            "failure_rate": failure_rate,
            "compliant": auth_rate >= 0.95 and failure_rate <= 0.05
        }
    
    def _check_regulatory_requirements(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance requirements"""
        
        requirements = {
            "data_retention": self._check_data_retention(report),
            "audit_completeness": self._check_audit_completeness(report),
            "model_validation": self._check_model_validation(report),
            "change_management": self._check_change_management(report)
        }
        
        # Calculate overall regulatory compliance
        compliant_requirements = sum(1 for req in requirements.values() if req.get('compliant', False))
        total_requirements = len(requirements)
        
        return {
            **requirements,
            "overall_compliance": compliant_requirements / total_requirements,
            "compliant": compliant_requirements == total_requirements
        }
    
    def _check_data_retention(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Check data retention compliance"""
        
        # Check if we have data spanning required retention period
        start_time = datetime.fromisoformat(report["period"]["start_time"])
        end_time = datetime.fromisoformat(report["period"]["end_time"])
        retention_days = (end_time - start_time).days
        
        # Medical AI typically requires 7+ years retention
        required_retention_days = 365 * 7  # 7 years
        
        return {
            "retention_period_days": retention_days,
            "required_retention_days": required_retention_days,
            "compliant": retention_days >= required_retention_days
        }
    
    def _check_audit_completeness(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Check audit trail completeness"""
        
        required_event_types = [
            "prediction", "model_training", "model_deployment",
            "data_ingestion", "model_registration"
        ]
        
        event_types = report["summary"].get("event_types", {})
        present_types = set(event_types.keys())
        missing_types = set(required_event_types) - present_types
        
        completeness_score = len(present_types & set(required_event_types)) / len(required_event_types)
        
        return {
            "required_event_types": required_event_types,
            "present_event_types": list(present_types),
            "missing_event_types": list(missing_types),
            "completeness_score": completeness_score,
            "compliant": completeness_score >= 1.0
        }
    
    def _check_model_validation(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Check model validation compliance"""
        
        # Look for model training and deployment events
        events = report.get("events", [])
        
        training_events = [e for e in events if e.get('event_type') == 'model_training']
        deployment_events = [e for e in events if e.get('event_type') == 'model_deployment']
        
        validated_models = 0
        total_models = len(training_events)
        
        for training_event in training_events:
            model_id = training_event.get('resource_id')
            
            # Check if model was validated before deployment
            corresponding_deployment = any(
                d.get('resource_id') == model_id and 
                d.get('timestamp') > training_event.get('timestamp')
                for d in deployment_events
            )
            
            if corresponding_deployment:
                validated_models += 1
        
        validation_rate = validated_models / total_models if total_models > 0 else 1
        
        return {
            "total_models": total_models,
            "validated_models": validated_models,
            "validation_rate": validation_rate,
            "compliant": validation_rate >= 1.0
        }
    
    def _check_change_management(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Check change management compliance"""
        
        events = report.get("events", [])
        
        # Look for configuration changes
        config_changes = [e for e in events if e.get('event_type') == 'configuration_change']
        
        # Check if changes are properly documented
        documented_changes = 0
        for change in config_changes:
            if (change.get('details') and 
                change.get('user_id') and 
                change.get('metadata')):
                documented_changes += 1
        
        documentation_rate = documented_changes / len(config_changes) if config_changes else 1
        
        return {
            "total_changes": len(config_changes),
            "documented_changes": documented_changes,
            "documentation_rate": documentation_rate,
            "compliant": documentation_rate >= 1.0
        }
    
    def _calculate_compliance_score(self, medical_compliance: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        
        scores = []
        weights = {
            "prediction_traceability": 0.25,
            "model_lineage_completeness": 0.20,
            "explanation_coverage": 0.15,
            "data_integrity": 0.20,
            "access_controls": 0.10,
            "regulatory_requirements": 0.10
        }
        
        for component, weight in weights.items():
            if component in medical_compliance and medical_compliance[component]:
                if component == "regulatory_requirements":
                    score = medical_compliance[component].get("overall_compliance", 0)
                else:
                    # Extract score from component
                    comp_data = medical_compliance[component]
                    if "traceability_score" in comp_data:
                        score = comp_data["traceability_score"]
                    elif "lineage_coverage" in comp_data:
                        score = comp_data["lineage_coverage"]
                    elif "explanation_coverage" in comp_data:
                        score = comp_data["explanation_coverage"]
                    elif "integrity_score" in comp_data:
                        score = comp_data["integrity_score"]
                    elif "authentication_rate" in comp_data:
                        score = comp_data["authentication_rate"]
                    else:
                        score = 1.0 if comp_data.get("compliant", False) else 0.0
                
                scores.append(score * weight)
        
        return sum(scores)

class AuditExplainabilityManager:
    """Integrated manager for audit trails and explainability"""
    
    def __init__(self, 
                 audit_manager: AuditTrailManager = None,
                 explainability_service: ExplainabilityService = None,
                 compliance_level: ComplianceLevel = ComplianceLevel.MEDICAL,
                 auto_explain_threshold: float = 0.7):
        
        self.audit_manager = audit_manager or get_audit_manager()
        self.explainability_service = explainability_service or get_explainability_service()
        self.compliance_level = compliance_level
        self.auto_explain_threshold = auto_explain_threshold
        
        # Report generator
        self.report_generator = ComplianceReportGenerator(self.audit_manager)
        
        # Thread pool for async operations
        self.executor = None
        
        logger.info(f"AuditExplainabilityManager initialized with {compliance_level.value} compliance")
    
    def log_prediction_with_explanation(self,
                                      prediction_id: str,
                                      model_id: str,
                                      model_version: str,
                                      image: Union[bytes, str],
                                      input_metadata: Dict[str, Any],
                                      prediction_result: str,
                                      confidence_score: float,
                                      processing_time_ms: float,
                                      api_context: Dict[str, Any] = None,
                                      force_explanation: bool = False,
                                      explanation_types: List[str] = None) -> ComprehensiveAuditRecord:
        """Log prediction with optional explanation based on confidence"""
        
        # Calculate input image hash
        if isinstance(image, bytes):
            input_image_hash = hashlib.sha256(image).hexdigest()
        elif isinstance(image, str):
            with open(image, 'rb') as f:
                input_image_hash = hashlib.sha256(f.read()).hexdigest()
        else:
            input_image_hash = "unknown"
        
        # Determine if explanation is needed
        needs_explanation = (
            force_explanation or 
            confidence_score < self.auto_explain_threshold or
            self.compliance_level in [ComplianceLevel.REGULATORY, ComplianceLevel.FULL_AUDIT]
        )
        
        explanation_result = None
        if needs_explanation:
            try:
                explanation_result = self.explainability_service.explain_prediction(
                    image=image,
                    prediction_id=prediction_id,
                    model_id=model_id,
                    model_version=model_version,
                    explanation_types=explanation_types
                )
                logger.info(f"Generated explanation for prediction {prediction_id}")
            except Exception as e:
                logger.error(f"Failed to generate explanation for {prediction_id}: {e}")
        
        # Log to audit trail
        audit_success = self.audit_manager.log_prediction(
            prediction_id=prediction_id,
            model_id=model_id,
            model_version=model_version,
            input_image_hash=input_image_hash,
            input_metadata=input_metadata,
            prediction_result=prediction_result,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            explanation_data=explanation_result.to_dict() if explanation_result else None,
            **api_context or {}
        )
        
        # Create comprehensive record
        record = ComprehensiveAuditRecord(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            model_id=model_id,
            model_version=model_version,
            input_image_hash=input_image_hash,
            input_metadata=input_metadata,
            prediction_result=prediction_result,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            explanation_result=explanation_result,
            audit_trail_id=str(uuid.uuid4()),
            compliance_level=self.compliance_level,
            api_context=api_context or {},
            lineage_verified=audit_success
        )
        
        logger.info(f"Logged comprehensive audit record for prediction {prediction_id}")
        return record
    
    def generate_explanation_on_demand(self,
                                     prediction_id: str,
                                     model_id: str,
                                     model_version: str,
                                     image: Union[bytes, str],
                                     explanation_types: List[str] = None) -> Optional[ExplanationResult]:
        """Generate explanation on demand for existing prediction"""
        
        try:
            explanation_result = self.explainability_service.explain_prediction(
                image=image,
                prediction_id=prediction_id,
                model_id=model_id,
                model_version=model_version,
                explanation_types=explanation_types
            )
            
            # Log explanation generation event
            self.audit_manager.database.store_audit_event({
                "event_id": str(uuid.uuid4()),
                "event_type": "explanation_generated",
                "timestamp": datetime.now(),
                "component": "explainability_service",
                "action": "generate_explanation",
                "resource_id": prediction_id,
                "resource_type": "prediction",
                "details": {
                    "model_id": model_id,
                    "model_version": model_version,
                    "explanation_types": explanation_types or []
                },
                "compliance_level": self.compliance_level.value
            })
            
            return explanation_result
            
        except Exception as e:
            logger.error(f"Failed to generate on-demand explanation: {e}")
            return None
    
    def generate_compliance_report(self,
                                 start_time: datetime,
                                 end_time: datetime,
                                 include_explanations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        return self.report_generator.generate_medical_compliance_report(
            start_time=start_time,
            end_time=end_time,
            include_explanations=include_explanations
        )
    
    def verify_prediction_lineage(self,
                                prediction_id: str) -> Dict[str, Any]:
        """Verify complete lineage for a prediction"""
        
        # Get prediction audit record
        predictions = self.audit_manager.database.query_prediction_audit(
            limit=1
        )
        
        prediction_record = None
        for pred in predictions:
            if pred.get('prediction_id') == prediction_id:
                prediction_record = pred
                break
        
        if not prediction_record:
            return {"verified": False, "error": "Prediction record not found"}
        
        # Get model lineage
        model_lineage = self.audit_manager.database.get_model_lineage(
            prediction_record['model_id'],
            prediction_record['model_version']
        )
        
        if not model_lineage:
            return {"verified": False, "error": "Model lineage not found"}
        
        # Verify lineage completeness
        required_lineage_fields = [
            "training_data_version", "training_data_hash", "training_config",
            "experiment_id", "code_version", "dependencies", "performance_metrics"
        ]
        
        missing_fields = [
            field for field in required_lineage_fields
            if field not in model_lineage or model_lineage[field] is None
        ]
        
        lineage_complete = len(missing_fields) == 0
        
        return {
            "verified": lineage_complete,
            "prediction_record": prediction_record,
            "model_lineage": model_lineage,
            "missing_lineage_fields": missing_fields,
            "lineage_completeness": 1.0 - (len(missing_fields) / len(required_lineage_fields))
        }
    
    def export_audit_data(self,
                         start_time: datetime,
                         end_time: datetime,
                         export_format: str = "json",
                         include_explanations: bool = True) -> str:
        """Export audit data for external compliance systems"""
        
        # Generate comprehensive report
        report = self.generate_compliance_report(
            start_time=start_time,
            end_time=end_time,
            include_explanations=include_explanations
        )
        
        # Add export metadata
        export_data = {
            "export_metadata": {
                "export_id": str(uuid.uuid4()),
                "exported_at": datetime.now().isoformat(),
                "export_format": export_format,
                "exported_by": "audit_explainability_manager",
                "data_integrity_hash": None
            },
            "compliance_report": report
        }
        
        # Calculate integrity hash
        content = json.dumps(report, sort_keys=True, default=str)
        export_data["export_metadata"]["data_integrity_hash"] = \
            hashlib.sha256(content.encode()).hexdigest()
        
        if export_format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def cleanup_old_data(self, retention_days: int = 2555):  # ~7 years default
        """Cleanup old audit data beyond retention period"""
        
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        # This would integrate with your database cleanup procedures
        logger.info(f"Cleanup requested for data older than {retention_days} days")
        
        # Log cleanup event
        self.audit_manager.database.store_audit_event({
            "event_id": str(uuid.uuid4()),
            "event_type": "data_cleanup",
            "timestamp": datetime.now(),
            "component": "audit_explainability_manager",
            "action": "cleanup_old_data",
            "details": {
                "retention_days": retention_days,
                "cutoff_time": cutoff_time.isoformat()
            },
            "compliance_level": self.compliance_level.value
        })

# Global manager instance
_audit_explainability_manager = None

def get_audit_explainability_manager(
    compliance_level: ComplianceLevel = ComplianceLevel.MEDICAL,
    auto_explain_threshold: float = 0.7
) -> AuditExplainabilityManager:
    """Get global audit explainability manager instance"""
    global _audit_explainability_manager
    
    if _audit_explainability_manager is None:
        _audit_explainability_manager = AuditExplainabilityManager(
            compliance_level=compliance_level,
            auto_explain_threshold=auto_explain_threshold
        )
    
    return _audit_explainability_manager