"""
Audit Trail System for Chest X-Ray Pneumonia Detection
Provides comprehensive logging, lineage tracking, and compliance reporting
"""

import logging
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import os

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    Json = None
    PSYCOPG2_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    PREDICTION = "prediction"
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    DATA_INGESTION = "data_ingestion"
    MODEL_REGISTRATION = "model_registration"
    RETRAINING_TRIGGER = "retraining_trigger"
    ALERT_GENERATED = "alert_generated"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_EXPORT = "data_export"

class ComplianceLevel(Enum):
    """Compliance requirement levels"""
    BASIC = "basic"
    MEDICAL = "medical"
    REGULATORY = "regulatory"
    FULL_AUDIT = "full_audit"

@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    component: str
    action: str
    resource_id: Optional[str]
    resource_type: Optional[str]
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    compliance_level: ComplianceLevel
    data_hash: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()
        if not self.data_hash and self.details:
            self.data_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash of event details for integrity verification"""
        content = json.dumps(self.details, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['compliance_level'] = self.compliance_level.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PredictionAuditRecord:
    """Comprehensive prediction audit record"""
    prediction_id: str
    timestamp: datetime
    model_id: str
    model_version: str
    input_image_hash: str
    input_metadata: Dict[str, Any]
    prediction_result: str
    confidence_score: float
    processing_time_ms: float
    api_endpoint: str
    client_ip: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    model_lineage: Dict[str, Any]
    explanation_available: bool = False
    explanation_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ModelLineage:
    """Model lineage information"""
    model_id: str
    model_version: str
    training_data_version: str
    training_data_hash: str
    training_config: Dict[str, Any]
    training_timestamp: datetime
    parent_model_id: Optional[str]
    experiment_id: str
    code_version: str
    dependencies: Dict[str, str]
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['training_timestamp'] = self.training_timestamp.isoformat()
        return data

class AuditTrailDatabase:
    """Database interface for audit trail storage"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.connection = None
        self.lock = threading.Lock()
        self._initialize_database()
        
        logger.info("AuditTrailDatabase initialized")
    
    def _get_connection(self):
        """Get database connection"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for database operations")
        
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(**self.connection_params)
        return self.connection
    
    def _initialize_database(self):
        """Initialize audit trail database schema"""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("PostgreSQL not available, using file-based storage")
            return
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Audit events table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS audit_events (
                            event_id UUID PRIMARY KEY,
                            event_type VARCHAR(50) NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            user_id VARCHAR(255),
                            session_id VARCHAR(255),
                            component VARCHAR(100) NOT NULL,
                            action VARCHAR(100) NOT NULL,
                            resource_id VARCHAR(255),
                            resource_type VARCHAR(100),
                            details JSONB,
                            metadata JSONB,
                            compliance_level VARCHAR(20) NOT NULL,
                            data_hash VARCHAR(64),
                            parent_event_id UUID,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (parent_event_id) REFERENCES audit_events(event_id)
                        )
                    """)
                    
                    # Prediction audit records table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS prediction_audit (
                            prediction_id UUID PRIMARY KEY,
                            timestamp TIMESTAMP NOT NULL,
                            model_id VARCHAR(255) NOT NULL,
                            model_version VARCHAR(50) NOT NULL,
                            input_image_hash VARCHAR(64) NOT NULL,
                            input_metadata JSONB,
                            prediction_result VARCHAR(50) NOT NULL,
                            confidence_score FLOAT NOT NULL,
                            processing_time_ms FLOAT NOT NULL,
                            api_endpoint VARCHAR(255),
                            client_ip INET,
                            user_agent TEXT,
                            session_id VARCHAR(255),
                            request_id VARCHAR(255),
                            model_lineage JSONB,
                            explanation_available BOOLEAN DEFAULT FALSE,
                            explanation_data JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Model lineage table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS model_lineage (
                            model_id VARCHAR(255) NOT NULL,
                            model_version VARCHAR(50) NOT NULL,
                            training_data_version VARCHAR(100) NOT NULL,
                            training_data_hash VARCHAR(64) NOT NULL,
                            training_config JSONB NOT NULL,
                            training_timestamp TIMESTAMP NOT NULL,
                            parent_model_id VARCHAR(255),
                            experiment_id VARCHAR(255) NOT NULL,
                            code_version VARCHAR(100) NOT NULL,
                            dependencies JSONB NOT NULL,
                            performance_metrics JSONB NOT NULL,
                            validation_results JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (model_id, model_version)
                        )
                    """)
                    
                    # Create indexes for better query performance
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_component ON audit_events(component)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_user ON audit_events(user_id)")
                    
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_audit_timestamp ON prediction_audit(timestamp)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_audit_model ON prediction_audit(model_id, model_version)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_audit_session ON prediction_audit(session_id)")
                    
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_lineage_model ON model_lineage(model_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_lineage_training ON model_lineage(training_timestamp)")
                    
                    conn.commit()
                    logger.info("Audit trail database schema initialized")
                    
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
    
    def store_audit_event(self, event: AuditEvent) -> bool:
        """Store audit event in database"""
        if not PSYCOPG2_AVAILABLE:
            return self._store_audit_event_file(event)
        
        try:
            with self.lock:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO audit_events 
                            (event_id, event_type, timestamp, user_id, session_id, component, 
                             action, resource_id, resource_type, details, metadata, 
                             compliance_level, data_hash, parent_event_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            event.event_id,
                            event.event_type.value,
                            event.timestamp,
                            event.user_id,
                            event.session_id,
                            event.component,
                            event.action,
                            event.resource_id,
                            event.resource_type,
                            Json(event.details),
                            Json(event.metadata),
                            event.compliance_level.value,
                            event.data_hash,
                            event.parent_event_id
                        ))
                        conn.commit()
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            return False
    
    def store_prediction_audit(self, record: PredictionAuditRecord) -> bool:
        """Store prediction audit record"""
        if not PSYCOPG2_AVAILABLE:
            return self._store_prediction_audit_file(record)
        
        try:
            with self.lock:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO prediction_audit 
                            (prediction_id, timestamp, model_id, model_version, input_image_hash,
                             input_metadata, prediction_result, confidence_score, processing_time_ms,
                             api_endpoint, client_ip, user_agent, session_id, request_id,
                             model_lineage, explanation_available, explanation_data)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            record.prediction_id,
                            record.timestamp,
                            record.model_id,
                            record.model_version,
                            record.input_image_hash,
                            Json(record.input_metadata),
                            record.prediction_result,
                            record.confidence_score,
                            record.processing_time_ms,
                            record.api_endpoint,
                            record.client_ip,
                            record.user_agent,
                            record.session_id,
                            record.request_id,
                            Json(record.model_lineage),
                            record.explanation_available,
                            Json(record.explanation_data) if record.explanation_data else None
                        ))
                        conn.commit()
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to store prediction audit record: {e}")
            return False
    
    def store_model_lineage(self, lineage: ModelLineage) -> bool:
        """Store model lineage information"""
        if not PSYCOPG2_AVAILABLE:
            return self._store_model_lineage_file(lineage)
        
        try:
            with self.lock:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO model_lineage 
                            (model_id, model_version, training_data_version, training_data_hash,
                             training_config, training_timestamp, parent_model_id, experiment_id,
                             code_version, dependencies, performance_metrics, validation_results)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (model_id, model_version) DO UPDATE SET
                                training_data_version = EXCLUDED.training_data_version,
                                training_data_hash = EXCLUDED.training_data_hash,
                                training_config = EXCLUDED.training_config,
                                training_timestamp = EXCLUDED.training_timestamp,
                                parent_model_id = EXCLUDED.parent_model_id,
                                experiment_id = EXCLUDED.experiment_id,
                                code_version = EXCLUDED.code_version,
                                dependencies = EXCLUDED.dependencies,
                                performance_metrics = EXCLUDED.performance_metrics,
                                validation_results = EXCLUDED.validation_results
                        """, (
                            lineage.model_id,
                            lineage.model_version,
                            lineage.training_data_version,
                            lineage.training_data_hash,
                            Json(lineage.training_config),
                            lineage.training_timestamp,
                            lineage.parent_model_id,
                            lineage.experiment_id,
                            lineage.code_version,
                            Json(lineage.dependencies),
                            Json(lineage.performance_metrics),
                            Json(lineage.validation_results)
                        ))
                        conn.commit()
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to store model lineage: {e}")
            return False
    
    def query_audit_events(self, 
                          start_time: datetime = None,
                          end_time: datetime = None,
                          event_type: AuditEventType = None,
                          component: str = None,
                          user_id: str = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        if not PSYCOPG2_AVAILABLE:
            return self._query_audit_events_file(start_time, end_time, event_type, component, user_id, limit)
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = "SELECT * FROM audit_events WHERE 1=1"
                    params = []
                    
                    if start_time:
                        query += " AND timestamp >= %s"
                        params.append(start_time)
                    
                    if end_time:
                        query += " AND timestamp <= %s"
                        params.append(end_time)
                    
                    if event_type:
                        query += " AND event_type = %s"
                        params.append(event_type.value)
                    
                    if component:
                        query += " AND component = %s"
                        params.append(component)
                    
                    if user_id:
                        query += " AND user_id = %s"
                        params.append(user_id)
                    
                    query += " ORDER BY timestamp DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    def query_prediction_audit(self,
                              start_time: datetime = None,
                              end_time: datetime = None,
                              model_id: str = None,
                              session_id: str = None,
                              limit: int = 1000) -> List[Dict[str, Any]]:
        """Query prediction audit records"""
        if not PSYCOPG2_AVAILABLE:
            return []
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = "SELECT * FROM prediction_audit WHERE 1=1"
                    params = []
                    
                    if start_time:
                        query += " AND timestamp >= %s"
                        params.append(start_time)
                    
                    if end_time:
                        query += " AND timestamp <= %s"
                        params.append(end_time)
                    
                    if model_id:
                        query += " AND model_id = %s"
                        params.append(model_id)
                    
                    if session_id:
                        query += " AND session_id = %s"
                        params.append(session_id)
                    
                    query += " ORDER BY timestamp DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Failed to query prediction audit: {e}")
            return []
    
    def get_model_lineage(self, model_id: str, model_version: str = None) -> Optional[Dict[str, Any]]:
        """Get model lineage information"""
        if not PSYCOPG2_AVAILABLE:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if model_version:
                        cursor.execute("""
                            SELECT * FROM model_lineage 
                            WHERE model_id = %s AND model_version = %s
                        """, (model_id, model_version))
                    else:
                        cursor.execute("""
                            SELECT * FROM model_lineage 
                            WHERE model_id = %s 
                            ORDER BY training_timestamp DESC 
                            LIMIT 1
                        """, (model_id,))
                    
                    row = cursor.fetchone()
                    return dict(row) if row else None
                    
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return None
    
    # File-based fallback methods
    def _store_audit_event_file(self, event: AuditEvent) -> bool:
        """Store audit event in file (fallback)"""
        try:
            audit_dir = Path("monitoring/audit_logs")
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = event.timestamp.strftime("%Y-%m-%d")
            file_path = audit_dir / f"audit_events_{date_str}.jsonl"
            
            with open(file_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store audit event to file: {e}")
            return False
    
    def _store_prediction_audit_file(self, record: PredictionAuditRecord) -> bool:
        """Store prediction audit in file (fallback)"""
        try:
            audit_dir = Path("monitoring/audit_logs")
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = record.timestamp.strftime("%Y-%m-%d")
            file_path = audit_dir / f"prediction_audit_{date_str}.jsonl"
            
            with open(file_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store prediction audit to file: {e}")
            return False
    
    def _store_model_lineage_file(self, lineage: ModelLineage) -> bool:
        """Store model lineage in file (fallback)"""
        try:
            audit_dir = Path("monitoring/audit_logs")
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = audit_dir / "model_lineage.jsonl"
            
            with open(file_path, "a") as f:
                f.write(json.dumps(lineage.to_dict()) + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store model lineage to file: {e}")
            return False
    
    def _query_audit_events_file(self, 
                                start_time: datetime = None,
                                end_time: datetime = None,
                                event_type: AuditEventType = None,
                                component: str = None,
                                user_id: str = None,
                                limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit events from files (fallback)"""
        try:
            audit_dir = Path("monitoring/audit_logs")
            if not audit_dir.exists():
                return []
            
            events = []
            
            # Read all audit event files
            for file_path in audit_dir.glob("audit_events_*.jsonl"):
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                event_data = json.loads(line.strip())
                                
                                # Apply filters
                                event_timestamp = datetime.fromisoformat(event_data.get('timestamp', ''))
                                
                                if start_time and event_timestamp < start_time:
                                    continue
                                if end_time and event_timestamp > end_time:
                                    continue
                                if event_type and event_data.get('event_type') != event_type.value:
                                    continue
                                if component and event_data.get('component') != component:
                                    continue
                                if user_id and event_data.get('user_id') != user_id:
                                    continue
                                
                                events.append(event_data)
                                
                except Exception as e:
                    logger.error(f"Failed to read audit file {file_path}: {e}")
                    continue
            
            # Sort by timestamp (newest first) and limit
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return events[:limit]
            
        except Exception as e:
            logger.error(f"Failed to query audit events from files: {e}")
            return []

class AuditTrailManager:
    """Main audit trail management system"""
    
    def __init__(self, 
                 database: AuditTrailDatabase,
                 compliance_level: ComplianceLevel = ComplianceLevel.MEDICAL):
        self.database = database
        self.compliance_level = compliance_level
        self.session_context = threading.local()
        
        logger.info(f"AuditTrailManager initialized with {compliance_level.value} compliance")
    
    def set_session_context(self, user_id: str = None, session_id: str = None):
        """Set session context for audit logging"""
        self.session_context.user_id = user_id
        self.session_context.session_id = session_id or str(uuid.uuid4())
    
    def log_prediction(self, 
                      prediction_id: str,
                      model_id: str,
                      model_version: str,
                      input_image_hash: str,
                      input_metadata: Dict[str, Any],
                      prediction_result: str,
                      confidence_score: float,
                      processing_time_ms: float,
                      api_endpoint: str = None,
                      client_ip: str = None,
                      user_agent: str = None,
                      request_id: str = None,
                      explanation_data: Dict[str, Any] = None) -> bool:
        """Log prediction with full audit trail"""
        
        # Get model lineage
        model_lineage = self.database.get_model_lineage(model_id, model_version) or {}
        
        # Create prediction audit record
        record = PredictionAuditRecord(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            model_id=model_id,
            model_version=model_version,
            input_image_hash=input_image_hash,
            input_metadata=input_metadata,
            prediction_result=prediction_result,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            api_endpoint=api_endpoint,
            client_ip=client_ip,
            user_agent=user_agent,
            session_id=getattr(self.session_context, 'session_id', None),
            request_id=request_id,
            model_lineage=model_lineage,
            explanation_available=explanation_data is not None,
            explanation_data=explanation_data
        )
        
        # Store prediction audit record
        success = self.database.store_prediction_audit(record)
        
        # Create audit event
        if success:
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.PREDICTION,
                timestamp=datetime.now(),
                user_id=getattr(self.session_context, 'user_id', None),
                session_id=getattr(self.session_context, 'session_id', None),
                component="prediction_service",
                action="predict",
                resource_id=prediction_id,
                resource_type="prediction",
                details={
                    "model_id": model_id,
                    "model_version": model_version,
                    "prediction_result": prediction_result,
                    "confidence_score": confidence_score,
                    "processing_time_ms": processing_time_ms
                },
                metadata={
                    "api_endpoint": api_endpoint,
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "request_id": request_id
                },
                compliance_level=self.compliance_level
            )
            
            self.database.store_audit_event(event)
        
        return success
    
    def log_model_training(self,
                          model_id: str,
                          model_version: str,
                          training_config: Dict[str, Any],
                          training_data_version: str,
                          training_data_hash: str,
                          experiment_id: str,
                          code_version: str,
                          dependencies: Dict[str, str],
                          performance_metrics: Dict[str, float],
                          validation_results: Dict[str, Any],
                          parent_model_id: str = None) -> bool:
        """Log model training with lineage"""
        
        # Create model lineage record
        lineage = ModelLineage(
            model_id=model_id,
            model_version=model_version,
            training_data_version=training_data_version,
            training_data_hash=training_data_hash,
            training_config=training_config,
            training_timestamp=datetime.now(),
            parent_model_id=parent_model_id,
            experiment_id=experiment_id,
            code_version=code_version,
            dependencies=dependencies,
            performance_metrics=performance_metrics,
            validation_results=validation_results
        )
        
        # Store lineage
        lineage_success = self.database.store_model_lineage(lineage)
        
        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.MODEL_TRAINING,
            timestamp=datetime.now(),
            user_id=getattr(self.session_context, 'user_id', None),
            session_id=getattr(self.session_context, 'session_id', None),
            component="training_service",
            action="train_model",
            resource_id=model_id,
            resource_type="model",
            details={
                "model_version": model_version,
                "training_data_version": training_data_version,
                "experiment_id": experiment_id,
                "performance_metrics": performance_metrics
            },
            metadata={
                "training_config": training_config,
                "code_version": code_version,
                "dependencies": dependencies,
                "validation_results": validation_results
            },
            compliance_level=self.compliance_level
        )
        
        audit_success = self.database.store_audit_event(event)
        
        return lineage_success and audit_success
    
    def log_model_deployment(self,
                           model_id: str,
                           model_version: str,
                           deployment_environment: str,
                           deployment_config: Dict[str, Any]) -> bool:
        """Log model deployment"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.MODEL_DEPLOYMENT,
            timestamp=datetime.now(),
            user_id=getattr(self.session_context, 'user_id', None),
            session_id=getattr(self.session_context, 'session_id', None),
            component="deployment_service",
            action="deploy_model",
            resource_id=model_id,
            resource_type="model",
            details={
                "model_version": model_version,
                "deployment_environment": deployment_environment,
                "deployment_config": deployment_config
            },
            metadata={
                "deployment_timestamp": datetime.now().isoformat()
            },
            compliance_level=self.compliance_level
        )
        
        return self.database.store_audit_event(event)
    
    def log_system_access(self,
                         user_id: str,
                         action: str,
                         resource: str,
                         success: bool,
                         details: Dict[str, Any] = None) -> bool:
        """Log system access events"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SYSTEM_ACCESS,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=getattr(self.session_context, 'session_id', None),
            component="access_control",
            action=action,
            resource_id=resource,
            resource_type="system_resource",
            details=details or {},
            metadata={
                "success": success
            },
            compliance_level=self.compliance_level
        )
        
        return self.database.store_audit_event(event)
    
    def generate_compliance_report(self,
                                  start_time: datetime,
                                  end_time: datetime,
                                  report_type: str = "full") -> Dict[str, Any]:
        """Generate compliance audit report"""
        
        # Query audit events
        events = self.database.query_audit_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Query prediction audits
        predictions = self.database.query_prediction_audit(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Generate report
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "compliance_level": self.compliance_level.value,
            "summary": {
                "total_events": len(events),
                "total_predictions": len(predictions),
                "event_types": {},
                "components": {},
                "users": set()
            },
            "events": events if report_type == "full" else [],
            "predictions": predictions if report_type == "full" else [],
            "integrity_check": self._verify_audit_integrity(events)
        }
        
        # Calculate summary statistics
        for event in events:
            event_type = event.get('event_type', 'unknown')
            component = event.get('component', 'unknown')
            user_id = event.get('user_id')
            
            report["summary"]["event_types"][event_type] = \
                report["summary"]["event_types"].get(event_type, 0) + 1
            
            report["summary"]["components"][component] = \
                report["summary"]["components"].get(component, 0) + 1
            
            if user_id:
                report["summary"]["users"].add(user_id)
        
        report["summary"]["unique_users"] = len(report["summary"]["users"])
        report["summary"]["users"] = list(report["summary"]["users"])
        
        return report
    
    def _verify_audit_integrity(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify audit trail integrity"""
        integrity_check = {
            "total_events": len(events),
            "hash_verified": 0,
            "hash_failed": 0,
            "missing_hash": 0,
            "integrity_score": 0.0
        }
        
        for event in events:
            data_hash = event.get('data_hash')
            details = event.get('details', {})
            
            if not data_hash:
                integrity_check["missing_hash"] += 1
                continue
            
            # Recalculate hash
            content = json.dumps(details, sort_keys=True, default=str)
            calculated_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if calculated_hash == data_hash:
                integrity_check["hash_verified"] += 1
            else:
                integrity_check["hash_failed"] += 1
        
        # Calculate integrity score
        if integrity_check["total_events"] > 0:
            integrity_check["integrity_score"] = \
                integrity_check["hash_verified"] / integrity_check["total_events"]
        
        return integrity_check

# Global audit trail manager instance
_audit_manager = None

def get_audit_manager(connection_params: Dict[str, str] = None,
                     compliance_level: ComplianceLevel = ComplianceLevel.MEDICAL) -> AuditTrailManager:
    """Get global audit trail manager instance"""
    global _audit_manager
    
    if _audit_manager is None:
        if connection_params is None:
            connection_params = {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", 5432)),
                "database": os.getenv("POSTGRES_DB", "mlops_audit"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "password")
            }
        
        database = AuditTrailDatabase(connection_params)
        _audit_manager = AuditTrailManager(database, compliance_level)
    
    return _audit_manager

def log_prediction_audit(prediction_id: str,
                        model_id: str,
                        model_version: str,
                        input_image_hash: str,
                        input_metadata: Dict[str, Any],
                        prediction_result: str,
                        confidence_score: float,
                        processing_time_ms: float,
                        **kwargs) -> bool:
    """Convenience function for logging predictions"""
    manager = get_audit_manager()
    return manager.log_prediction(
        prediction_id=prediction_id,
        model_id=model_id,
        model_version=model_version,
        input_image_hash=input_image_hash,
        input_metadata=input_metadata,
        prediction_result=prediction_result,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
        **kwargs
    )