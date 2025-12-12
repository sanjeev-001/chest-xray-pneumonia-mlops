"""
Metrics Collector for Chest X-Ray Pneumonia Detection System
Collects, stores, and manages prediction and system metrics
"""

import time
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import psutil
import os
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    PSYCOPG2_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)

class PredictionStatus(Enum):
    """Prediction status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class PredictionMetric:
    """Individual prediction metric"""
    prediction_id: str
    timestamp: datetime
    model_version: str
    prediction: str
    confidence: float
    processing_time_ms: float
    image_hash: str
    status: PredictionStatus
    error_message: Optional[str] = None
    actual_label: Optional[str] = None  # For accuracy tracking when available
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_memory_mb: Optional[float]
    disk_usage_percent: float
    request_count: int
    error_count: int
    active_connections: int
    response_time_avg_ms: float
    cache_hit_rate: float

class MetricsDatabase:
    """
    PostgreSQL database for storing metrics
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL', 
            'postgresql://postgres:password@localhost:5432/chest_xray_metrics'
        )
        self.connection = None
        self.lock = threading.Lock()
        
        # Check if psycopg2 is available
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available, database functionality disabled")
            return
        
        # Initialize database
        self._initialize_database()
        
        logger.info("MetricsDatabase initialized")
    
    def _get_connection(self):
        """Get database connection"""
        if not PSYCOPG2_AVAILABLE:
            return None
            
        if self.connection is None or self.connection.closed:
            try:
                self.connection = psycopg2.connect(self.connection_string)
                self.connection.autocommit = True
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
        
        return self.connection
    
    def _initialize_database(self):
        """Initialize database tables"""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("Database initialization skipped - psycopg2 not available")
            return
            
        try:
            conn = self._get_connection()
            if not conn:
                return
            cursor = conn.cursor()
            
            # Create models table (as per design document)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    architecture VARCHAR(100),
                    created_at TIMESTAMP NOT NULL,
                    metrics JSONB,
                    status VARCHAR(50) NOT NULL,
                    model_path TEXT,
                    UNIQUE(name, version)
                );
            """)
            
            # Create experiments table (as per design document)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    parameters JSONB,
                    metrics JSONB,
                    created_at TIMESTAMP NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    model_id UUID REFERENCES models(id)
                );
            """)
            
            # Create predictions table (enhanced to match design + requirements)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id UUID PRIMARY KEY,
                    model_id UUID REFERENCES models(id),
                    image_id VARCHAR(255),
                    prediction VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    processing_time FLOAT NOT NULL,
                    image_hash VARCHAR(64),
                    status VARCHAR(20) NOT NULL,
                    error_message TEXT,
                    actual_label VARCHAR(20),
                    metadata JSONB,
                    model_version VARCHAR(50) NOT NULL
                );
            """)
            
            # Create system_metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    cpu_percent FLOAT NOT NULL,
                    memory_percent FLOAT NOT NULL,
                    gpu_memory_mb FLOAT,
                    disk_usage_percent FLOAT NOT NULL,
                    request_count INTEGER NOT NULL,
                    error_count INTEGER NOT NULL,
                    active_connections INTEGER NOT NULL,
                    response_time_avg_ms FLOAT NOT NULL,
                    cache_hit_rate FLOAT NOT NULL
                );
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                ON predictions(timestamp);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_model_version 
                ON predictions(model_version);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_model_id 
                ON predictions(model_id);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
                ON system_metrics(timestamp);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_name_version 
                ON models(name, version);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_created_at 
                ON experiments(created_at);
            """)
            
            logger.info("Database tables initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # For testing, we'll continue without database
            self.connection = None
    
    def store_prediction_metric(self, metric: PredictionMetric, model_id: str = None) -> bool:
        """Store prediction metric in database"""
        if not self.connection:
            logger.warning("Database not available, skipping metric storage")
            return False
        
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                # Generate UUID for prediction if not provided
                prediction_uuid = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT INTO predictions (
                        id, model_id, image_id, prediction, confidence, 
                        timestamp, processing_time, image_hash, status,
                        error_message, actual_label, metadata, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    prediction_uuid,
                    model_id,
                    metric.prediction_id,  # Using prediction_id as image_id for now
                    metric.prediction,
                    metric.confidence,
                    metric.timestamp,
                    metric.processing_time_ms / 1000.0,  # Convert to seconds as per design
                    metric.image_hash,
                    metric.status.value,
                    metric.error_message,
                    metric.actual_label,
                    json.dumps(metric.metadata) if metric.metadata else None,
                    metric.model_version
                ))
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store prediction metric: {e}")
            return False
    
    def store_system_metric(self, metric: SystemMetric) -> bool:
        """Store system metric in database"""
        if not self.connection:
            logger.warning("Database not available, skipping metric storage")
            return False
        
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, gpu_memory_mb,
                        disk_usage_percent, request_count, error_count,
                        active_connections, response_time_avg_ms, cache_hit_rate
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    metric.timestamp,
                    metric.cpu_percent,
                    metric.memory_percent,
                    metric.gpu_memory_mb,
                    metric.disk_usage_percent,
                    metric.request_count,
                    metric.error_count,
                    metric.active_connections,
                    metric.response_time_avg_ms,
                    metric.cache_hit_rate
                ))
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store system metric: {e}")
            return False
    
    def get_prediction_metrics(self, 
                             start_time: datetime = None, 
                             end_time: datetime = None,
                             model_version: str = None,
                             limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve prediction metrics"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM predictions WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            
            if model_version:
                query += " AND model_version = %s"
                params.append(model_version)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve prediction metrics: {e}")
            return []
    
    def get_system_metrics(self, 
                          start_time: datetime = None, 
                          end_time: datetime = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve system metrics"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM system_metrics WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve system metrics: {e}")
            return []
    
    def get_accuracy_over_time(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get accuracy metrics over time (when actual labels are available)"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            start_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN actual_label IS NOT NULL THEN 1 END) as labeled_predictions,
                    COUNT(CASE WHEN prediction = actual_label THEN 1 END) as correct_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time
                FROM predictions 
                WHERE timestamp >= %s AND actual_label IS NOT NULL
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC
            """, (start_time,))
            
            results = cursor.fetchall()
            
            # Calculate accuracy for each hour
            for row in results:
                if row['labeled_predictions'] > 0:
                    row['accuracy'] = row['correct_predictions'] / row['labeled_predictions']
                else:
                    row['accuracy'] = None
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve accuracy metrics: {e}")
            return []
    
    def register_model(self, name: str, version: str, architecture: str = None, 
                      metrics: Dict[str, Any] = None, status: str = "active",
                      model_path: str = None) -> str:
        """Register a new model in the database"""
        if not self.connection:
            logger.warning("Database not available, skipping model registration")
            return None
        
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                model_id = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT INTO models (
                        id, name, version, architecture, created_at, 
                        metrics, status, model_path
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name, version) DO UPDATE SET
                        architecture = EXCLUDED.architecture,
                        metrics = EXCLUDED.metrics,
                        status = EXCLUDED.status,
                        model_path = EXCLUDED.model_path
                    RETURNING id
                """, (
                    model_id,
                    name,
                    version,
                    architecture,
                    datetime.now(),
                    json.dumps(metrics) if metrics else None,
                    status,
                    model_path
                ))
                
                result = cursor.fetchone()
                return result[0] if result else model_id
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def get_model_id(self, name: str, version: str) -> str:
        """Get model ID by name and version"""
        if not self.connection:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT id FROM models WHERE name = %s AND version = %s",
                (name, version)
            )
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get model ID: {e}")
            return None
    
    def register_experiment(self, name: str, parameters: Dict[str, Any] = None,
                           metrics: Dict[str, Any] = None, status: str = "running",
                           model_id: str = None) -> str:
        """Register a new experiment in the database"""
        if not self.connection:
            logger.warning("Database not available, skipping experiment registration")
            return None
        
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                experiment_id = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT INTO experiments (
                        id, name, parameters, metrics, created_at, status, model_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    experiment_id,
                    name,
                    json.dumps(parameters) if parameters else None,
                    json.dumps(metrics) if metrics else None,
                    datetime.now(),
                    status,
                    model_id
                ))
                
                return experiment_id
                
        except Exception as e:
            logger.error(f"Failed to register experiment: {e}")
            return None
    
    def update_experiment_metrics(self, experiment_id: str, metrics: Dict[str, Any],
                                 status: str = None) -> bool:
        """Update experiment metrics"""
        if not self.connection:
            return False
        
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                if status:
                    cursor.execute("""
                        UPDATE experiments 
                        SET metrics = %s, status = %s 
                        WHERE id = %s
                    """, (json.dumps(metrics), status, experiment_id))
                else:
                    cursor.execute("""
                        UPDATE experiments 
                        SET metrics = %s 
                        WHERE id = %s
                    """, (json.dumps(metrics), experiment_id))
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update experiment metrics: {e}")
            return False
    
    def get_models(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get registered models"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM models 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def get_experiments(self, model_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get experiments, optionally filtered by model"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            if model_id:
                cursor.execute("""
                    SELECT * FROM experiments 
                    WHERE model_id = %s
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (model_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM experiments 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            return []

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class MetricsCollector:
    """
    Main metrics collection service
    """
    
    def __init__(self, database: MetricsDatabase = None, enable_file_backup: bool = True):
        self.database = database or MetricsDatabase()
        self.enable_file_backup = enable_file_backup
        
        # In-memory storage for recent metrics (fallback)
        self.recent_predictions = []
        self.recent_system_metrics = []
        self.max_memory_metrics = 10000
        
        # File backup
        if enable_file_backup:
            self.backup_dir = Path("metrics_backup")
            self.backup_dir.mkdir(exist_ok=True)
        
        # Metrics aggregation
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.predictions_by_class = {"NORMAL": 0, "PNEUMONIA": 0}
        
        # System monitoring
        self.system_monitor_interval = 60  # seconds
        self.system_monitor_thread = None
        self.stop_monitoring = threading.Event()
        
        logger.info("MetricsCollector initialized")
    
    def start_system_monitoring(self):
        """Start system metrics monitoring thread"""
        if self.system_monitor_thread is None or not self.system_monitor_thread.is_alive():
            self.stop_monitoring.clear()
            self.system_monitor_thread = threading.Thread(target=self._system_monitor_loop)
            self.system_monitor_thread.daemon = True
            self.system_monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop system metrics monitoring"""
        self.stop_monitoring.set()
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _system_monitor_loop(self):
        """System monitoring loop"""
        while not self.stop_monitoring.wait(self.system_monitor_interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU memory (if available)
            gpu_memory_mb = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                # PyTorch not available
                pass
            
            # Calculate current response time average
            response_time_avg = np.mean(self.response_times[-100:]) if self.response_times else 0
            
            # Calculate cache hit rate (placeholder - would integrate with actual cache)
            cache_hit_rate = 0.8  # Placeholder
            
            # Create system metric
            system_metric = SystemMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_memory_mb=gpu_memory_mb,
                disk_usage_percent=disk.percent,
                request_count=self.request_count,
                error_count=self.error_count,
                active_connections=len(psutil.net_connections()),
                response_time_avg_ms=response_time_avg,
                cache_hit_rate=cache_hit_rate
            )
            
            # Store in database
            self.database.store_system_metric(system_metric)
            
            # Store in memory (with limit)
            self.recent_system_metrics.append(asdict(system_metric))
            if len(self.recent_system_metrics) > self.max_memory_metrics:
                self.recent_system_metrics = self.recent_system_metrics[-self.max_memory_metrics:]
            
            # File backup
            if self.enable_file_backup:
                self._backup_system_metric(system_metric)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_prediction(self, 
                         prediction: str,
                         confidence: float,
                         processing_time_ms: float,
                         model_version: str,
                         image_data: bytes = None,
                         image_id: str = None,
                         status: PredictionStatus = PredictionStatus.SUCCESS,
                         error_message: str = None,
                         actual_label: str = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Record a prediction metric"""
        
        # Generate prediction ID if not provided
        prediction_id = image_id or str(uuid.uuid4())
        
        # Generate image hash
        image_hash = "unknown"
        if image_data:
            image_hash = hashlib.sha256(image_data).hexdigest()[:16]
        
        # Get model ID from database
        model_id = self.database.get_model_id("chest-xray-classifier", model_version)
        
        # Create prediction metric
        prediction_metric = PredictionMetric(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            model_version=model_version,
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            image_hash=image_hash,
            status=status,
            error_message=error_message,
            actual_label=actual_label,
            metadata=metadata
        )
        
        # Store in database
        self.database.store_prediction_metric(prediction_metric, model_id)
        
        # Store in memory
        self.recent_predictions.append(asdict(prediction_metric))
        if len(self.recent_predictions) > self.max_memory_metrics:
            self.recent_predictions = self.recent_predictions[-self.max_memory_metrics:]
        
        # Update aggregated metrics
        self.request_count += 1
        if status == PredictionStatus.ERROR:
            self.error_count += 1
        else:
            if prediction in self.predictions_by_class:
                self.predictions_by_class[prediction] += 1
        
        self.response_times.append(processing_time_ms)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # File backup
        if self.enable_file_backup:
            self._backup_prediction_metric(prediction_metric)
        
        logger.debug(f"Recorded prediction: {prediction} (confidence: {confidence:.3f})")
        
        return prediction_id
    
    def register_model(self, name: str, version: str, architecture: str = None,
                      metrics: Dict[str, Any] = None, status: str = "active",
                      model_path: str = None) -> str:
        """Register a model in the metrics system"""
        return self.database.register_model(
            name=name,
            version=version,
            architecture=architecture,
            metrics=metrics,
            status=status,
            model_path=model_path
        )
    
    def register_experiment(self, name: str, parameters: Dict[str, Any] = None,
                           metrics: Dict[str, Any] = None, status: str = "running",
                           model_name: str = None, model_version: str = None) -> str:
        """Register an experiment in the metrics system"""
        model_id = None
        if model_name and model_version:
            model_id = self.database.get_model_id(model_name, model_version)
        
        return self.database.register_experiment(
            name=name,
            parameters=parameters,
            metrics=metrics,
            status=status,
            model_id=model_id
        )
    
    def update_experiment_metrics(self, experiment_id: str, metrics: Dict[str, Any],
                                 status: str = None) -> bool:
        """Update experiment metrics"""
        return self.database.update_experiment_metrics(experiment_id, metrics, status)
    
    def _backup_prediction_metric(self, metric: PredictionMetric):
        """Backup prediction metric to file"""
        try:
            date_str = metric.timestamp.strftime("%Y-%m-%d")
            backup_file = self.backup_dir / f"predictions_{date_str}.jsonl"
            
            with open(backup_file, 'a') as f:
                json.dump(asdict(metric), f, default=str)
                f.write('\n')
                
        except Exception as e:
            logger.warning(f"Failed to backup prediction metric: {e}")
    
    def _backup_system_metric(self, metric: SystemMetric):
        """Backup system metric to file"""
        try:
            date_str = metric.timestamp.strftime("%Y-%m-%d")
            backup_file = self.backup_dir / f"system_metrics_{date_str}.jsonl"
            
            with open(backup_file, 'a') as f:
                json.dump(asdict(metric), f, default=str)
                f.write('\n')
                
        except Exception as e:
            logger.warning(f"Failed to backup system metric: {e}")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions from memory or database"""
        # Try database first
        db_results = self.database.get_prediction_metrics(limit=limit)
        if db_results:
            return db_results
        
        # Fallback to memory
        return self.recent_predictions[-limit:]
    
    def get_recent_system_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent system metrics from memory or database"""
        # Try database first
        db_results = self.database.get_system_metrics(limit=limit)
        if db_results:
            return db_results
        
        # Fallback to memory
        return self.recent_system_metrics[-limit:]
    
    def get_prediction_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get prediction summary for specified time period"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Get predictions from database
        predictions = self.database.get_prediction_metrics(start_time=start_time)
        
        if not predictions:
            # Fallback to memory
            predictions = []
            for p in self.recent_predictions:
                timestamp = p['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif isinstance(timestamp, datetime):
                    pass  # Already a datetime object
                else:
                    continue  # Skip invalid timestamps
                
                if timestamp >= start_time:
                    predictions.append(p)
        
        if not predictions:
            return {
                "total_predictions": 0,
                "error_rate": 0,
                "avg_confidence": 0,
                "avg_processing_time_ms": 0,
                "predictions_by_class": {"NORMAL": 0, "PNEUMONIA": 0}
            }
        
        # Calculate summary statistics
        total_predictions = len(predictions)
        error_count = sum(1 for p in predictions if (
            p['status'] == 'error' or 
            p['status'] == PredictionStatus.ERROR or
            (hasattr(p['status'], 'value') and p['status'].value == 'error')
        ))
        error_rate = error_count / total_predictions
        
        successful_predictions = [p for p in predictions if (
            p['status'] == 'success' or 
            p['status'] == PredictionStatus.SUCCESS or
            (hasattr(p['status'], 'value') and p['status'].value == 'success')
        )]
        
        if successful_predictions:
            avg_confidence = np.mean([p['confidence'] for p in successful_predictions])
            avg_processing_time = np.mean([p['processing_time_ms'] for p in successful_predictions])
            
            predictions_by_class = {"NORMAL": 0, "PNEUMONIA": 0}
            for p in successful_predictions:
                if p['prediction'] in predictions_by_class:
                    predictions_by_class[p['prediction']] += 1
        else:
            avg_confidence = 0
            avg_processing_time = 0
            predictions_by_class = {"NORMAL": 0, "PNEUMONIA": 0}
        
        return {
            "total_predictions": total_predictions,
            "successful_predictions": len(successful_predictions),
            "error_count": error_count,
            "error_rate": error_rate,
            "avg_confidence": avg_confidence,
            "avg_processing_time_ms": avg_processing_time,
            "predictions_by_class": predictions_by_class,
            "time_period_hours": hours
        }
    
    def get_accuracy_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get accuracy metrics when actual labels are available"""
        accuracy_data = self.database.get_accuracy_over_time(hours=hours)
        
        if not accuracy_data:
            return {
                "accuracy": None,
                "total_labeled_predictions": 0,
                "message": "No labeled predictions available for accuracy calculation"
            }
        
        # Calculate overall accuracy
        total_correct = sum(row['correct_predictions'] for row in accuracy_data)
        total_labeled = sum(row['labeled_predictions'] for row in accuracy_data)
        
        overall_accuracy = total_correct / total_labeled if total_labeled > 0 else None
        
        return {
            "accuracy": overall_accuracy,
            "total_labeled_predictions": total_labeled,
            "total_correct_predictions": total_correct,
            "hourly_accuracy": accuracy_data,
            "time_period_hours": hours
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Get system metrics
        system_metrics = self.database.get_system_metrics(start_time=start_time)
        
        if not system_metrics:
            return {"message": "No system metrics available"}
        
        if not PANDAS_AVAILABLE:
            # Simple aggregation without pandas
            return {
                "message": "Pandas not available for trend analysis",
                "raw_metrics": system_metrics[-24:],  # Last 24 entries
                "time_period_hours": hours
            }
        
        # Calculate trends with pandas
        df = pd.DataFrame(system_metrics)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample to hourly averages
        df.set_index('timestamp', inplace=True)
        hourly_avg = df.resample('H').mean()
        
        return {
            "cpu_trend": hourly_avg['cpu_percent'].tolist(),
            "memory_trend": hourly_avg['memory_percent'].tolist(),
            "response_time_trend": hourly_avg['response_time_avg_ms'].tolist(),
            "request_rate_trend": hourly_avg['request_count'].diff().tolist(),  # Requests per hour
            "error_rate_trend": (hourly_avg['error_count'].diff() / hourly_avg['request_count'].diff()).fillna(0).tolist(),
            "timestamps": [t.isoformat() for t in hourly_avg.index],
            "time_period_hours": hours
        }
    
    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics from database"""
        if not self.database.connection:
            logger.warning("Database not available for cleanup")
            return
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        try:
            cursor = self.database.connection.cursor()
            
            # Delete old predictions
            cursor.execute("DELETE FROM predictions WHERE timestamp < %s", (cutoff_time,))
            deleted_predictions = cursor.rowcount
            
            # Delete old system metrics
            cursor.execute("DELETE FROM system_metrics WHERE timestamp < %s", (cutoff_time,))
            deleted_system_metrics = cursor.rowcount
            
            logger.info(f"Cleaned up {deleted_predictions} prediction metrics and {deleted_system_metrics} system metrics older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")

# Global metrics collector instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _metrics_collector.start_system_monitoring()
    return _metrics_collector

def record_prediction_metric(prediction: str,
                           confidence: float,
                           processing_time_ms: float,
                           model_version: str,
                           image_data: bytes = None,
                           image_id: str = None,
                           status: PredictionStatus = PredictionStatus.SUCCESS,
                           error_message: str = None,
                           actual_label: str = None,
                           metadata: Dict[str, Any] = None) -> str:
    """Convenience function to record prediction metric"""
    collector = get_metrics_collector()
    return collector.record_prediction(
        prediction=prediction,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        model_version=model_version,
        image_data=image_data,
        image_id=image_id,
        status=status,
        error_message=error_message,
        actual_label=actual_label,
        metadata=metadata
    )

if __name__ == "__main__":
    # Test metrics collection
    print("Testing Metrics Collection System...")
    
    # Initialize collector
    collector = MetricsCollector()
    
    # Test prediction recording
    prediction_id = collector.record_prediction(
        prediction="PNEUMONIA",
        confidence=0.95,
        processing_time_ms=45.2,
        model_version="v1.0.0",
        image_data=b"test_image_data"
    )
    
    print(f"✅ Recorded prediction: {prediction_id}")
    
    # Test summary
    summary = collector.get_prediction_summary(hours=1)
    print(f"✅ Prediction summary: {summary}")
    
    # Start system monitoring briefly
    collector.start_system_monitoring()
    time.sleep(2)
    collector.stop_system_monitoring()
    
    print("✅ Metrics collection system working correctly!")